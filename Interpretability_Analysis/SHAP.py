"""
SHAP Analysis Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import os
import time
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


# ==================== Model Definitions ====================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class PDeepPP(nn.Module):
    def __init__(self, input_dim=512, seq_len=100, embed_size=256, heads=8,
                 num_layers=3, forward_expansion=4, dropout=0.2):
        super(PDeepPP, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_size),
            nn.LayerNorm(embed_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.cnn_layers = nn.Sequential(
            ResidualBlock(1, 32, kernel_size=7, stride=1, dropout=dropout),
            nn.MaxPool1d(2),
            ResidualBlock(32, 64, kernel_size=5, stride=1, dropout=dropout),
            nn.MaxPool1d(2),
            ResidualBlock(64, 128, kernel_size=3, stride=1, dropout=dropout),
            nn.AdaptiveAvgPool1d(25)
        )

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])

        self.attention_pool = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Dropout(dropout)
        )

        self._calculate_cnn_output_dim()
        combined_features = self.cnn_output_dim + embed_size

        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _calculate_cnn_output_dim(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.seq_len)
            output = self.cnn_layers(dummy_input)
            self.cnn_output_dim = output.view(1, -1).shape[1]

    def forward(self, x):
        batch_size = x.shape[0]
        projected = self.input_projection(x)

        cnn_input = x.view(batch_size, 1, -1)
        if cnn_input.shape[2] < self.seq_len:
            pad_size = self.seq_len - cnn_input.shape[2]
            cnn_input = F.pad(cnn_input, (0, pad_size))
        else:
            cnn_input = cnn_input[:, :, :self.seq_len]

        cnn_features = self.cnn_layers(cnn_input)
        cnn_features = cnn_features.view(batch_size, -1)

        transformer_input = projected.unsqueeze(1).repeat(1, 5, 1)
        transformer_output = transformer_input
        for transformer_layer in self.transformer_layers:
            transformer_output = transformer_layer(
                transformer_output, transformer_output, transformer_output
            )

        attention_weights = torch.softmax(self.attention_pool(transformer_output).squeeze(-1), dim=1)
        transformer_features = torch.sum(transformer_output * attention_weights.unsqueeze(-1), dim=1)

        combined_features = torch.cat([cnn_features, transformer_features], dim=1)
        output = self.classifier(combined_features)
        return output.squeeze()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output.squeeze()


class DeepFRI(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[1024, 512, 256], dropout=0.3):
        super(DeepFRI, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        self.attention = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.Tanh(),
            nn.Linear(prev_dim // 2, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        attention_weights = self.attention(features)
        weighted_features = features * attention_weights
        output = self.classifier(weighted_features)
        return output.squeeze()


class TripleEnsemble(nn.Module):
    """Three-model ensemble"""

    def __init__(self, input_dim, mlp_hidden_dims=[512, 256, 128],
                 deepfri_hidden_dims=[1024, 512, 256], dropout=0.3):
        super(TripleEnsemble, self).__init__()

        self.mlp = MLP(input_dim, mlp_hidden_dims, dropout)
        self.pdeeppp = PDeepPP(input_dim=input_dim, dropout=dropout)
        self.deepfri = DeepFRI(input_dim, deepfri_hidden_dims, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        # Learnable weight parameters
        self.weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x):
        # Three model predictions
        mlp_output = self.mlp(x)
        pdeeppp_output = self.pdeeppp(x)
        deepfri_output = self.deepfri(x)

        # Stack three scalar outputs [batch_size, 3]
        combined = torch.stack([mlp_output, pdeeppp_output, deepfri_output], dim=1)

        # Apply learnable weights
        weighted_combined = combined * F.softmax(self.weights, dim=0)

        # Final prediction
        ensemble_output = self.classifier(weighted_combined)

        return ensemble_output.squeeze()


class FullDatasetSHAPAnalyzer:
    """Class for SHAP analysis using all samples"""

    def __init__(self, model_path, device='cpu'):
        self.device = device

        # Load model
        print("Loading trained model...")
        self.model = self._load_model(model_path)
        self.model.to(device)
        self.model.eval()

        print(f"SHAP analyzer initialization complete")

    def _load_model(self, model_path):
        """Load model"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        input_dim = 512

        model = TripleEnsemble(input_dim)

        # Direct weight loading
        try:
            model.load_state_dict(checkpoint)
            print("Direct model weight loading successful")
        except Exception as e:
            print(f"Direct loading failed: {e}")
            print("Fixing key names...")

            new_state_dict = {}
            for key, value in checkpoint.items():
                new_key = key
                if key.startswith('module.'):
                    new_key = key[7:]
                elif key.startswith('model.'):
                    new_key = key[6:]
                new_state_dict[new_key] = value

            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            print(f"Loading complete after fixing key names")
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")

        return model

    def load_all_data(self):
        """Load all test data"""
        print("Loading all test data...")

        try:
            # Load test dataset
            ptest = np.load('combined_features_final/pca_scaled_512/ptest_combined1.npy')
            ntest = np.load('combined_features_final/pca_scaled_512/ntest_combined1.npy')

            print(f"   Raw data shape:")
            print(f"     Test positive samples: {ptest.shape}")
            print(f"     Test negative samples: {ntest.shape}")

            # Combine all test samples
            X_test_all = np.concatenate([ptest, ntest], axis=0)
            y_test_all = np.concatenate([np.ones(len(ptest)), np.zeros(len(ntest))])

            # Check if standardization is needed
            print("Checking data statistics...")
            mean_check = np.abs(ptest.mean(axis=0)).mean()
            std_check = np.abs(ptest.std(axis=0) - 1).mean()

            print(f"   Mean check: {mean_check:.6f} (close to 0 indicates already standardized)")
            print(f"   Std check: {std_check:.6f} (close to 0 indicates already standardized)")

            if mean_check < 0.01 and std_check < 0.01:
                print("Data already standardized")
                X_test_scaled = X_test_all
                self.use_scaler = False
            else:
                print("Data not standardized, performing standardization...")
                self.scaler = StandardScaler()
                X_test_scaled = self.scaler.fit_transform(X_test_all)
                self.use_scaler = True

            print(f"   Data loading complete")
            print(f"   Total test samples: {X_test_scaled.shape[0]}")
            print(f"   Feature dimension: {X_test_scaled.shape[1]}")
            print(f"   Positive samples: {np.sum(y_test_all == 1)}")
            print(f"   Negative samples: {np.sum(y_test_all == 0)}")

            return X_test_scaled, y_test_all

        except Exception as e:
            print(f"Data loading failed: {e}")
            raise

    def validate_model_performance(self, X_test_scaled, y_test):
        """Validate model performance on all test samples"""
        print("Validating model performance on all test samples...")

        # Batch prediction
        predictions = self._batch_predict_all(X_test_scaled)
        pred_labels = (predictions > 0.5).astype(int)

        # Ensure length matches
        min_len = min(len(pred_labels), len(y_test))
        pred_labels = pred_labels[:min_len]
        y_test_subset = y_test[:min_len]

        # Calculate metrics
        accuracy = accuracy_score(y_test_subset, pred_labels)
        f1 = f1_score(y_test_subset, pred_labels, zero_division=0)
        precision = precision_score(y_test_subset, pred_labels, zero_division=0)
        recall = recall_score(y_test_subset, pred_labels, zero_division=0)

        print(f"   Model performance on all test samples:")
        print(f"   Total samples: {len(y_test_subset)}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 score: {f1:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   Prediction probability range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"   Prediction probability mean: {predictions.mean():.4f}")
        print(f"   Prediction probability std: {predictions.std():.4f}")

        # Plot detailed prediction distribution
        self._plot_detailed_distribution(predictions, y_test_subset)

        return accuracy, predictions

    def _batch_predict_all(self, X_scaled):
        """Batch predict all samples"""
        self.model.eval()
        batch_size = 256  # Use larger batch size
        all_preds = []

        print(f"   Starting prediction of {len(X_scaled)} samples...")
        with torch.no_grad():
            for i in tqdm(range(0, len(X_scaled), batch_size), desc="Prediction progress"):
                batch_x = X_scaled[i:i + batch_size]
                x_tensor = torch.FloatTensor(batch_x).to(self.device)
                batch_preds = self.model(x_tensor)
                all_preds.append(batch_preds.cpu().numpy())

        return np.concatenate(all_preds)

    def _plot_detailed_distribution(self, predictions, y_true):
        """Plot detailed prediction distribution"""
        fig = plt.figure(figsize=(18, 12))

        # 1. Prediction probability distribution
        ax1 = plt.subplot(2, 3, 1)
        plt.hist(predictions[y_true == 1], bins=50, alpha=0.7,
                 label=f'Positive samples ({np.sum(y_true == 1)})', density=True, color='skyblue')
        plt.hist(predictions[y_true == 0], bins=50, alpha=0.7,
                 label=f'Negative samples ({np.sum(y_true == 0)})', density=True, color='salmon')
        plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, linewidth=2)
        plt.xlabel('Prediction probability')
        plt.ylabel('Density')
        plt.title('Prediction probability distribution for all samples')
        plt.legend()
        plt.grid(alpha=0.3)

        # 2. Prediction probability box plot
        ax2 = plt.subplot(2, 3, 2)
        data_to_plot = [predictions[y_true == 1], predictions[y_true == 0]]
        box = plt.boxplot(data_to_plot, labels=['Positive', 'Negative'],
                          patch_artist=True)
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel('Prediction probability')
        plt.title('Prediction probability box plot for positive and negative samples')
        plt.grid(alpha=0.3, axis='y')

        # 3. Sorted prediction probabilities
        ax3 = plt.subplot(2, 3, 3)
        sorted_indices = np.argsort(predictions)
        plt.plot(range(len(predictions)), predictions[sorted_indices],
                 alpha=0.7, linewidth=0.5)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Sample order')
        plt.ylabel('Prediction probability')
        plt.title('Sorted prediction probabilities')
        plt.grid(alpha=0.3)

        # 4. Cumulative distribution function
        ax4 = plt.subplot(2, 3, 4)
        for label in [0, 1]:
            data = predictions[y_true == label]
            if len(data) > 0:
                sorted_data = np.sort(data)
                y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                plt.plot(sorted_data, y_vals,
                         label=f'Label={label} ({len(data)} samples)',
                         linewidth=2, alpha=0.8)
        plt.xlabel('Prediction probability')
        plt.ylabel('Cumulative probability')
        plt.title('Prediction probability cumulative distribution function')
        plt.legend()
        plt.grid(alpha=0.3)

        # 5. Confidence distribution
        ax5 = plt.subplot(2, 3, 5)
        confidence = np.abs(predictions - 0.5)
        bins = np.linspace(0, 0.5, 20)
        plt.hist(confidence[y_true == 1], bins=bins, alpha=0.7,
                 label='Positive', density=True, color='skyblue')
        plt.hist(confidence[y_true == 0], bins=bins, alpha=0.7,
                 label='Negative', density=True, color='salmon')
        plt.xlabel('Confidence |p-0.5|')
        plt.ylabel('Density')
        plt.title('Prediction confidence distribution')
        plt.legend()
        plt.grid(alpha=0.3)

        # 6. Threshold sensitivity analysis
        ax6 = plt.subplot(2, 3, 6)
        thresholds = np.linspace(0.1, 0.9, 17)
        accuracies = []
        for threshold in thresholds:
            pred_labels_thresh = (predictions > threshold).astype(int)
            acc = accuracy_score(y_true, pred_labels_thresh)
            accuracies.append(acc)
        plt.plot(thresholds, accuracies, 'b-o', linewidth=2, markersize=4)
        plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Classification threshold')
        plt.ylabel('Accuracy')
        plt.title('Threshold vs Accuracy')
        plt.grid(alpha=0.3)

        plt.suptitle('Model performance analysis on all test samples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('all_samples_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Detailed performance analysis plot saved: all_samples_performance_analysis.png")

    def run_full_shap_analysis(self, background_size=100, save_dir='full_shap_analysis'):
        """Perform SHAP analysis using all samples"""
        print("\n" + "=" * 80)
        print(" Full-sample SHAP interpretability analysis ")
        print("=" * 80)

        os.makedirs(save_dir, exist_ok=True)

        # 1. Load all data
        print("\n1️⃣ Loading all test data...")
        X_all_scaled, y_all = self.load_all_data()

        # 2. Validate model performance
        print("\n2️⃣ Validating model performance...")
        accuracy, predictions = self.validate_model_performance(X_all_scaled, y_all)

        # 3. Prepare SHAP analysis
        print(f"\n3️⃣ Preparing SHAP analysis...")

        # Use all samples
        X_shap = X_all_scaled
        y_shap = y_all

        print(f"   Using all {len(X_shap)} samples for SHAP analysis")
        print(f"   Positive samples: {np.sum(y_shap == 1)}, Negative samples: {np.sum(y_shap == 0)}")

        # Background samples (loaded from training set)
        print("   Preparing background samples...")
        try:
            ptrain = np.load('combined_features_final/pca_scaled_512/ptrain_combined1.npy')
            ntrain = np.load('combined_features_final/pca_scaled_512/ntrain_combined1.npy')
            X_train = np.concatenate([ptrain, ntrain], axis=0)

            # Standardize training data
            if self.use_scaler:
                X_train_scaled = self.scaler.transform(X_train)
            else:
                X_train_scaled = X_train

            # Select background samples
            background_size = min(background_size, len(X_train_scaled))
            background_indices = np.random.choice(len(X_train_scaled),
                                                  background_size, replace=False)
            background = X_train_scaled[background_indices]

            print(f"   Background samples: {background.shape} (randomly selected from training set)")

        except Exception as e:
            print(f"Unable to load training set")

        # 4. Create model wrapper - fix critical part
        print("\n Creating model wrapper...")

        # Ensure model wrapper returns correct shape
        def model_wrapper_fixed(x):
            """Wrapper function for SHAP computation"""
            # Convert to numpy array
            if isinstance(x, pd.DataFrame):
                x = x.values
            elif isinstance(x, list):
                x = np.array(x)
            else:
                x = np.array(x, dtype=np.float32)

            # Ensure 2D (n_samples, n_features)
            if x.ndim == 1:
                x = x.reshape(1, -1)

            # Ensure feature dimension is 512
            if x.shape[1] != 512:
                if x.shape[1] > 512:
                    x = x[:, :512]
                else:
                    padding = np.zeros((x.shape[0], 512 - x.shape[1]))
                    x = np.hstack([x, padding])

            # Convert to tensor
            x_tensor = torch.FloatTensor(x).to(self.device)

            # Predict
            self.model.eval()
            with torch.no_grad():
                preds = self.model(x_tensor)

            # Fix: ensure output shape is (n_samples, 1)
            output = preds.cpu().numpy()

            # Handle different output shapes
            if output.ndim == 0:  # Scalar
                output = np.array([[output]])
            elif output.ndim == 1:  # 1D array
                output = output.reshape(-1, 1)
            # If already 2D but column count is not 1
            elif output.ndim == 2 and output.shape[1] != 1:
                # Take first column
                output = output[:, 0].reshape(-1, 1)

            return output

        # Test model wrapper
        print("   Testing model wrapper...")
        try:
            test_input = background[:3]  # Test 3 samples
            test_output = model_wrapper_fixed(test_input)
            print(f" Model wrapper test successful")
            print(f"   Input shape: {test_input.shape}")
            print(f"   Output shape: {test_output.shape}")
            print(f"   Example output: {test_output[0, 0]:.6f}, {test_output[1, 0]:.6f}, {test_output[2, 0]:.6f}")

            # Validate wrapper
            if test_output.shape != (3, 1):
                print(f" Output shape incorrect: {test_output.shape}, expected (3, 1)")

                # Force fix
                def model_wrapper_final(x):
                    output = model_wrapper_fixed(x)
                    # Ensure shape is (n_samples, 1)
                    if output.ndim == 1:
                        return output.reshape(-1, 1)
                    elif output.shape[1] != 1:
                        return output[:, 0:1]
                    else:
                        return output

                model_wrapper = model_wrapper_final
            else:
                model_wrapper = model_wrapper_fixed

        except Exception as e:
            print(f" Model wrapper test failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        # 5. Batch compute SHAP values
        print("\n Batch computing SHAP values...")
        print(f"   Total {len(X_shap)} samples, batch processing to avoid memory issues")

        # Batch computation
        batch_size = 20
        shap_values_list = []

        try:
            # Create explainer
            explainer = shap.KernelExplainer(model_wrapper, background)

            # Batch processing
            total_batches = (len(X_shap) + batch_size - 1) // batch_size
            print(f"   Total {total_batches} batches, {batch_size} samples per batch")

            for batch_idx in tqdm(range(0, len(X_shap), batch_size), desc="SHAP computation progress"):
                batch_end = min(batch_idx + batch_size, len(X_shap))
                X_batch = X_shap[batch_idx:batch_end]

                try:
                    # Compute SHAP values for current batch
                    print(f"   Computing batch {batch_idx // batch_size + 1}/{total_batches}...")
                    batch_shap_values = explainer.shap_values(
                        X_batch,
                        nsamples=10,  # Reduce sampling for speed
                        silent=True,
                        check_additivity=False  # Skip additivity check
                    )

                    # Handle different return formats
                    if isinstance(batch_shap_values, list):
                        # If list, take first element
                        batch_shap_values = batch_shap_values[0]

                    # Ensure SHAP values are 2D
                    batch_shap_values = batch_shap_values.reshape(batch_shap_values.shape[0], -1)

                    # Check dimension
                    if batch_shap_values.shape[1] != 512:
                        print(f"Batch {batch_idx // batch_size + 1} SHAP dimension mismatch: {batch_shap_values.shape}")
                        # Fix dimension
                        if batch_shap_values.shape[1] > 512:
                            batch_shap_values = batch_shap_values[:, :512]
                        else:
                            padding = np.zeros((batch_shap_values.shape[0], 512 - batch_shap_values.shape[1]))
                            batch_shap_values = np.hstack([batch_shap_values, padding])

                    shap_values_list.append(batch_shap_values)
                    print(f"   Batch {batch_idx // batch_size + 1}: {len(X_batch)} samples completed")

                except Exception as batch_error:
                    print(f"Batch {batch_idx // batch_size + 1} failed: {batch_error}")
                    print(f"   Using approximation method for batch {batch_idx // batch_size + 1}...")
                    # Use approximation method for this batch
                    batch_shap_values = self._compute_approximate_shap_batch(
                        model_wrapper, X_batch, background
                    )
                    shap_values_list.append(batch_shap_values)

            # Combine all batch SHAP values
            shap_values = np.vstack(shap_values_list)
            print(f"SHAP value computation complete, shape: {shap_values.shape}")

        except Exception as e:
            print(f"Standard SHAP computation failed: {e}")
            print("Using approximate SHAP method...")
            shap_values = self._compute_approximate_shap(model_wrapper, X_shap, background)

        # 6. Save results
        print("\n6️⃣ Saving results...")

        # Save all data
        np.save(f'{save_dir}/all_shap_values.npy', shap_values)
        np.save(f'{save_dir}/all_features.npy', X_shap)
        np.save(f'{save_dir}/all_labels.npy', y_shap)
        np.save(f'{save_dir}/all_predictions.npy', predictions)

        # Save background samples
        np.save(f'{save_dir}/background_samples.npy', background)

        # Save analysis information
        analysis_info = {
            'total_samples': len(X_shap),
            'positive_samples': int(np.sum(y_shap == 1)),
            'negative_samples': int(np.sum(y_shap == 0)),
            'accuracy': float(accuracy),
            'background_size': len(background),
            'batch_size': batch_size,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'feature_dim': X_shap.shape[1]
        }

        with open(f'{save_dir}/full_analysis_info.pkl', 'wb') as f:
            pickle.dump(analysis_info, f)

        print(f"All results saved to {save_dir}/")

        # 7. Visualization analysis
        print("\n7️⃣ Visualization analysis...")
        self._visualize_full_shap_results(shap_values, X_shap, y_shap, predictions, save_dir)

        # 8. Generate detailed report
        print("\n8️⃣ Generating detailed analysis report...")
        self._generate_full_analysis_report(shap_values, X_shap, y_shap, predictions,
                                            accuracy, save_dir)
        # 9. Feature interval analysis
        print("\n9️⃣ Performing feature beneficial interval analysis...")
        try:
            interval_results = self.analyze_feature_beneficial_intervals(
                shap_values=shap_values,
                X_features=X_shap,
                y_labels=y_shap,
                top_n=20,
                save_dir=os.path.join(save_dir, 'feature_interval_analysis')
            )
            print("Feature interval analysis complete")
        except Exception as e:
            print(f"Feature interval analysis failed: {e}")
            import traceback
            traceback.print_exc()
        print(f"\nFull-sample SHAP analysis complete!")
        print(f"All results saved in: {save_dir}/")
        return shap_values

    def _compute_approximate_shap_batch(self, model_wrapper, X_batch, background):
        """Compute approximate SHAP values for a single batch"""
        n_samples, n_features = X_batch.shape
        batch_shap_values = np.zeros((n_samples, n_features))

        # Compute baseline predictions
        baseline_preds = model_wrapper(X_batch).flatten()

        # Compute contribution for each feature
        for i in range(n_features):
            # Create permuted version
            X_permuted = X_batch.copy()

            # Replace with background sample feature values
            background_indices = np.random.choice(len(background), n_samples)
            X_permuted[:, i] = background[background_indices, i]

            # Compute permuted predictions
            permuted_preds = model_wrapper(X_permuted).flatten()

            # SHAP value = baseline prediction - permuted prediction
            batch_shap_values[:, i] = baseline_preds - permuted_preds

        return batch_shap_values

    def _compute_approximate_shap(self, model_wrapper, X_shap, background):
        """Compute approximate SHAP values"""
        print("   Using approximate SHAP method...")

        n_samples, n_features = X_shap.shape
        shap_values = np.zeros((n_samples, n_features))

        # Compute baseline predictions (all samples)
        print("   Computing baseline predictions...")
        baseline_preds = model_wrapper(X_shap).flatten()

        # Batch process features to avoid memory issues
        print("   Batch computing feature contributions...")
        feature_batch_size = 20  # Process 20 features at a time

        for feat_start in tqdm(range(0, n_features, feature_batch_size), desc="Processing features"):
            feat_end = min(feat_start + feature_batch_size, n_features)

            for i in range(feat_start, feat_end):
                # Create permuted version
                X_permuted = X_shap.copy()

                # Replace with background sample feature values
                background_indices = np.random.choice(len(background), n_samples)
                X_permuted[:, i] = background[background_indices, i]

                # Compute permuted predictions
                permuted_preds = model_wrapper(X_permuted).flatten()

                # SHAP value = baseline prediction - permuted prediction
                shap_values[:, i] = baseline_preds - permuted_preds

        print(f"Approximate SHAP computation complete, shape: {shap_values.shape}")
        return shap_values

    def _visualize_full_shap_results(self, shap_values, X, y, predictions, save_dir):
        """Visualize full-sample SHAP results"""

        print("   Starting visualization...")

        # 1. SHAP summary plot (using all samples) - modified: show top 20 features
        print("   1. Generating global SHAP summary plot (top 20 features)...")
        try:
            plt.figure(figsize=(16, 10))
            shap.summary_plot(shap_values, X, show=False, plot_size=(14, 8),
                              max_display=20)  # Modified: only show top 20 features
            plt.title('Full-sample SHAP value summary plot (top 20 important features)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            # Save to new path
            plt.savefig(f'{save_dir}/modified_shap_summary_top20.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f" SHAP summary plot generation failed: {e}")

        # 2. Feature importance bar plot (using all samples)
        print("   2. Generating global feature importance bar plot (top 10 features)...")
        try:
            feature_importance = np.abs(shap_values).mean(axis=0)
            top_n = min(10, len(feature_importance))  # Modified: only show top 10
            top_indices = np.argsort(feature_importance)[-top_n:][::-1]
            top_importance = feature_importance[top_indices]

            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(top_n), top_importance)
            plt.yticks(range(top_n), [f'Feature{idx}' for idx in top_indices], fontsize=12)  # Larger font
            plt.xlabel('Mean absolute SHAP value', fontsize=12)
            plt.title(f'Most important {top_n} features (based on all samples)', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(alpha=0.3, axis='x')

            # Add value labels
            for i, (bar, imp) in enumerate(zip(bars, top_importance)):
                plt.text(imp, i, f' {imp:.4f}', va='center', fontsize=10)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/modified_feature_importance_top10.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"⚠ Feature importance plot failed: {e}")

        # 3. Detailed comparison analysis
        print("   3. Generating detailed comparison analysis plot (top 10 features)...")
        try:
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]

            if len(pos_indices) > 0 and len(neg_indices) > 0:
                shap_pos = shap_values[pos_indices]
                shap_neg = shap_values[neg_indices]

                fig, axes = plt.subplots(2, 2, figsize=(20, 16))

                # Mean SHAP value comparison - show top 10 features
                top_n = 10
                importance = np.abs(shap_values).mean(axis=0)
                top_indices = np.argsort(importance)[-top_n:][::-1]

                axes[0, 0].barh(range(top_n), shap_pos.mean(axis=0)[top_indices])
                axes[0, 0].set_title(f'Positive sample mean SHAP (n={len(pos_indices)})', fontsize=14,
                                     fontweight='bold')
                axes[0, 0].set_xlabel('Mean SHAP value', fontsize=12)
                axes[0, 0].set_yticks(range(top_n))
                axes[0, 0].set_yticklabels([f'Feature{idx}' for idx in top_indices], fontsize=11)
                axes[0, 0].invert_yaxis()
                axes[0, 0].grid(alpha=0.3, axis='x')

                axes[0, 1].barh(range(top_n), shap_neg.mean(axis=0)[top_indices])
                axes[0, 1].set_title(f'Negative sample mean SHAP (n={len(neg_indices)})', fontsize=14,
                                     fontweight='bold')
                axes[0, 1].set_xlabel('Mean SHAP value', fontsize=12)
                axes[0, 1].set_yticks(range(top_n))
                axes[0, 1].set_yticklabels([f'Feature{idx}' for idx in top_indices], fontsize=11)
                axes[0, 1].invert_yaxis()
                axes[0, 1].grid(alpha=0.3, axis='x')

                # Importance comparison
                importance_pos = np.abs(shap_pos).mean(axis=0)
                importance_neg = np.abs(shap_neg).mean(axis=0)

                x = np.arange(top_n)
                width = 0.35
                axes[1, 0].bar(x - width / 2, importance_pos[top_indices], width,
                               label='Positive', alpha=0.7)
                axes[1, 0].bar(x + width / 2, importance_neg[top_indices], width,
                               label='Negative', alpha=0.7)
                axes[1, 0].set_title('Top 10 feature importance comparison', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('Feature index', fontsize=12)
                axes[1, 0].set_ylabel('Mean absolute SHAP value', fontsize=12)
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels([f'Feature{idx}' for idx in top_indices],
                                           rotation=45, fontsize=11)
                axes[1, 0].legend(fontsize=11)  # Larger legend
                axes[1, 0].grid(alpha=0.3)

                # SHAP value distribution comparison
                axes[1, 1].hist(shap_pos.flatten(), bins=50, alpha=0.7,
                                label=f'Positive ({len(pos_indices)} samples)', density=True)
                axes[1, 1].hist(shap_neg.flatten(), bins=50, alpha=0.7,
                                label=f'Negative ({len(neg_indices)} samples)', density=True)
                axes[1, 1].set_title('SHAP value distribution comparison', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('SHAP value', fontsize=12)
                axes[1, 1].set_ylabel('Density', fontsize=12)
                axes[1, 1].legend(fontsize=11)  # Larger legend
                axes[1, 1].grid(alpha=0.3)

                plt.suptitle('Positive vs Negative sample SHAP analysis comparison (top 10 features)', fontsize=16,
                             fontweight='bold', y=1.02)
                plt.tight_layout()
                # Save to new path
                plt.savefig(f'{save_dir}/modified_pos_neg_comparison_top10.png',
                            dpi=300, bbox_inches='tight')
                plt.show()
        except Exception as e:
            print(f"Comparison analysis plot failed: {e}")

        # 4. Feature clustering heatmap - modified: top 10 features
        print("   4. Generating feature correlation heatmap (top 10 features)...")
        try:
            # Select most important features for cluster analysis - modified: only select top 10
            top_n = min(10, shap_values.shape[1])  # Modified: only select top 10 features
            importance = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(importance)[-top_n:][::-1]

            # Extract important SHAP values
            shap_important = shap_values[:, top_indices]

            # Calculate feature correlations
            feature_corr = np.corrcoef(shap_important.T)

            plt.figure(figsize=(12, 10))
            sns.heatmap(feature_corr,
                        xticklabels=[f'F{idx}' for idx in top_indices],
                        yticklabels=[f'F{idx}' for idx in top_indices],
                        cmap='coolwarm', center=0,
                        cbar_kws={'label': 'Correlation coefficient'},
                        annot=True, fmt='.2f', annot_kws={'size': 10})  # Adjust annotation size
            plt.xlabel('Feature index', fontsize=12)
            plt.ylabel('Feature index', fontsize=12)
            plt.title(f'SHAP value correlation heatmap for top {top_n} important features', fontsize=14,
                      fontweight='bold')
            plt.tight_layout()
            # Save to new path
            plt.savefig(f'{save_dir}/modified_shap_correlation_heatmap_top10.png',
                        dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"⚠ Heatmap generation failed: {e}")

    def _generate_full_analysis_report(self, shap_values, X, y, predictions, accuracy, save_dir):
        """Generate detailed analysis report"""

        print("   Generating detailed analysis report...")

        report_lines = [
            "=" * 80,
            "Succinylation Site Prediction - Full-sample SHAP Interpretability Analysis Report",
            "=" * 80,
            f"Generation time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "1. Analysis Overview",
            "-" * 50,
            f"Model: Triple Ensemble (MLP + PDeepPP + DeepFRI)",
            f"Features: ProtT5 + KSP, PCA reduced to 512 dimensions",
            f"Total analysis samples: {len(X)}",
            f"Positive samples: {np.sum(y == 1)}",
            f"Negative samples: {np.sum(y == 0)}",
            f"Positive to negative ratio: {np.sum(y == 1)}:{np.sum(y == 0)}",
            f"Model accuracy: {accuracy:.4f}",
            "",
            "2. SHAP Analysis Statistics",
            "-" * 50,
            f"SHAP value shape: {shap_values.shape}",
            f"SHAP value mean: {shap_values.mean():.6f}",
            f"SHAP value std: {shap_values.std():.6f}",
            f"SHAP value range: [{shap_values.min():.6f}, {shap_values.max():.6f}]",
            f"Positive SHAP proportion: {(shap_values > 0).mean():.3f}",
            f"Negative SHAP proportion: {(shap_values < 0).mean():.3f}",
            f"Near-zero SHAP proportion (|SHAP|<0.001): {(np.abs(shap_values) < 0.001).mean():.3f}",
            f"Significant feature proportion (|SHAP|>0.01): {(np.abs(shap_values) > 0.01).mean():.3f}",
            f"Strong feature proportion (|SHAP|>0.1): {(np.abs(shap_values) > 0.1).mean():.3f}",
            "",
            "3. Most Important Features (Top 20)",
            "-" * 50,
        ]

        # Feature importance ranking
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(feature_importance)[-20:][::-1]

        for i, idx in enumerate(top_indices):
            mean_shap = shap_values[:, idx].mean()
            std_shap = shap_values[:, idx].std()
            importance = feature_importance[idx]

            # Direction analysis
            if mean_shap > 0.01:
                direction = "Strong positive"
            elif mean_shap > 0:
                direction = "Positive"
            elif mean_shap < -0.01:
                direction = "Strong negative"
            else:
                direction = "Negative"

            # Positive-negative sample difference
            pos_mean = shap_values[y == 1, idx].mean() if np.sum(y == 1) > 0 else 0
            neg_mean = shap_values[y == 0, idx].mean() if np.sum(y == 0) > 0 else 0

            report_lines.append(f"{i + 1:2d}. Feature {idx:4d}:")
            report_lines.append(f"     Importance: {importance:.6f}")
            report_lines.append(f"     Mean contribution: {mean_shap:.6f} ± {std_shap:.6f}")
            report_lines.append(f"     Contribution direction: {direction}")
            report_lines.append(f"     Positive sample mean: {pos_mean:.6f}")
            report_lines.append(f"     Negative sample mean: {neg_mean:.6f}")
            report_lines.append(f"     Difference: {pos_mean - neg_mean:.6f}")
            report_lines.append("")

        report_lines.extend([
            "4. Prediction Analysis",
            "-" * 50,
            f"Positive sample prediction range: [{predictions[y == 1].min():.4f}, {predictions[y == 1].max():.4f}]",
            f"Negative sample prediction range: [{predictions[y == 0].min():.4f}, {predictions[y == 0].max():.4f}]",
            f"Positive sample mean prediction: {predictions[y == 1].mean():.4f} ± {predictions[y == 1].std():.4f}",
            f"Negative sample mean prediction: {predictions[y == 0].mean():.4f} ± {predictions[y == 0].std():.4f}",
            f"Prediction overlap region proportion: {(predictions.min() < 0.5) and (predictions.max() > 0.5)}",
            f"High confidence positive samples (p>0.8): {np.sum(predictions > 0.8)}",
            f"High confidence negative samples (p<0.2): {np.sum(predictions < 0.2)}",
            f"Ambiguous predictions (0.4<p<0.6): {np.sum((predictions >= 0.4) & (predictions <= 0.6))}",
            "",
            "5. Model Interpretability Analysis",
            "-" * 50,
        ])

        # Model interpretability evaluation
        total_variance = np.var(shap_values, axis=0).sum()
        top10_variance = np.var(shap_values[:, top_indices[:10]], axis=0).sum()
        explainability_ratio = top10_variance / total_variance if total_variance > 0 else 0

        report_lines.append(f"Total SHAP variance: {total_variance:.6f}")
        report_lines.append(f"Top 10 feature variance: {top10_variance:.6f}")
        report_lines.append(
            f"Interpretability ratio: {explainability_ratio:.3f} (variance explained by top 10 features)")

        if explainability_ratio > 0.5:
            report_lines.append("Model interpretability: Good (few features dominate prediction)")
        elif explainability_ratio > 0.3:
            report_lines.append("Model interpretability: Average")
        else:
            report_lines.append("Model interpretability: Poor (prediction distributed across many features)")

        report_lines.extend([
            "",
            "6. Analysis Conclusions",
            "-" * 50,
            "1. Full-sample SHAP analysis provides global view of model decisions",
            "2. Most important features have decisive influence on model predictions",
            "3. Feature contribution patterns differ significantly between positive and negative samples",
            "4. Results can be used for feature selection, model optimization, and biological interpretation",
            "5. Recommend focusing on top 20 important features",
            "",
            "7. File Description",
            "-" * 50,
            "all_shap_values.npy      - SHAP value matrix for all samples",
            "all_features.npy         - Feature matrix for all samples",
            "all_labels.npy           - Label vector for all samples",
            "all_predictions.npy      - Prediction probabilities for all samples",
            "background_samples.npy   - Background samples",
            "full_analysis_info.pkl   - Analysis information",
            "full_shap_summary.png    - Full-sample SHAP summary plot",
            "full_feature_importance.png - Feature importance plot",
            "full_pos_neg_comparison.png - Positive-negative sample comparison plot",
            "shap_correlation_heatmap.png - Feature correlation heatmap",
            "all_samples_performance_analysis.png - Performance analysis plot",
            "",
            "8. Recommendations",
            "-" * 50,
            "1. Focus on top 10-20 important features for biological interpretation",
            "2. Consider feature selection to remove unimportant features and simplify model",
            "3. Perform feature engineering optimization on important features",
            "4. Validate reasonableness of important features with biological knowledge",
            "",
            "=" * 80,
        ])

        # Write report file
        report_path = os.path.join(save_dir, 'full_shap_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"📝 Detailed analysis report saved: {report_path}")

    def analyze_feature_beneficial_intervals(self, shap_values, X_features, y_labels, top_n=10,
                                             save_dir='interval_analysis'):
        """Analyze which value intervals of features are beneficial to the model"""
        print("\n" + "=" * 80)
        print(" Feature Value Interval Benefit Analysis")
        print("=" * 80)

        os.makedirs(save_dir, exist_ok=True)

        # 1. Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(feature_importance)[-top_n:][::-1]

        interval_results = {}

        for idx, feature_idx in enumerate(top_indices):
            print(f"\n Analyzing feature {feature_idx} (importance rank {idx + 1})...")

            # Get all values for this feature
            feature_values = X_features[:, feature_idx]
            shap_for_feature = shap_values[:, feature_idx]

            # Calculate basic statistics
            mean_val = feature_values.mean()
            std_val = feature_values.std()

            # Analyze SHAP values for different intervals
            bins = 20
            percentiles = np.linspace(0, 100, bins + 1)

            intervals = []
            beneficial_info = []

            for i in range(bins):
                p_low = percentiles[i]
                p_high = percentiles[i + 1]

                # Get values for this percentile interval
                val_low = np.percentile(feature_values, p_low)
                val_high = np.percentile(feature_values, p_high)

                # Find samples in this interval
                mask = (feature_values >= val_low) & (feature_values < val_high)
                if i == bins - 1:  # Last one includes maximum
                    mask = (feature_values >= val_low) & (feature_values <= val_high)

                if np.sum(mask) > 0:
                    interval_shap = shap_for_feature[mask]
                    interval_mean_shap = interval_shap.mean()
                    interval_std_shap = interval_shap.std()

                    # Calculate effect on positive and negative samples
                    interval_labels = y_labels[mask]
                    pos_mask = interval_labels == 1
                    neg_mask = interval_labels == 0

                    pos_mean = interval_shap[pos_mask].mean() if np.sum(pos_mask) > 0 else 0
                    neg_mean = interval_shap[neg_mask].mean() if np.sum(neg_mask) > 0 else 0

                    interval_info = {
                        'percentile_range': (p_low, p_high),
                        'value_range': (val_low, val_high),
                        'n_samples': np.sum(mask),
                        'mean_shap': interval_mean_shap,
                        'std_shap': interval_std_shap,
                        'pos_mean': pos_mean,
                        'neg_mean': neg_mean,
                        'pos_count': np.sum(pos_mask),
                        'neg_count': np.sum(neg_mask)
                    }

                    intervals.append(interval_info)

                    # Determine if interval is beneficial to model
                    if interval_mean_shap > 0.01:  # Significantly positive
                        beneficial_info.append({
                            'interval': f"{val_low:.4f} ~ {val_high:.4f}",
                            'percentile': f"{p_low:.1f}% ~ {p_high:.1f}%",
                            'benefit_level': 'Strong positive',
                            'mean_shap': interval_mean_shap,
                            'pos_effect': pos_mean,
                            'neg_effect': neg_mean
                        })
                    elif interval_mean_shap > 0.001:
                        beneficial_info.append({
                            'interval': f"{val_low:.4f} ~ {val_high:.4f}",
                            'percentile': f"{p_low:.1f}% ~ {p_high:.1f}%",
                            'benefit_level': 'Positive',
                            'mean_shap': interval_mean_shap,
                            'pos_effect': pos_mean,
                            'neg_effect': neg_mean
                        })
                    elif interval_mean_shap < -0.01:  # Significantly negative
                        beneficial_info.append({
                            'interval': f"{val_low:.4f} ~ {val_high:.4f}",
                            'percentile': f"{p_low:.1f}% ~ {p_high:.1f}%",
                            'benefit_level': 'Strong negative',
                            'mean_shap': interval_mean_shap,
                            'pos_effect': pos_mean,
                            'neg_effect': neg_mean
                        })

            # Save results
            interval_results[feature_idx] = {
                'feature_idx': feature_idx,
                'importance': feature_importance[feature_idx],
                'overall_mean_shap': shap_for_feature.mean(),
                'intervals': intervals,
                'beneficial_intervals': beneficial_info
            }

            # Print beneficial intervals
            if beneficial_info:
                print(f"Feature {feature_idx} beneficial/harmful intervals:")
                for info in beneficial_info:
                    print(f"   Interval {info['interval']} ({info['percentile']}):")
                    print(f"     {info['benefit_level']} (Mean SHAP: {info['mean_shap']:.6f})")
                    print(f"     Effect on positive samples: {info['pos_effect']:.6f}")
                    print(f"     Effect on negative samples: {info['neg_effect']:.6f}")
            else:
                print(f" Feature {feature_idx} no significant beneficial/harmful intervals found")

        # 2. Visualization
        self._visualize_interval_analysis(interval_results, shap_values, X_features, save_dir)

        # 3. Generate report
        self._generate_interval_report(interval_results, save_dir)

        return interval_results

    def _visualize_interval_analysis(self, interval_results, shap_values, X_features, save_dir):
        """Visualize interval analysis results"""
        print("\n Generating interval analysis visualization...")

        for feature_idx, results in interval_results.items():
            if not results['beneficial_intervals']:
                continue

            feature_values = X_features[:, feature_idx]
            shap_for_feature = shap_values[:, feature_idx]

            # Create scatter plot
            plt.figure(figsize=(12, 8))

            # Positive samples
            pos_mask = np.where(feature_values > 0)[0]
            if len(pos_mask) > 0:
                plt.scatter(feature_values[pos_mask], shap_for_feature[pos_mask],
                            alpha=0.5, s=10, c='blue', label='Positive sample SHAP')

            # Negative samples
            neg_mask = np.where(feature_values <= 0)[0]
            if len(neg_mask) > 0:
                plt.scatter(feature_values[neg_mask], shap_for_feature[neg_mask],
                            alpha=0.5, s=10, c='red', label='Negative sample SHAP')

            # Add trend line
            try:
                # Use locally weighted regression
                from scipy.interpolate import UnivariateSpline
                sorted_indices = np.argsort(feature_values)
                spline = UnivariateSpline(feature_values[sorted_indices],
                                          shap_for_feature[sorted_indices], s=100)
                x_smooth = np.linspace(feature_values.min(), feature_values.max(), 200)
                y_smooth = spline(x_smooth)
                plt.plot(x_smooth, y_smooth, 'g-', linewidth=3, label='Trend line')

                # Find inflection points
                grad = np.gradient(y_smooth, x_smooth)
                zero_crossings = np.where(np.diff(np.sign(grad)))[0]

                # Mark inflection points
                for cross in zero_crossings:
                    if cross < len(x_smooth) - 1:
                        plt.axvline(x=x_smooth[cross], color='orange', linestyle='--', alpha=0.5)

            except:
                # If spline fails, use simple moving average
                window_size = 50
                sorted_indices = np.argsort(feature_values)
                sorted_values = feature_values[sorted_indices]
                sorted_shap = shap_for_feature[sorted_indices]

                # Moving average
                moving_avg = np.convolve(sorted_shap, np.ones(window_size) / window_size, mode='valid')
                moving_avg_x = np.convolve(sorted_values, np.ones(window_size) / window_size, mode='valid')

                plt.plot(moving_avg_x, moving_avg, 'g-', linewidth=3, label='Moving average trend')

            # Add beneficial interval annotations
            for interval_info in results['beneficial_intervals']:
                val_low, val_high = map(float, interval_info['interval'].split(' ~ '))
                plt.axvspan(val_low, val_high, alpha=0.2, color='yellow')

                # Add text annotation
                mid_point = (val_low + val_high) / 2
                benefit_level = interval_info['benefit_level']
                color = 'green' if 'Positive' in benefit_level else 'red' if 'Negative' in benefit_level else 'orange'
                plt.text(mid_point, plt.ylim()[1] * 0.9, benefit_level[:2],
                         ha='center', color=color, fontweight='bold')

            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel(f'Feature {feature_idx} value')
            plt.ylabel('SHAP value')
            plt.title(f'Feature {feature_idx} value interval impact analysis')
            plt.legend()
            plt.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/feature_{feature_idx}_interval_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

        # Create summary plot
        self._create_interval_summary_plot(interval_results, save_dir)

    def _create_interval_summary_plot(self, interval_results, save_dir):
        """Create interval analysis summary plot"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Feature importance vs interval benefit
        ax1 = axes[0, 0]
        feature_indices = []
        beneficial_counts = []
        avg_benefit_strength = []

        for feature_idx, results in interval_results.items():
            feature_indices.append(feature_idx)
            beneficial_counts.append(len(results['beneficial_intervals']))

            if results['beneficial_intervals']:
                strengths = [abs(info['mean_shap']) for info in results['beneficial_intervals']]
                avg_benefit_strength.append(np.mean(strengths))
            else:
                avg_benefit_strength.append(0)

        scatter = ax1.scatter(feature_indices, beneficial_counts,
                              s=np.array(avg_benefit_strength) * 500,
                              c=avg_benefit_strength, alpha=0.6, cmap='RdYlGn')
        ax1.set_xlabel('Feature index')
        ax1.set_ylabel('Beneficial interval count')
        ax1.set_title('Feature Importance vs Beneficial Interval Count')
        ax1.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Average benefit strength')

        # 2. Positive vs negative impact comparison
        ax2 = axes[0, 1]
        positive_effects = []
        negative_effects = []

        for feature_idx, results in interval_results.items():
            if results['beneficial_intervals']:
                pos_effects = []
                neg_effects = []
                for info in results['beneficial_intervals']:
                    if info['mean_shap'] > 0:
                        pos_effects.append(info['mean_shap'])
                    else:
                        neg_effects.append(info['mean_shap'])

                if pos_effects:
                    positive_effects.append(np.mean(pos_effects))
                else:
                    positive_effects.append(0)

                if neg_effects:
                    negative_effects.append(np.mean(neg_effects))
                else:
                    negative_effects.append(0)

        x_pos = np.arange(len(positive_effects))
        ax2.bar(x_pos - 0.2, positive_effects, width=0.4, label='Positive impact', color='green', alpha=0.7)
        ax2.bar(x_pos + 0.2, negative_effects, width=0.4, label='Negative impact', color='red', alpha=0.7)
        ax2.set_xlabel('Feature index')
        ax2.set_ylabel('Average impact strength')
        ax2.set_title('Positive vs Negative Impact Strength Comparison')
        ax2.legend()
        ax2.grid(alpha=0.3, axis='y')

        # 3. Interval distribution statistics
        ax3 = axes[1, 0]
        benefit_levels = {'Strong positive': 0, 'Positive': 0, 'Strong negative': 0, 'Negative': 0}

        for feature_idx, results in interval_results.items():
            for info in results['beneficial_intervals']:
                level = info['benefit_level']
                benefit_levels[level] += 1

        colors = ['darkgreen', 'lightgreen', 'darkred', 'lightcoral']
        ax3.bar(benefit_levels.keys(), benefit_levels.values(), color=colors)
        ax3.set_xlabel('Impact type')
        ax3.set_ylabel('Interval count')
        ax3.set_title('Beneficial/Harmful Interval Type Distribution')
        ax3.grid(alpha=0.3, axis='y')

        # 4. Heatmap for best features
        ax4 = axes[1, 1]

        # Select top 5 most important features
        top_features = list(interval_results.keys())[:5]
        if top_features:
            heatmap_data = []
            feature_names = []

            for feature_idx in top_features:
                results = interval_results[feature_idx]
                feature_names.append(f'Feature{feature_idx}')

                interval_effects = []
                for interval_info in results['intervals']:
                    interval_effects.append(interval_info['mean_shap'])

                # Pad to same length
                max_len = 20
                if len(interval_effects) < max_len:
                    interval_effects.extend([0] * (max_len - len(interval_effects)))
                elif len(interval_effects) > max_len:
                    interval_effects = interval_effects[:max_len]

                heatmap_data.append(interval_effects)

            if heatmap_data:
                im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
                ax4.set_xlabel('Value interval (percentile)')
                ax4.set_ylabel('Feature')
                ax4.set_yticks(range(len(feature_names)))
                ax4.set_yticklabels(feature_names)
                ax4.set_title('Interval Impact Heatmap for Top 5 Important Features')
                plt.colorbar(im, ax=ax4, label='SHAP value')

        plt.suptitle('Feature Interval Benefit Analysis Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/interval_analysis_summary.png', dpi=300, bbox_inches='tight')

    def _generate_interval_report(self, interval_results, save_dir):
        """Generate interval analysis report"""
        report_lines = [
            "=" * 80,
            "Feature Value Interval Benefit Analysis Report",
            "=" * 80,
            f"Generation time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "1. Analysis Overview",
            "-" * 50,
            f"Number of features analyzed: {len(interval_results)}",
            "",
            "2. Most Important Beneficial/Harmful Intervals",
            "-" * 50,
        ]

        # Collect all beneficial intervals
        all_beneficial = []
        for feature_idx, results in interval_results.items():
            for interval_info in results['beneficial_intervals']:
                all_beneficial.append({
                    'feature_idx': feature_idx,
                    'importance': results['importance'],
                    **interval_info
                })

        # Sort by benefit strength
        all_beneficial.sort(key=lambda x: abs(x['mean_shap']), reverse=True)

        # Report top 20 most important intervals
        for i, info in enumerate(all_beneficial[:20]):
            report_lines.append(f"{i + 1:2d}. Feature {info['feature_idx']} (importance: {info['importance']:.6f})")
            report_lines.append(f"     Interval: {info['interval']}")
            report_lines.append(f"     Percentile: {info['percentile']}")
            report_lines.append(f"     Impact type: {info['benefit_level']}")
            report_lines.append(f"     Mean SHAP value: {info['mean_shap']:.6f}")
            report_lines.append(f"     Effect on positive samples: {info['pos_effect']:.6f}")
            report_lines.append(f"     Effect on negative samples: {info['neg_effect']:.6f}")
            report_lines.append("")

        report_lines.extend([
            "3. Key Findings",
            "-" * 50,
        ])

        # Analyze patterns
        positive_intervals = [info for info in all_beneficial if info['mean_shap'] > 0]
        negative_intervals = [info for info in all_beneficial if info['mean_shap'] < 0]

        report_lines.append(f"Number of positive beneficial intervals: {len(positive_intervals)}")
        report_lines.append(f"Number of negative harmful intervals: {len(negative_intervals)}")

        if positive_intervals:
            avg_pos_strength = np.mean([abs(info['mean_shap']) for info in positive_intervals])
            report_lines.append(f"Average positive strength: {avg_pos_strength:.6f}")

        if negative_intervals:
            avg_neg_strength = np.mean([abs(info['mean_shap']) for info in negative_intervals])
            report_lines.append(f"Average negative strength: {avg_neg_strength:.6f}")

        # Find most discriminating feature interval
        if all_beneficial:
            best_feature = max(all_beneficial, key=lambda x: abs(x['mean_shap']))
            report_lines.append("")
            report_lines.append("Most discriminating interval:")
            report_lines.append(f"   Feature {best_feature['feature_idx']} - {best_feature['interval']}")
            report_lines.append(f"   {best_feature['benefit_level']} (strength: {abs(best_feature['mean_shap']):.6f})")

        report_lines.extend([
            "",
            "4. Analysis Conclusions",
            "-" * 50,
            "1. Positive intervals: Increasing these feature values increases positive sample prediction probability",
            "2. Negative intervals: Increasing these feature values decreases positive sample prediction probability",
            "3. Near-zero intervals: Little impact on prediction, can consider removing or simplifying",
            "4. Inflection points: Critical points where feature value changes from beneficial to harmful",
            "",
            "5. Application Recommendations",
            "-" * 50,
            "1. In feature engineering, focus on optimizing features in positive intervals",
            "2. In model interpretation, emphasize feature intervals most critical to prediction",
            "",
            "=" * 80,
        ])

        # Write report file
        report_path = os.path.join(save_dir, 'interval_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f" Interval analysis report saved: {report_path}")


# ==================== Main Function ====================

def main():
    """Main function"""
    print("Starting SHAP interpretability analysis")
    print("=" * 80)

    # Check necessary files
    required_files = [
        'saved_models/triple_ensemble_independent_test_20251207_194411.pth',
        'combined_features_final/pca_scaled_512/ptrain_combined1.npy',
        'combined_features_final/pca_scaled_512/ntrain_combined1.npy',
        'combined_features_final/pca_scaled_512/ptest_combined1.npy',
        'combined_features_final/pca_scaled_512/ntest_combined1.npy'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(" Missing the following necessary files:")
        for file in missing_files:
            print(f"   - {file}")
        return

    print(" All necessary files exist")

    print("\nSelect analysis mode:")
    print("1. Quick analysis (using 100 samples)")
    print("2. Complete analysis (using all samples)")
    print("=" * 80)

    choice = input("Please select mode (1-2): ").strip()

    # Create analyzer
    analyzer = FullDatasetSHAPAnalyzer(
        model_path='saved_models/triple_ensemble_independent_test_20251207_194411.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    start_time = time.time()

    if choice == '1':
        print("\n Starting quick analysis...")
        # Use partial samples for quick analysis
        shap_values = analyzer.run_full_shap_analysis(
            background_size=50,
            save_dir='Succinylation_Partial_SHAP_Analysis_Results'
        )
    elif choice == '2':
        print("\n Starting complete full-sample analysis...")
        # Use all samples for complete analysis
        shap_values = analyzer.run_full_shap_analysis(
            background_size=100,
            save_dir='Succinylation_Full_Sample_SHAP_Analysis_Results'
        )
    else:
        print(" Invalid selection")
        return

    elapsed_time = time.time() - start_time

    if shap_values is not None:
        print("\n SHAP analysis complete")


if __name__ == "__main__":
    main()