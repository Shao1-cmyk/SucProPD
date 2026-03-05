import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, \
    matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats
import itertools
from collections import Counter
import os

import pickle
import time
import matplotlib.pyplot as plt
import glob
import time
import joblib

warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3, num_classes=1):
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
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output.squeeze()


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
                 num_layers=3, forward_expansion=4, dropout=0.2, num_classes=1):
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
            nn.Linear(128, num_classes),
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
    def __init__(self, input_dim, mlp_hidden_dims=[512, 256, 128],
                 deepfri_hidden_dims=[1024, 512, 256], dropout=0.3, num_classes=1):
        super(TripleEnsemble, self).__init__()
        self.mlp = MLP(input_dim, mlp_hidden_dims, dropout, num_classes)
        self.pdeeppp = PDeepPP(input_dim=input_dim, dropout=dropout, num_classes=num_classes)
        self.deepfri = DeepFRI(input_dim, deepfri_hidden_dims, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes),
            nn.Sigmoid()
        )
        self.weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x):
        mlp_output = self.mlp(x).unsqueeze(1)
        pdeeppp_output = self.pdeeppp(x).unsqueeze(1)
        deepfri_output = self.deepfri(x).unsqueeze(1)
        combined = torch.cat([mlp_output, pdeeppp_output, deepfri_output], dim=1)
        weighted_combined = combined * F.softmax(self.weights, dim=0)
        ensemble_output = self.classifier(weighted_combined)
        return ensemble_output.squeeze()


class ModelManager:
    def __init__(self, model_dir='saved_models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def find_latest_model(self, pattern='*.pth'):
        model_files = glob.glob(os.path.join(self.model_dir, pattern))
        if not model_files:
            model_files = glob.glob(pattern)

        if not model_files:
            return None

        model_files.sort(key=os.path.getmtime, reverse=True)
        return model_files[0]

    def load_trained_model(self, model_path=None, input_dim=512, device='cuda'):
        if model_path is None:
            model_path = self.find_latest_model()
            if model_path is None:
                print("No trained model found.")
                return None

        print(f"Loading model from: {model_path}")

        model = TripleEnsemble(input_dim)

        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)

        except Exception as e:
            try:
                if os.path.exists(model_path):
                    model_data = torch.load(model_path, map_location=device, weights_only=False)

                    if isinstance(model_data, dict):
                        possible_keys = ['model_state_dict', 'state_dict', 'model', 'weights', 'net']
                        loaded = False
                        for key in possible_keys:
                            if key in model_data:
                                try:
                                    model.load_state_dict(model_data[key])
                                    loaded = True
                                    break
                                except:
                                    continue

                        if not loaded:
                            try:
                                model.load_state_dict(model_data)
                            except:
                                return None
                    else:
                        model.load_state_dict(model_data)
            except Exception as e2:
                print(f"All loading attempts failed: {e2}")
                return None

        model.to(device)
        model.eval()

        print("Model loaded successfully.")
        print(f"Device: {device}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        if hasattr(model, 'weights'):
            weights = F.softmax(model.weights, dim=0).detach().cpu().numpy()
            print(f"Ensemble weights - MLP: {weights[0]:.3f}, PDeepPP: {weights[1]:.3f}, DeepFRI: {weights[2]:.3f}")

        return model


class F1BasedInterpretability:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.prot_t5_dim = 1024
        self.cksaap_dim = 1200
        self.total_original_features = self.prot_t5_dim + self.cksaap_dim

    def _extract_position_from_prott5_index(self, feature_index):
        if feature_index < self.prot_t5_dim:
            position_idx = 0
            internal_feature = feature_index
            feature_type = "ProtT5_center"
            return {
                'feature_type': feature_type,
                'position': position_idx,
                'position_relative': 0,
                'internal_feature_idx': internal_feature
            }
        else:
            cksaap_idx = feature_index - self.prot_t5_dim
            feature_type = "CKSAAP"
            return {
                'feature_type': feature_type,
                'position': None,
                'position_relative': None,
                'internal_feature_idx': cksaap_idx
            }

    def _analyze_cksaap_feature(self, cksaap_idx):
        if cksaap_idx < 400:
            aa1_idx = cksaap_idx // 20
            aa2_idx = cksaap_idx % 20
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            aa1 = amino_acids[aa1_idx]
            aa2 = amino_acids[aa2_idx]
            return {
                'aa_pair': f"{aa1}{aa2}",
                'position_type': 'left_adjacent',
                'distance_from_center': -1
            }
        elif cksaap_idx < 800:
            rel_idx = cksaap_idx - 400
            aa1_idx = rel_idx // 20
            aa2_idx = rel_idx % 20
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            aa1 = amino_acids[aa1_idx]
            aa2 = amino_acids[aa2_idx]
            return {
                'aa_pair': f"{aa1}{aa2}",
                'position_type': 'right_adjacent',
                'distance_from_center': 1
            }
        else:
            rel_idx = cksaap_idx - 800
            aa_pair_idx = rel_idx % 400
            distance_group = rel_idx // 400 + 2

            aa1_idx = aa_pair_idx // 20
            aa2_idx = aa_pair_idx % 20
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            aa1 = amino_acids[aa1_idx]
            aa2 = amino_acids[aa2_idx]

            if distance_group <= 3:
                return {
                    'aa_pair': f"{aa1}{aa2}",
                    'position_type': f'left_{distance_group}',
                    'distance_from_center': -distance_group
                }
            else:
                return {
                    'aa_pair': f"{aa1}{aa2}",
                    'position_type': f'right_{distance_group - 3}',
                    'distance_from_center': distance_group - 3
                }

    def trace_pca_feature_to_sequence(self, pca_feature_idx, pca_components, top_n=10):
        if pca_feature_idx >= pca_components.shape[0]:
            print(f"Error: PCA feature index {pca_feature_idx} out of range")
            return None

        pca_weights = pca_components[pca_feature_idx]
        abs_weights = np.abs(pca_weights)
        top_indices = np.argsort(abs_weights)[-top_n:][::-1]

        trace_results = []
        for idx in top_indices:
            weight = pca_weights[idx]
            abs_weight = abs_weights[idx]

            if idx < self.prot_t5_dim:
                feature_info = self._extract_position_from_prott5_index(idx)
                trace_results.append({
                    'original_feature_idx': idx,
                    'weight': weight,
                    'abs_weight': abs_weight,
                    'feature_type': feature_info['feature_type'],
                    'position': feature_info['position'],
                    'position_relative': 0,
                    'distance_from_center': 0,
                    'internal_idx': feature_info['internal_feature_idx'],
                    'feature_desc': f"ProtT5_center"
                })
            else:
                cksaap_idx = idx - self.prot_t5_dim
                aa_info = self._analyze_cksaap_feature_detailed(cksaap_idx)
                trace_results.append({
                    'original_feature_idx': idx,
                    'weight': weight,
                    'abs_weight': abs_weight,
                    'feature_type': 'CKSAAP',
                    'aa_pair': aa_info['aa_pair'],
                    'position_type': aa_info['position_type'],
                    'position_relative': aa_info['distance_from_center'],
                    'distance_from_center': aa_info['distance_from_center'],
                    'internal_idx': cksaap_idx,
                    'feature_desc': aa_info['description']
                })

        return trace_results

    def _evaluate_f1_score(self, X, y):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).float().cpu().numpy()

        f1 = f1_score(y, predictions, zero_division=0)
        return f1

    def _evaluate_multiple_metrics(self, X, y):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).float().cpu().numpy()
            probabilities = outputs.cpu().numpy()

        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)
        mcc = matthews_corrcoef(y, predictions)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'predictions': predictions,
            'probabilities': probabilities
        }

    def load_sequences(self, pos_file, neg_file, test_pos_file, test_neg_file):
        def read_sequences(filepath):
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
            return []

        self.train_pos_seqs = read_sequences(pos_file)
        self.train_neg_seqs = read_sequences(neg_file)
        self.test_pos_seqs = read_sequences(test_pos_file)
        self.test_neg_seqs = read_sequences(test_neg_file)

        self.all_seqs = self.train_pos_seqs + self.train_neg_seqs + \
                        self.test_pos_seqs + self.test_neg_seqs

        print(f"Loaded {len(self.train_pos_seqs)} positive training sequences")
        print(f"Loaded {len(self.train_neg_seqs)} negative training sequences")
        print(f"Loaded {len(self.test_pos_seqs)} positive test sequences")
        print(f"Loaded {len(self.test_neg_seqs)} negative test sequences")

        return self.all_seqs

    def analyze_amino_acid_distribution(self, sequences, title="Amino Acid Distribution", save_path=None):
        if not sequences:
            print("No sequences provided for analysis")
            return None

        all_aas = ''.join(sequences)
        aa_counts = Counter(all_aas)

        standard_aas = 'ACDEFGHIKLMNPQRSTVWY'
        aa_labels = [f'{aa}' for aa in standard_aas]
        aa_values = [aa_counts.get(aa, 0) for aa in standard_aas]

        plt.figure(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(standard_aas)))
        bars = plt.bar(aa_labels, aa_values, color=colors)

        plt.xlabel('Amino Acid')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.xticks(rotation=45)

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height)}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return aa_counts

    def calculate_f1_feature_importance(self, X_train, y_train, X_test, y_test, n_iterations=3):
        print("Calculating feature importance via permutation...")

        baseline_metrics = self._evaluate_multiple_metrics(X_test, y_test)
        baseline_f1 = baseline_metrics['f1']
        print(f"Baseline F1: {baseline_f1:.4f}")
        print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"Baseline Precision: {baseline_metrics['precision']:.4f}")
        print(f"Baseline Recall: {baseline_metrics['recall']:.4f}")
        print(f"Baseline MCC: {baseline_metrics['mcc']:.4f}")

        n_features = X_train.shape[1]
        f1_importances = np.zeros(n_features)
        acc_importances = np.zeros(n_features)
        prec_importances = np.zeros(n_features)
        recall_importances = np.zeros(n_features)

        print(f"Analyzing {n_features} features...")

        for i in range(n_features):
            permuted_f1s = []
            permuted_accs = []
            permuted_precs = []
            permuted_recalls = []

            for _ in range(n_iterations):
                X_permuted = X_test.copy()
                np.random.shuffle(X_permuted[:, i])
                metrics = self._evaluate_multiple_metrics(X_permuted, y_test)
                permuted_f1s.append(metrics['f1'])
                permuted_accs.append(metrics['accuracy'])
                permuted_precs.append(metrics['precision'])
                permuted_recalls.append(metrics['recall'])

            f1_importances[i] = baseline_f1 - np.mean(permuted_f1s)
            acc_importances[i] = baseline_metrics['accuracy'] - np.mean(permuted_accs)
            prec_importances[i] = baseline_metrics['precision'] - np.mean(permuted_precs)
            recall_importances[i] = baseline_metrics['recall'] - np.mean(permuted_recalls)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{n_features} features")

        print("Feature importance calculation complete.")

        return {
            'f1_importances': f1_importances,
            'acc_importances': acc_importances,
            'prec_importances': prec_importances,
            'recall_importances': recall_importances,
            'baseline_metrics': baseline_metrics
        }

    def plot_f1_feature_importance(self, importances, top_n=10, save_path='feature_importance_f1_top10.png'):
        sorted_indices = np.argsort(importances['f1_importances'])[-top_n:][::-1]
        sorted_f1_importances = importances['f1_importances'][sorted_indices]
        sorted_acc_importances = importances['acc_importances'][sorted_indices]
        sorted_prec_importances = importances['prec_importances'][sorted_indices]
        sorted_recall_importances = importances['recall_importances'][sorted_indices]

        plt.figure(figsize=(14, 8))

        x = np.arange(top_n)
        width = 0.2

        plt.bar(x - 1.5 * width, sorted_f1_importances, width, label='F1 Importance', color='#ff6b6b')
        plt.bar(x - 0.5 * width, sorted_acc_importances, width, label='Accuracy Importance', color='#4ecdc4')
        plt.bar(x + 0.5 * width, sorted_prec_importances, width, label='Precision Importance', color='#45b7d1')
        plt.bar(x + 1.5 * width, sorted_recall_importances, width, label='Recall Importance', color='#96ceb4')

        plt.xticks(x, [f'F{idx}' for idx in sorted_indices], rotation=45, fontsize=10)
        plt.xlabel('Feature Index', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.title(f'F1-based Feature Importance (Top {top_n})', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3, linestyle='--')

        baseline_text = f"Baseline:\nF1={importances['baseline_metrics']['f1']:.4f}\nAcc={importances['baseline_metrics']['accuracy']:.4f}"
        plt.text(0.02, 0.98, baseline_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return sorted_indices, sorted_f1_importances

    def _analyze_cksaap_feature_detailed(self, cksaap_idx):
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

        if cksaap_idx < 400:
            aa_pair_idx = cksaap_idx
            aa1_idx = aa_pair_idx // 20
            aa2_idx = aa_pair_idx % 20
            aa1 = amino_acids[aa1_idx]
            aa2 = amino_acids[aa2_idx]
            return {
                'aa_pair': f"{aa1}{aa2}",
                'position_type': 'left_adjacent',
                'distance_from_center': -1,
                'position_relative': -1,
                'description': f"left_adjacent(-1): {aa1}{aa2}"
            }
        elif cksaap_idx < 800:
            rel_idx = cksaap_idx - 400
            aa_pair_idx = rel_idx
            aa1_idx = aa_pair_idx // 20
            aa2_idx = aa_pair_idx % 20
            aa1 = amino_acids[aa1_idx]
            aa2 = amino_acids[aa2_idx]
            return {
                'aa_pair': f"{aa1}{aa2}",
                'position_type': 'right_adjacent',
                'distance_from_center': 1,
                'position_relative': 1,
                'description': f"right_adjacent(+1): {aa1}{aa2}"
            }
        else:
            rel_idx = cksaap_idx - 800
            aa_pair_idx = rel_idx % 400
            distance_group = rel_idx // 400

            aa1_idx = aa_pair_idx // 20
            aa2_idx = aa_pair_idx % 20
            aa1 = amino_acids[aa1_idx]
            aa2 = amino_acids[aa2_idx]

            if distance_group < 4:
                distance = -(distance_group + 2)
                return {
                    'aa_pair': f"{aa1}{aa2}",
                    'position_type': f'left',
                    'distance_from_center': distance,
                    'position_relative': distance,
                    'description': f"left({distance}): {aa1}{aa2}"
                }
            else:
                distance = (distance_group - 4) + 2
                return {
                    'aa_pair': f"{aa1}{aa2}",
                    'position_type': f'right',
                    'distance_from_center': distance,
                    'position_relative': distance,
                    'description': f"right(+{distance}): {aa1}{aa2}"
                }

    def plot_f1_correlations(self, importances, save_path='f1_correlations.png'):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        mask = (importances['f1_importances'] > 0) & (importances['acc_importances'] > 0)

        axes[0].scatter(importances['f1_importances'][mask], importances['acc_importances'][mask],
                        alpha=0.5, s=10, color='blue')
        corr = np.corrcoef(importances['f1_importances'][mask], importances['acc_importances'][mask])[0, 1]
        axes[0].set_xlabel('F1 Importance')
        axes[0].set_ylabel('Accuracy Importance')
        axes[0].set_title(f'F1 vs Accuracy (r={corr:.3f})')
        axes[0].grid(alpha=0.3)

        axes[1].scatter(importances['f1_importances'][mask], importances['prec_importances'][mask],
                        alpha=0.5, s=10, color='green')
        corr = np.corrcoef(importances['f1_importances'][mask], importances['prec_importances'][mask])[0, 1]
        axes[1].set_xlabel('F1 Importance')
        axes[1].set_ylabel('Precision Importance')
        axes[1].set_title(f'F1 vs Precision (r={corr:.3f})')
        axes[1].grid(alpha=0.3)

        axes[2].scatter(importances['f1_importances'][mask], importances['recall_importances'][mask],
                        alpha=0.5, s=10, color='red')
        corr = np.corrcoef(importances['f1_importances'][mask], importances['recall_importances'][mask])[0, 1]
        axes[2].set_xlabel('F1 Importance')
        axes[2].set_ylabel('Recall Importance')
        axes[2].set_title(f'F1 vs Recall (r={corr:.3f})')
        axes[2].grid(alpha=0.3)

        axes[3].scatter(importances['acc_importances'][mask], importances['prec_importances'][mask],
                        alpha=0.5, s=10, color='orange')
        corr = np.corrcoef(importances['acc_importances'][mask], importances['prec_importances'][mask])[0, 1]
        axes[3].set_xlabel('Accuracy Importance')
        axes[3].set_ylabel('Precision Importance')
        axes[3].set_title(f'Accuracy vs Precision (r={corr:.3f})')
        axes[3].grid(alpha=0.3)

        axes[4].scatter(importances['acc_importances'][mask], importances['recall_importances'][mask],
                        alpha=0.5, s=10, color='purple')
        corr = np.corrcoef(importances['acc_importances'][mask], importances['recall_importances'][mask])[0, 1]
        axes[4].set_xlabel('Accuracy Importance')
        axes[4].set_ylabel('Recall Importance')
        axes[4].set_title(f'Accuracy vs Recall (r={corr:.3f})')
        axes[4].grid(alpha=0.3)

        axes[5].scatter(importances['prec_importances'][mask], importances['recall_importances'][mask],
                        alpha=0.5, s=10, color='brown')
        corr = np.corrcoef(importances['prec_importances'][mask], importances['recall_importances'][mask])[0, 1]
        axes[5].set_xlabel('Precision Importance')
        axes[5].set_ylabel('Recall Importance')
        axes[5].set_title(f'Precision vs Recall (r={corr:.3f})')
        axes[5].grid(alpha=0.3)

        plt.suptitle('Correlation Analysis of Metric Importances', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'f1_acc_corr': np.corrcoef(importances['f1_importances'][mask], importances['acc_importances'][mask])[0, 1],
            'f1_prec_corr': np.corrcoef(importances['f1_importances'][mask], importances['prec_importances'][mask])[
                0, 1],
            'f1_recall_corr': np.corrcoef(importances['f1_importances'][mask], importances['recall_importances'][mask])[
                0, 1]
        }

    def analyze_model_component_f1_contributions(self, X_test, y_test):
        print("Analyzing model component contributions...")

        self.model.eval()
        X_tensor = torch.FloatTensor(X_test).to(self.device)

        with torch.no_grad():
            full_outputs = self.model(X_tensor)
            full_predictions = (full_outputs > 0.5).float().cpu().numpy()
            full_f1 = f1_score(y_test, full_predictions, zero_division=0)

            mlp_outputs = self.model.mlp(X_tensor)
            pdeeppp_outputs = self.model.pdeeppp(X_tensor)
            deepfri_outputs = self.model.deepfri(X_tensor)

            mlp_predictions = (mlp_outputs > 0.5).float().cpu().numpy()
            pdeeppp_predictions = (pdeeppp_outputs > 0.5).float().cpu().numpy()
            deepfri_predictions = (deepfri_outputs > 0.5).float().cpu().numpy()

            mlp_f1 = f1_score(y_test, mlp_predictions, zero_division=0)
            pdeeppp_f1 = f1_score(y_test, pdeeppp_predictions, zero_division=0)
            deepfri_f1 = f1_score(y_test, deepfri_predictions, zero_division=0)

        if hasattr(self.model, 'weights'):
            weights = F.softmax(self.model.weights, dim=0).detach().cpu().numpy()
        else:
            weights = np.array([1 / 3, 1 / 3, 1 / 3])

        weighted_f1 = np.sum([mlp_f1 * weights[0], pdeeppp_f1 * weights[1], deepfri_f1 * weights[2]])

        print(f"Full model F1: {full_f1:.4f}")
        print(f"MLP F1: {mlp_f1:.4f} (weight: {weights[0]:.3f})")
        print(f"PDeepPP F1: {pdeeppp_f1:.4f} (weight: {weights[1]:.3f})")
        print(f"DeepFRI F1: {deepfri_f1:.4f} (weight: {weights[2]:.3f})")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(f"Ensemble gain: {full_f1 - weighted_f1:.4f}")

        plt.figure(figsize=(12, 8))

        components = ['MLP', 'PDeepPP', 'DeepFRI', 'Ensemble']
        f1_scores = [mlp_f1, pdeeppp_f1, deepfri_f1, full_f1]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD166']

        bars = plt.bar(components, f1_scores, color=colors, alpha=0.8)

        plt.ylabel('F1 Score', fontsize=12)
        plt.title('Model Component F1 Contributions', fontsize=14, fontweight='bold')
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)

        for bar, f1_score_val in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{f1_score_val:.4f}', ha='center', va='bottom', fontsize=10)

        weight_text = f"Ensemble weights:\nMLP: {weights[0]:.3f}\nPDeepPP: {weights[1]:.3f}\nDeepFRI: {weights[2]:.3f}"
        plt.text(0.02, 0.98, weight_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        plt.savefig('model_components_f1_contributions.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'full_f1': full_f1,
            'mlp_f1': mlp_f1,
            'pdeeppp_f1': pdeeppp_f1,
            'deepfri_f1': deepfri_f1,
            'weights': weights,
            'weighted_f1': weighted_f1,
            'ensemble_gain': full_f1 - weighted_f1
        }

    def visualize_f1_decision_boundary(self, X, y, n_samples=1000, save_path='f1_decision_boundary.png'):
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sampled = X[indices]
            y_sampled = y[indices]
        else:
            X_sampled = X
            y_sampled = y

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = tsne.fit_transform(X_sampled)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sampled).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = outputs.cpu().numpy()
            predictions = (outputs > 0.5).float().cpu().numpy()

        tp_mask = (predictions == 1) & (y_sampled == 1)
        tn_mask = (predictions == 0) & (y_sampled == 0)
        fp_mask = (predictions == 1) & (y_sampled == 0)
        fn_mask = (predictions == 0) & (y_sampled == 1)

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        scatter1 = axes[0, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_sampled,
                                      cmap='coolwarm', alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[0, 0].set_title('True Labels (0=negative, 1=positive)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('t-SNE Component 1', fontsize=12)
        axes[0, 0].set_ylabel('t-SNE Component 2', fontsize=12)
        plt.colorbar(scatter1, ax=axes[0, 0], label='Label')

        scatter2 = axes[0, 1].scatter(X_2d[:, 0], X_2d[:, 1], c=probabilities,
                                      cmap='RdYlBu', alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[0, 1].set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('t-SNE Component 1', fontsize=12)
        axes[0, 1].set_ylabel('t-SNE Component 2', fontsize=12)
        plt.colorbar(scatter2, ax=axes[0, 1], label='Probability')

        colors = np.zeros(len(X_2d))
        colors[tp_mask] = 1
        colors[tn_mask] = 2
        colors[fp_mask] = 3
        colors[fn_mask] = 4

        scatter3 = axes[1, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=colors,
                                      cmap='Set2', alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[1, 0].set_title('F1-based Classification (TP/TN/FP/FN)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('t-SNE Component 1', fontsize=12)
        axes[1, 0].set_ylabel('t-SNE Component 2', fontsize=12)

        cbar = plt.colorbar(scatter3, ax=axes[1, 0], ticks=[1.5, 2.5, 3.5, 4.5])
        cbar.ax.set_yticklabels(['TP', 'TN', 'FP', 'FN'])

        confidence = np.abs(probabilities - 0.5) * 2
        scatter4 = axes[1, 1].scatter(X_2d[:, 0], X_2d[:, 1], c=confidence,
                                      cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[1, 1].set_title('Prediction Confidence', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('t-SNE Component 1', fontsize=12)
        axes[1, 1].set_ylabel('t-SNE Component 2', fontsize=12)
        plt.colorbar(scatter4, ax=axes[1, 1], label='Confidence')

        tp_count = np.sum(tp_mask)
        tn_count = np.sum(tn_mask)
        fp_count = np.sum(fp_mask)
        fn_count = np.sum(fn_mask)

        accuracy = (tp_count + tn_count) / len(y_sampled)
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        stats_text = f"TP: {tp_count}\nTN: {tn_count}\nFP: {fp_count}\nFN: {fn_count}\n\nAcc: {accuracy:.3f}\nPrec: {precision:.3f}\nRec: {recall:.3f}\nF1: {f1:.3f}"
        axes[1, 0].text(0.02, 0.98, stats_text, transform=axes[1, 0].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.suptitle('F1-based Decision Boundary Visualization', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'tp_count': tp_count,
            'tn_count': tn_count,
            'fp_count': fp_count,
            'fn_count': fn_count,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def analyze_f1_prediction_distribution(self, X_test, y_test, save_path='f1_prediction_distribution.png'):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        thresholds = np.linspace(0, 1, 101)
        f1_scores = []
        precisions = []
        recalls = []

        for threshold in thresholds:
            pred_labels = (predictions >= threshold).astype(int)
            f1 = f1_score(y_test, pred_labels, zero_division=0)
            precision = precision_score(y_test, pred_labels, zero_division=0)
            recall = recall_score(y_test, pred_labels, zero_division=0)
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]

        print(f"Best F1: {best_f1:.4f} (threshold: {best_threshold:.3f})")

        plt.figure(figsize=(14, 10))

        plt.subplot(2, 2, 1)
        plt.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1')
        plt.axvline(x=best_threshold, color='r', linestyle='--', alpha=0.7, label=f'Best threshold={best_threshold:.3f}')
        plt.axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='Default threshold=0.5')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 vs Threshold')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(thresholds, precisions, 'g-', linewidth=2, label='Precision')
        plt.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
        plt.axvline(x=best_threshold, color='b', linestyle='--', alpha=0.7)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision and Recall vs Threshold')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.hist(predictions[y_test == 1], bins=30, alpha=0.7, label='Positive', color='red', density=True)
        plt.hist(predictions[y_test == 0], bins=30, alpha=0.7, label='Negative', color='blue', density=True)
        plt.axvline(x=best_threshold, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Best threshold')
        plt.axvline(x=0.5, color='g', linestyle='--', linewidth=2, alpha=0.7, label='Default threshold')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Density')
        plt.title('Prediction Distribution')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(2, 2, 4)
        best_pred_labels = (predictions >= best_threshold).astype(int)
        cm = confusion_matrix(y_test, best_pred_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred Negative', 'Pred Positive'],
                    yticklabels=['True Negative', 'True Positive'])
        plt.title(f'Confusion Matrix (threshold={best_threshold:.3f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.suptitle('F1-based Prediction Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'default_f1': f1_scores[50],
            'thresholds': thresholds,
            'f1_scores': f1_scores,
            'precisions': precisions,
            'recalls': recalls
        }

    def analyze_top_features_positions(self, importances, pca_components, top_k=10):
        f1_importances = importances['f1_importances']
        top_pca_indices = np.argsort(f1_importances)[-top_k:][::-1]

        print("=" * 80)
        print("Feature Traceback to Sequence Positions")
        print("=" * 80)
        print(f"Original features: ProtT5({self.prot_t5_dim}) + CKSAAP({self.cksaap_dim})")
        print(f"PCA features: {pca_components.shape[0]}")
        print(f"Analyzing top {top_k} features")
        print("Position representation: center=0, left=negative, right=positive")
        print("=" * 80)

        position_analysis = {
            'positions': [],
            'ckSAAP_features': [],
            'protT5_features': []
        }

        position_distribution = Counter()
        amino_acid_freq = Counter()

        for i, pca_idx in enumerate(top_pca_indices):
            importance_score = f1_importances[pca_idx]

            print(f"\n[{i + 1}] PCA feature F{pca_idx}")
            print(f"  F1 importance: {importance_score:.6f}")
            print(f"  Corresponding original features:")

            trace_results = self.trace_pca_feature_to_sequence(pca_idx, pca_components, top_n=3)

            if trace_results:
                for j, result in enumerate(trace_results):
                    print(f"    - {result['feature_desc']}, weight={result['weight']:.4f}")

                    if 'position_relative' in result:
                        pos = result['position_relative']
                        position_distribution[pos] += 1
                        position_analysis['positions'].append({
                            'pca_idx': pca_idx,
                            'importance': importance_score,
                            'position': pos,
                            'feature_type': result['feature_type'],
                            'feature_desc': result['feature_desc']
                        })

                    if result['feature_type'] == 'CKSAAP' and 'aa_pair' in result:
                        aa_pair = result['aa_pair']
                        amino_acid_freq.update(aa_pair)
                        position_analysis['ckSAAP_features'].append({
                            'pca_idx': pca_idx,
                            'aa_pair': aa_pair,
                            'position': result['position_relative'],
                            'description': result['feature_desc']
                        })
                    elif result['feature_type'] == 'ProtT5_center':
                        position_analysis['protT5_features'].append({
                            'pca_idx': pca_idx,
                            'position': 0,
                            'description': 'ProtT5 center features'
                        })

        print("\n" + "=" * 80)
        print("Position Distribution Analysis")
        print("=" * 80)

        print(f"\n1. Position distribution (center=0):")
        sorted_positions = sorted(position_distribution.items())
        for pos, count in sorted_positions:
            if pos == 0:
                print(f"   Center(0): {count} features")
            elif pos < 0:
                print(f"   Left({pos}): {count} features")
            else:
                print(f"   Right(+{pos}): {count} features")

        print(f"\n2. Amino acid frequency (top 10):")
        if amino_acid_freq:
            total_aa = sum(amino_acid_freq.values())
            for aa, count in amino_acid_freq.most_common(10):
                freq = count / total_aa * 100
                print(f"   {aa}: {count} ({freq:.1f}%)")

        print(f"\n3. Position preference analysis:")
        if position_distribution:
            all_positions = [pos for pos, count in position_distribution.items() for _ in range(count)]
            avg_distance = np.mean(np.abs(all_positions))
            print(f"   Average absolute distance from center: {avg_distance:.2f}")

            left_count = sum(count for pos, count in position_distribution.items() if pos < 0)
            right_count = sum(count for pos, count in position_distribution.items() if pos > 0)
            center_count = position_distribution.get(0, 0)
            print(f"   Left features: {left_count}")
            print(f"   Center features: {center_count}")
            print(f"   Right features: {right_count}")

        self._plot_position_distribution_detailed(position_distribution, amino_acid_freq)

        return position_analysis

    def _plot_position_distribution_detailed(self, position_dist, aa_freq, save_path='position_analysis_detailed.png'):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        if position_dist:
            sorted_positions = sorted(position_dist.items())
            positions = [pos for pos, _ in sorted_positions]
            counts = [count for _, count in sorted_positions]

            colors = []
            for pos in positions:
                if pos < 0:
                    colors.append('#3498db')
                elif pos == 0:
                    colors.append('#2ecc71')
                else:
                    colors.append('#e74c3c')

            bars = axes[0, 0].bar(range(len(positions)), counts, color=colors, alpha=0.7)
            axes[0, 0].set_title('Position Distribution (center=0)', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Relative Position', fontsize=12)
            axes[0, 0].set_ylabel('Feature Count', fontsize=12)

            x_labels = [f'{pos}' for pos in positions]
            axes[0, 0].set_xticks(range(len(positions)))
            axes[0, 0].set_xticklabels(x_labels, rotation=45)

            for bar, count in zip(bars, counts):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height,
                                f'{count}', ha='center', va='bottom', fontsize=9)

        if aa_freq:
            sorted_aa = sorted(aa_freq.items(), key=lambda x: x[1], reverse=True)
            aa_labels = [item[0] for item in sorted_aa[:15]]
            aa_counts = [item[1] for item in sorted_aa[:15]]

            colors = plt.cm.viridis(np.linspace(0, 1, len(aa_labels)))
            bars = axes[0, 1].bar(aa_labels, aa_counts, color=colors)
            axes[0, 1].set_title('Amino Acid Frequency', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Amino Acid', fontsize=12)
            axes[0, 1].set_ylabel('Count', fontsize=12)
            axes[0, 1].tick_params(axis='x', rotation=45)

            for bar, count in zip(bars, aa_counts):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height,
                                f'{count}', ha='center', va='bottom', fontsize=9)

        if position_dist:
            left_counts = {pos: count for pos, count in position_dist.items() if pos < 0}
            right_counts = {pos: count for pos, count in position_dist.items() if pos > 0}

            if left_counts:
                left_pos = list(left_counts.keys())
                left_vals = list(left_counts.values())
                axes[1, 0].bar(left_pos, left_vals, color='skyblue', alpha=0.7, label='Left')

            if right_counts:
                right_pos = list(right_counts.keys())
                right_vals = list(right_counts.values())
                axes[1, 0].bar(right_pos, right_vals, color='lightcoral', alpha=0.7, label='Right')

            axes[1, 0].axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Center')
            axes[1, 0].set_title('Left-Right Distribution Comparison', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Relative Position', fontsize=12)
            axes[1, 0].set_ylabel('Feature Count', fontsize=12)
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)

        if position_dist:
            distances = [abs(pos) for pos, count in position_dist.items() for _ in range(count)]
            distance_counts = Counter(distances)

            dist_positions = sorted(distance_counts.keys())
            dist_counts = [distance_counts[d] for d in dist_positions]

            bars = axes[1, 1].bar(dist_positions, dist_counts, color='#9b59b6', alpha=0.7)
            axes[1, 1].set_title('Distance from Center Distribution', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Absolute Distance from Center', fontsize=12)
            axes[1, 1].set_ylabel('Feature Count', fontsize=12)
            axes[1, 1].set_xticks(dist_positions)
            axes[1, 1].grid(alpha=0.3)

            for bar, count in zip(bars, dist_counts):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height,
                                f'{count}', ha='center', va='bottom', fontsize=9)

        plt.suptitle('Succinylation Site Position Feature Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_biological_significance(self, importances, pca_components, top_k=30):
        print("\n" + "=" * 80)
        print("Biological Significance Analysis")
        print("=" * 80)

        f1_importances = importances['f1_importances']
        top_pca_indices = np.argsort(f1_importances)[-top_k:][::-1]

        important_aa_pairs = Counter()
        position_patterns = Counter()

        succinylation_preference = {
            'K': 'Lysine',
            'R': 'Arginine',
            'S': 'Serine',
            'T': 'Threonine',
            'Y': 'Tyrosine',
            'D': 'Aspartic acid',
            'E': 'Glutamic acid',
            'P': 'Proline',
            'G': 'Glycine'
        }

        print("\nTop 5 most important features:")
        for i, pca_idx in enumerate(top_pca_indices[:5]):
            importance_score = f1_importances[pca_idx]
            print(f"\n[{i + 1}] PCA feature F{pca_idx} (importance: {importance_score:.4f})")

            trace_results = self.trace_pca_feature_to_sequence(pca_idx, pca_components, top_n=3)

            if trace_results:
                for result in trace_results[:2]:
                    print(f"  - {result['feature_desc']}")

                    if result['feature_type'] == 'CKSAAP':
                        aa_pair = result['aa_pair']
                        important_aa_pairs[aa_pair] += 1

                        aa1, aa2 = aa_pair[0], aa_pair[1]
                        bio_info1 = succinylation_preference.get(aa1, "")
                        bio_info2 = succinylation_preference.get(aa2, "")

                        position_info = f"Position: {result['position_type']}"

                        if 'K' in aa_pair:
                            print(f"     Contains Lysine(K)")

                        if aa1 == 'P' or aa2 == 'P':
                            print(f"     Contains Proline(P)")

                        if aa1 == 'D' or aa2 == 'D':
                            print(f"     Contains Aspartic acid(D)")

                        print(f"    {position_info}")

        print("\n" + "=" * 80)
        print("Amino Acid Pair Frequency")
        print("=" * 80)

        if important_aa_pairs:
            print("\nImportant amino acid pair frequency:")
            for aa_pair, count in important_aa_pairs.most_common(20):
                print(f"  {aa_pair}: {count}")

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)

        key_amino_acids = {
            'K': {'name': 'Lysine', 'count': 0},
            'R': {'name': 'Arginine', 'count': 0},
            'D': {'name': 'Aspartic acid', 'count': 0},
            'E': {'name': 'Glutamic acid', 'count': 0},
            'P': {'name': 'Proline', 'count': 0},
            'S': {'name': 'Serine', 'count': 0},
            'T': {'name': 'Threonine', 'count': 0}
        }

        for aa_pair, count in important_aa_pairs.items():
            for aa in aa_pair:
                if aa in key_amino_acids:
                    key_amino_acids[aa]['count'] += count

        print("\nKey amino acids in important features:")
        for aa, info in key_amino_acids.items():
            if info['count'] > 0:
                print(f"  {aa}({info['name']}): {info['count']}")

        return {
            'important_aa_pairs': important_aa_pairs,
            'key_amino_acids': key_amino_acids
        }

    def run_f1_based_analysis(self, X_train, y_train, X_test, y_test,
                              sequences=None, save_dir='F1_Interpretability_Results',
                              pca_components=None):
        os.makedirs(save_dir, exist_ok=True)

        print("=" * 80)
        print("Succinylation Site Prediction - F1-based Interpretability Analysis")
        print("=" * 80)

        results = {}

        if sequences:
            print("\n1. Analyzing amino acid distribution...")
            try:
                aa_counts = self.analyze_amino_acid_distribution(
                    sequences[:500],
                    "Amino Acid Distribution in Succinylation Dataset",
                    save_path=f'{save_dir}/amino_acid_distribution.png'
                )
                results['aa_counts'] = aa_counts
                print("Amino acid distribution analysis complete.")
            except Exception as e:
                print(f"Amino acid distribution analysis failed: {e}")

        print("\n2. Calculating F1-based feature importance...")
        try:
            importances = self.calculate_f1_feature_importance(X_train, y_train, X_test, y_test, n_iterations=3)
            top_indices, top_importances = self.plot_f1_feature_importance(
                importances, top_n=10,
                save_path=f'{save_dir}/feature_importance_f1_top10.png'
            )

            np.save(f'{save_dir}/feature_importances_f1.npy', importances['f1_importances'])
            pd.DataFrame({
                'feature_index': top_indices,
                'f1_importance': top_importances,
                'acc_importance': importances['acc_importances'][top_indices],
                'prec_importance': importances['prec_importances'][top_indices],
                'recall_importance': importances['recall_importances'][top_indices]
            }).to_csv(f'{save_dir}/top10_features_f1.csv', index=False)

            results['importances'] = importances
            results['top_features_f1'] = top_indices
            print("F1-based feature importance analysis complete.")
        except Exception as e:
            print(f"F1 feature importance calculation failed: {e}")

        if pca_components is not None:
            print("\n3. Analyzing feature position distribution (top 10 features)...")
            try:
                position_analysis = self.analyze_top_features_positions(
                    importances, pca_components, top_k=10
                )
                results['position_analysis'] = position_analysis
                print("Sequence position analysis complete.")

                bio_analysis = self.analyze_biological_significance(
                    importances, pca_components, top_k=10
                )
                results['biological_analysis'] = bio_analysis
                print("Biological significance analysis complete.")

            except Exception as e:
                print(f"Sequence position analysis failed: {e}")

        print("\n4. Analyzing metric importance correlations...")
        try:
            correlations = self.plot_f1_correlations(
                importances,
                save_path=f'{save_dir}/f1_correlations.png'
            )
            results['correlations'] = correlations
            print("Correlation analysis complete.")
        except Exception as e:
            print(f"Correlation analysis failed: {e}")

        print("\n5. Analyzing model component F1 contributions...")
        try:
            component_results = self.analyze_model_component_f1_contributions(X_test, y_test)
            results['component_contributions'] = component_results
            print("Model component analysis complete.")
        except Exception as e:
            print(f"Model component analysis failed: {e}")

        print("\n6. Visualizing F1-based decision boundaries...")
        try:
            decision_stats = self.visualize_f1_decision_boundary(
                X_test, y_test, n_samples=500,
                save_path=f'{save_dir}/f1_decision_boundary.png'
            )
            results['decision_stats'] = decision_stats
            print("Decision boundary visualization complete.")
        except Exception as e:
            print(f"Decision boundary visualization failed: {e}")

        print("\n7. Analyzing prediction distribution vs F1...")
        try:
            prediction_analysis = self.analyze_f1_prediction_distribution(
                X_test, y_test,
                save_path=f'{save_dir}/f1_prediction_analysis.png'
            )
            results['prediction_analysis'] = prediction_analysis
            print("Prediction distribution analysis complete.")
        except Exception as e:
            print(f"Prediction distribution analysis failed: {e}")

        print("\n" + "=" * 80)
        print("F1-based interpretability analysis complete.")
        print(f"Results saved to: {save_dir}")
        print("=" * 80)

        self._generate_f1_summary_report(results, save_dir)

        return results

    def generate_detailed_feature_report(self, importances, pca_components, top_k=30, save_dir='.'):
        print("\n" + "=" * 80)
        print("Feature Traceback Report")
        print("=" * 80)

        f1_importances = importances['f1_importances']
        top_pca_indices = np.argsort(f1_importances)[-top_k:][::-1]

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Succinylation Site Prediction - Feature Traceback Analysis Report")
        report_lines.append("=" * 80)
        report_lines.append(f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(
            f"Original features: ProtT5({self.prot_t5_dim}) + CKSAAP({self.cksaap_dim}) = {self.total_original_features}")
        report_lines.append(f"PCA features: {pca_components.shape[0]}")
        report_lines.append("=" * 80)
        report_lines.append("")

        key_amino_acids = ['K', 'D', 'P', 'R', 'S', 'T', 'E', 'G']
        aa_counts = {aa: 0 for aa in key_amino_acids}
        position_counts = {}

        for i, pca_idx in enumerate(top_pca_indices):
            importance_score = f1_importances[pca_idx]

            report_lines.append(f"\nFeature F{pca_idx} (importance: {importance_score:.6f})")

            trace_results = self.trace_pca_feature_to_sequence(pca_idx, pca_components, top_n=5)

            if trace_results:
                for j, result in enumerate(trace_results):
                    report_lines.append(f"  {j + 1}. {result['feature_desc']}, weight={result['weight']:.4f}")

                    if result['feature_type'] == 'CKSAAP' and 'aa_pair' in result:
                        aa_pair = result['aa_pair']
                        for aa in aa_pair:
                            if aa in aa_counts:
                                aa_counts[aa] += 1

                        if 'distance_from_center' in result and result['distance_from_center'] is not None:
                            distance = result['distance_from_center']
                            if distance in position_counts:
                                position_counts[distance] += 1
                            else:
                                position_counts[distance] = 1

        report_lines.append("\n" + "=" * 80)
        report_lines.append("Statistical Summary")
        report_lines.append("=" * 80)

        report_lines.append("\nKey amino acid frequency:")
        total_aa_count = sum(aa_counts.values())
        for aa in key_amino_acids:
            count = aa_counts[aa]
            if count > 0:
                frequency = count / total_aa_count * 100 if total_aa_count > 0 else 0
                report_lines.append(f"  {aa}: {count} ({frequency:.1f}%)")

        report_lines.append("\nPosition distribution (distance from center):")
        sorted_positions = sorted(position_counts.items())
        for distance, count in sorted_positions:
            if distance < 0:
                report_lines.append(f"  Left {abs(distance)}: {count}")
            elif distance > 0:
                report_lines.append(f"  Right {distance}: {count}")
            else:
                report_lines.append(f"  Center: {count}")

        report_lines.append("\n" + "=" * 80)
        report_lines.append("Summary")
        report_lines.append("=" * 80)

        if aa_counts['K'] > 0:
            report_lines.append(f"\nFound {aa_counts['K']} features containing Lysine(K).")

        if aa_counts['D'] > 0:
            report_lines.append(f"\nFound {aa_counts['D']} features containing Aspartic acid(D).")

        if aa_counts['P'] > 0:
            report_lines.append(f"\nFound {aa_counts['P']} features containing Proline(P).")

        report_path = os.path.join(save_dir, 'detailed_feature_trace_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"Detailed report saved to: {report_path}")
        print("=" * 80)

        return {
            'aa_counts': aa_counts,
            'position_counts': position_counts,
            'report_path': report_path
        }

    def _generate_f1_summary_report(self, results, save_dir):
        report_lines = [
            "=" * 80,
            "Succinylation Site Prediction - F1-based Interpretability Analysis Report",
            "=" * 80,
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        if 'importances' in results and 'baseline_metrics' in results['importances']:
            baseline = results['importances']['baseline_metrics']
            report_lines.append("Model Performance Baseline:")
            report_lines.append(f"  Accuracy: {baseline['accuracy']:.4f}")
            report_lines.append(f"  Precision: {baseline['precision']:.4f}")
            report_lines.append(f"  Recall: {baseline['recall']:.4f}")
            report_lines.append(f"  F1: {baseline['f1']:.4f}")
            report_lines.append(f"  MCC: {baseline['mcc']:.4f}")
            report_lines.append("")

        if 'top_features_f1' in results and 'importances' in results:
            report_lines.append("Top 10 features by F1 importance:")
            top_indices = results['top_features_f1'][:10]
            f1_importances = results['importances']['f1_importances'][top_indices]
            for idx, imp in zip(top_indices, f1_importances):
                report_lines.append(f"  Feature {idx}: {imp:.6f}")
            report_lines.append("")

        if 'component_contributions' in results:
            comp = results['component_contributions']
            report_lines.append("Model Component F1 Contributions:")
            report_lines.append(f"  Full model F1: {comp['full_f1']:.4f}")
            report_lines.append(f"  MLP F1: {comp['mlp_f1']:.4f} (weight: {comp['weights'][0]:.3f})")
            report_lines.append(f"  PDeepPP F1: {comp['pdeeppp_f1']:.4f} (weight: {comp['weights'][1]:.3f})")
            report_lines.append(f"  DeepFRI F1: {comp['deepfri_f1']:.4f} (weight: {comp['weights'][2]:.3f})")
            report_lines.append(f"  Ensemble gain: {comp['ensemble_gain']:.4f}")
            report_lines.append("")

        if 'prediction_analysis' in results:
            pred = results['prediction_analysis']
            report_lines.append("F1 Threshold Analysis:")
            report_lines.append(f"  Best threshold: {pred['best_threshold']:.3f}")
            report_lines.append(f"  Best F1: {pred['best_f1']:.4f}")
            report_lines.append(f"  Default threshold (0.5) F1: {pred['default_f1']:.4f}")
            report_lines.append(f"  F1 improvement: {pred['best_f1'] - pred['default_f1']:.4f}")
            report_lines.append("")

        if 'correlations' in results:
            corr = results['correlations']
            report_lines.append("Metric Correlation Analysis:")
            report_lines.append(f"  F1 vs Accuracy: {corr['f1_acc_corr']:.3f}")
            report_lines.append(f"  F1 vs Precision: {corr['f1_prec_corr']:.3f}")
            report_lines.append(f"  F1 vs Recall: {corr['f1_recall_corr']:.3f}")
            report_lines.append("")

        report_path = os.path.join(save_dir, 'f1_interpretability_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"Summary report saved to: {report_path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("=" * 80)
    print("Succinylation Site Prediction - F1-based Interpretability Analysis")
    print("=" * 80)

    print("\n1. Loading feature data...")
    try:
        oneHotPos = np.load('combined_features_final/pca_scaled_512/ptrain_combined1.npy')
        oneHotNeg = np.load('combined_features_final/pca_scaled_512/ntrain_combined1.npy')
        test_pos = np.load('combined_features_final/pca_scaled_512/ptest_combined1.npy')
        test_neg = np.load('combined_features_final/pca_scaled_512/ntest_combined1.npy')

        print(f"Training positive samples: {len(oneHotPos)}")
        print(f"Training negative samples: {len(oneHotNeg)}")
        print(f"Test positive samples: {len(test_pos)}")
        print(f"Test negative samples: {len(test_neg)}")
        print(f"Feature dimension: {oneHotPos.shape[1]}")

        X_train = np.concatenate([oneHotPos, oneHotNeg], axis=0)
        y_train = np.concatenate([np.ones(len(oneHotPos)), np.zeros(len(oneHotNeg))])
        X_test = np.concatenate([test_pos, test_neg], axis=0)
        y_test = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("Data preprocessing complete.")

    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    print("\n2. Loading sequence data...")
    sequences = []
    try:
        def load_sequences(filepath):
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
            return []

        seq_files = [
            'trainP_mirror.txt', 'trainP.txt', 'pos_train.txt',
            'trainN_mirror.txt', 'trainN.txt', 'neg_train.txt',
            'testP_mirror.txt', 'testP.txt', 'pos_test.txt',
            'testN_mirror.txt', 'testN.txt', 'neg_test.txt'
        ]

        loaded_files = []
        for file in seq_files:
            if os.path.exists(file):
                seqs = load_sequences(file)
                if seqs:
                    sequences.extend(seqs)
                    loaded_files.append(file)

        if sequences:
            print(f"Loaded {len(sequences)} sequences from:")
            for file in loaded_files:
                print(f"  - {file}")
        else:
            print("No sequence files found.")

    except Exception as e:
        print(f"Sequence loading warning: {e}")

    print("\n3. Loading PCA components...")
    pca_components = None
    try:
        possible_paths = [
            'combined_features_final/pca_info/pca_components.npy',
            'combined_features_final/pca_models/pca_model_512.pkl',
            'pca_components.npy',
            '../pca_components.npy',
            'combined_features_final/pca_scaled_512/pca_model.pkl',
            'pca_model.pkl',
            '../pca_model.pkl'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    if path.endswith('.npy'):
                        pca_components = np.load(path)
                        print(f"Loaded PCA components from {path}: {pca_components.shape}")
                        break
                    elif path.endswith('.pkl'):
                        import joblib
                        pca_model = joblib.load(path)
                        if hasattr(pca_model, 'components_'):
                            pca_components = pca_model.components_
                            print(f"Loaded PCA model from {path}: {pca_components.shape}")
                        break
                except Exception as e:
                    continue

        if pca_components is None:
            print("No PCA components file found.")

    except Exception as e:
        print(f"PCA components loading failed: {e}")

    print("\n4. Loading trained model...")
    model_manager = ModelManager('saved_models')

    model_patterns = ['triple_ensemble_*.pth', '*.pth', '*.pt']
    available_models = []

    for pattern in model_patterns:
        models = glob.glob(os.path.join('saved_models', pattern))
        if not models:
            models = glob.glob(pattern)

        if models:
            available_models.extend(models)

    available_models = list(set(available_models))

    if available_models:
        available_models.sort(key=os.path.getmtime, reverse=True)
        model_path = available_models[0]
        filename = os.path.basename(model_path)
        filesize = os.path.getsize(model_path) / (1024 * 1024)
        filetime = time.strftime('%Y-%m-%d %H:%M:%S',
                                 time.localtime(os.path.getmtime(model_path)))
        print(f"Auto-selected latest model: {filename} ({filesize:.1f} MB, modified: {filetime})")
    else:
        print("No model files found.")

    input_dim = X_train.shape[1]
    print(f"Input feature dimension: {input_dim}")

    model = model_manager.load_trained_model(model_path, input_dim, device)

    if model is None:
        print("Model loading failed.")
        return

    print("\n5. Starting F1-based interpretability analysis...")
    print("=" * 80)

    interpreter = F1BasedInterpretability(model, device)

    results = interpreter.run_f1_based_analysis(
        X_train_scaled, y_train, X_test_scaled, y_test,
        sequences=sequences if sequences else None,
        save_dir='Succinylation_F1_Interpretability_Results',
        pca_components=pca_components
    )

    print("\n" + "=" * 80)
    print("F1-based interpretability analysis complete.")
    print("=" * 80)

    if pca_components is not None and 'importances' in results:
        print("\n6. Generating detailed feature traceback report...")
        try:
            if hasattr(interpreter, 'generate_detailed_feature_report'):
                detailed_report = interpreter.generate_detailed_feature_report(
                    results['importances'],
                    pca_components,
                    top_k=30,
                    save_dir='Succinylation_F1_Interpretability_Results'
                )
                print("Feature traceback report generated.")
        except Exception as e:
            print(f"Detailed report generation failed: {e}")

    print("\n" + "=" * 80)
    print("Analysis Summary")
    print("=" * 80)

    if 'importances' in results and 'baseline_metrics' in results['importances']:
        baseline = results['importances']['baseline_metrics']
        print("Model Performance:")
        print(f"  F1: {baseline['f1']:.4f}")
        print(f"  Accuracy: {baseline['accuracy']:.4f}")
        print(f"  Precision: {baseline['precision']:.4f}")
        print(f"  Recall: {baseline['recall']:.4f}")
        print(f"  MCC: {baseline['mcc']:.4f}")

    if pca_components is not None:
        print(f"PCA components loaded: {pca_components.shape}")
    else:
        print("PCA components not loaded.")

    print("\nResult files:")
    result_dir = 'Succinylation_F1_Interpretability_Results'
    if os.path.exists(result_dir):
        files = glob.glob(os.path.join(result_dir, '*'))
        for file in files[:10]:
            print(f"  {os.path.basename(file)}")
        if len(files) > 10:
            print(f"  ... {len(files)} files total")

    print("\n" + "=" * 80)
    print("Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()