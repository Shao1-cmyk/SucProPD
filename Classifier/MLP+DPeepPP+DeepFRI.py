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
            nn.Linear(3, 16),  # 3个模型的输出
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes),
            nn.Sigmoid()
        )

        self.weights = nn.Parameter(torch.ones(3) / 3)  # 初始等权重

    def forward(self, x):

        mlp_output = self.mlp(x).unsqueeze(1)  # [batch_size, 1]
        pdeeppp_output = self.pdeeppp(x).unsqueeze(1)  # [batch_size, 1]
        deepfri_output = self.deepfri(x).unsqueeze(1)  # [batch_size, 1]

        combined = torch.cat([mlp_output, pdeeppp_output, deepfri_output], dim=1)  # [batch_size, 3]

        weighted_combined = combined * F.softmax(self.weights, dim=0)

        ensemble_output = self.classifier(weighted_combined)

        return ensemble_output.squeeze()


# ==================== 训练和评估函数 ====================
def calculate_all_metrics(y_true, y_pred, y_probs):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    return {
        "SN": sn, "SP": sp, "ACC": acc, "MCC": mcc,
        "F1": f1, "AUC": roc_auc, "Precision": precision
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=100, patience=10):

    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if scheduler is not None:
            scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def evaluate_model(model, test_loader):

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def triple_ensemble_10fold(PPT1, PPT2, model_type='triple_ensemble'):

    X = np.concatenate([PPT1, PPT2], axis=0)
    y = np.concatenate([np.ones(len(PPT1)), np.zeros(len(PPT2))])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics = {"SN": [], "SP": [], "ACC": [], "MCC": [], "F1": [], "AUC": [], "Precision": []}

    fold_results = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"\n--- Triple Ensemble Fold {fold + 1}/10 ---")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        val_size = int(0.1 * len(X_train))
        X_train, X_val = X_train[val_size:], X_train[:val_size]
        y_train, y_val = y_train[val_size:], y_train[:val_size]

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        input_dim = X.shape[1]
        model = TripleEnsemble(input_dim)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        print(f"Training Triple Ensemble model for fold {fold + 1}...")
        model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, patience=10)

        y_pred, y_probs, y_true = evaluate_model(model, test_loader)
        fold_metrics = calculate_all_metrics(y_true, y_pred, y_probs)

        for key in metrics:
            metrics[key].append(fold_metrics[key])

        fold_result = {'Fold': fold + 1}
        fold_result.update(fold_metrics)
        fold_results.append(fold_result)

        print(f"Triple Ensemble - Fold {fold + 1} Results:")
        for metric, value in fold_metrics.items():
            print(f"  {metric}: {value:.4f}")

    final_results = {}
    for key in metrics:
        values = metrics[key]
        final_results[f'{key}_mean'] = np.mean(values)
        final_results[f'{key}_std'] = np.std(values)

    print(f"\nTriple Ensemble - 10-Fold Cross Validation Summary:")
    print("Metric\t\tMean ± Std")
    print("-" * 30)
    for key in ['SN', 'SP', 'ACC', 'MCC', 'F1', 'AUC', 'Precision']:
        mean_val = final_results[f'{key}_mean']
        std_val = final_results[f'{key}_std']
        print(f"{key}\t\t{mean_val:.4f} ± {std_val:.4f}")

    return final_results, fold_results



def triple_ensemble_test(train_pos, train_neg, test_pos, test_neg, model_type='triple_ensemble'):

    X_train = np.concatenate([train_pos, train_neg], axis=0)
    y_train = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
    X_test = np.concatenate([test_pos, test_neg], axis=0)
    y_test = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])

    print("Training set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_test.shape, y_test.shape)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    val_size = int(0.1 * len(X_train))
    X_train, X_val = X_train[val_size:], X_train[:val_size]
    y_train, y_val = y_train[val_size:], y_train[:val_size]

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = X_train.shape[1]
    model = TripleEnsemble(input_dim)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"Training Triple Ensemble model for independent test...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, patience=10)

    y_pred, y_probs, y_true = evaluate_model(model, test_loader)
    results = calculate_all_metrics(y_true, y_pred, y_probs)

    return results



def main():

    print("Loading data...")
    oneHotPos = np.load('combined_features_final/pca_scaled_512/ptrain_combined1.npy')
    oneHotNeg = np.load('combined_features_final/pca_scaled_512/ntrain_combined1.npy')
    test_pos = np.load('combined_features_final/pca_scaled_512/ptest_combined1.npy')
    test_neg = np.load('combined_features_final/pca_scaled_512/ntest_combined1.npy')

    print("Number of positive training samples:", len(oneHotPos))
    print("Number of negative training samples:", len(oneHotNeg))
    print("Number of positive test samples:", len(test_pos))
    print("Number of negative test samples:", len(test_neg))
    print("Feature dimension:", oneHotPos.shape[1])

    print("\n" + "=" * 80)
    print("TESTING TRIPLE ENSEMBLE MODEL (MLP + PDeepPP + DeepFRI)")
    print("=" * 80)

    try:

        print(f"\n10-FOLD CROSS VALIDATION RESULTS (Triple Ensemble)")
        cv_final_results, cv_fold_results = triple_ensemble_10fold(oneHotPos, oneHotNeg, model_type='triple_ensemble')

        print(f"\nINDEPENDENT TEST RESULTS (Triple Ensemble)")
        test_results = triple_ensemble_test(oneHotPos, oneHotNeg, test_pos, test_neg, model_type='triple_ensemble')

        print("\n" + "=" * 100)
        print("TRIPLE ENSEMBLE MODEL FINAL RESULTS")
        print("=" * 100)

        print("\n10-FOLD CROSS VALIDATION (Mean ± Std):")
        print("Metric\t\tMean ± Std")
        print("-" * 30)
        for key in ['SN', 'SP', 'ACC', 'MCC', 'F1', 'AUC', 'Precision']:
            mean_val = cv_final_results[f'{key}_mean']
            std_val = cv_final_results[f'{key}_std']
            print(f"{key}\t\t{mean_val:.4f} ± {std_val:.4f}")

        print("\nINDEPENDENT TEST:")
        for metric, value in test_results.items():
            print(f"{metric}: {value:.4f}")

        cv_df = pd.DataFrame(cv_fold_results)
        cv_df.to_csv('triple_ensemble_10fold_results.csv', index=False)

        summary_data = {
            'Model': ['Triple_Ensemble'],
            'SN_mean': [cv_final_results['SN_mean']],
            'SN_std': [cv_final_results['SN_std']],
            'SP_mean': [cv_final_results['SP_mean']],
            'SP_std': [cv_final_results['SP_std']],
            'ACC_mean': [cv_final_results['ACC_mean']],
            'ACC_std': [cv_final_results['ACC_std']],
            'MCC_mean': [cv_final_results['MCC_mean']],
            'MCC_std': [cv_final_results['MCC_std']],
            'AUC_mean': [cv_final_results['AUC_mean']],
            'AUC_std': [cv_final_results['AUC_std']]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('triple_ensemble_summary.csv', index=False)

        print(f"\nResults saved to:")
        print(f"- triple_ensemble_10fold_results.csv")
        print(f"- triple_ensemble_summary.csv")

    except Exception as e:
        print(f"Error testing Triple Ensemble model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()