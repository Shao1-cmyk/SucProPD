import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, \
    matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)


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
            nn.AdaptiveAvgPool1d(25)  # 统一输出长度
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
            print(f"CNN output dimension: {self.cnn_output_dim}")

    def forward(self, x):
        batch_size = x.shape[0]

        projected = self.input_projection(x)  # [batch, embed_size]

        cnn_input = x.view(batch_size, 1, -1)

        if cnn_input.shape[2] < self.seq_len:

            pad_size = self.seq_len - cnn_input.shape[2]
            cnn_input = F.pad(cnn_input, (0, pad_size))
        else:

            cnn_input = cnn_input[:, :, :self.seq_len]

        cnn_features = self.cnn_layers(cnn_input)
        cnn_features = cnn_features.view(batch_size, -1)

        transformer_input = projected.unsqueeze(1).repeat(1, 5, 1)  # 创建长度为5的序列

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


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10):

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


def pdeeppp_cv_train_test(PPT1, PPT2, test_pos, test_neg, model_type='full'):


    X = np.concatenate([PPT1, PPT2], axis=0)
    y = np.concatenate([np.ones(len(PPT1)), np.zeros(len(PPT2))])

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics = {"SN": [], "SP": [], "ACC": [], "MCC": [], "F1": [], "AUC": [], "Precision": []}

    fold_results = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"\n--- PDeepPP Fold {fold + 1}/10 ---")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

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
        if model_type == 'full':
            model = PDeepPP(input_dim=input_dim, dropout=0.3)


        print(f"Using {model_type} PDeepPP model with input_dim={input_dim}")

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100)

        y_pred, y_probs, y_true = evaluate_model(model, test_loader)
        fold_metrics = calculate_all_metrics(y_true, y_pred, y_probs)

        for key in metrics:
            metrics[key].append(fold_metrics[key])

        fold_result = {'Fold': fold + 1}
        fold_result.update(fold_metrics)
        fold_results.append(fold_result)

        print(f"PDeepPP - Fold {fold + 1} Results:")
        for metric, value in fold_metrics.items():
            print(f"  {metric}: {value:.4f}")

    cv_final_results = {}
    for key in metrics:
        values = metrics[key]
        if values:
            cv_final_results[f'{key}_mean'] = np.mean(values)
            cv_final_results[f'{key}_std'] = np.std(values)
        else:
            cv_final_results[f'{key}_mean'] = 0.0
            cv_final_results[f'{key}_std'] = 0.0

    print(f"\nPDeepPP - 10-Fold Cross Validation Summary:")
    print("Metric\t\tMean ± Std")
    print("-" * 30)
    for key in ['SN', 'SP', 'ACC', 'MCC', 'F1', 'AUC', 'Precision']:
        mean_val = cv_final_results[f'{key}_mean']
        std_val = cv_final_results[f'{key}_std']
        print(f"{key}\t\t{mean_val:.4f} ± {std_val:.4f}")

    print(f"\nPDeepPP INDEPENDENT TEST RESULTS")
    test_results = pdeeppp_independent_test(PPT1, PPT2, test_pos, test_neg, model_type=model_type)

    return cv_final_results, fold_results, test_results


def pdeeppp_independent_test(train_pos, train_neg, test_pos, test_neg, model_type='full'):


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
    if model_type == 'full':
        model = PDeepPP(input_dim=input_dim, dropout=0.3)


    print(f"Using {model_type} PDeepPP model for independent test")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100)

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
    print("TESTING FULL PDEEPPP MODEL")
    print("=" * 80)

    try:
        cv_final_results, cv_fold_results, test_results = pdeeppp_cv_train_test(
            oneHotPos, oneHotNeg, test_pos, test_neg, model_type='full'
        )

        print("\nPDeepPP Final Results:")
        print("10-Fold CV Results:")
        for key in ['SN', 'SP', 'ACC', 'MCC', 'F1', 'AUC', 'Precision']:
            mean_val = cv_final_results[f'{key}_mean']
            std_val = cv_final_results[f'{key}_std']
            print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")

        print("\nIndependent Test Results:")
        for metric, value in test_results.items():
            print(f"{metric}: {value:.4f}")

    except Exception as e:
        print(f"Error testing PDeepPP: {e}")


if __name__ == "__main__":
    main()