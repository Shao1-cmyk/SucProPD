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


class ESMFineTuner(nn.Module):


    def __init__(self, input_dim=512, esm_hidden_dim=1280, dropout=0.2):
        super(ESMFineTuner, self).__init__()


        self.feature_enhancer = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, esm_hidden_dim),
            nn.LayerNorm(esm_hidden_dim),
            nn.GELU()
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=esm_hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(esm_hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        enhanced_features = self.feature_enhancer(x)

        batch_size = x.size(0)
        seq_features = enhanced_features.unsqueeze(1)  # [batch, 1, hidden_dim]

        attn_output, attn_weights = self.multihead_attn(
            seq_features, seq_features, seq_features
        )

        pooled = attn_output.mean(dim=1)

        output = self.classifier(pooled)
        return output.squeeze()


def calculate_all_metrics(y_true, y_pred, y_probs):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sn = tp / (tp + fn) if (tp + fn) > 0 else 0

    sp = tn / (tn + fp) if (tn + fp) > 0 else 0

    acc = accuracy_score(y_true, y_pred)

    mcc = matthews_corrcoef(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    precision = precision_score(y_true, y_pred)

    return {
        "SN": sn, "SP": sp, "ACC": acc, "MCC": mcc,
        "F1": f1, "AUC": roc_auc, "Precision": precision
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=100, patience=10):

    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

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

        train_losses.append(train_loss)
        val_losses.append(val_loss)

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

    return model, train_losses, val_losses


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


def esm_cv_train_test(PPT1, PPT2, test_pos, test_neg, esm_hidden_dim=1280, dropout=0.2):


    X = np.concatenate([PPT1, PPT2], axis=0)
    y = np.concatenate([np.ones(len(PPT1)), np.zeros(len(PPT2))])

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics = {"SN": [], "SP": [], "ACC": [], "MCC": [], "F1": [], "AUC": [], "Precision": []}

    fold_results = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"\n--- ESMFineTuner Fold {fold + 1}/10 ---")

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
        model = ESMFineTuner(input_dim=input_dim, esm_hidden_dim=esm_hidden_dim, dropout=dropout)

        print(f"ESMFineTuner Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, patience=10
        )

        y_pred, y_probs, y_true = evaluate_model(model, test_loader)
        fold_metrics = calculate_all_metrics(y_true, y_pred, y_probs)

        for key in metrics:
            metrics[key].append(fold_metrics[key])

        fold_result = {'Fold': fold + 1}
        fold_result.update(fold_metrics)
        fold_results.append(fold_result)

        print(f"ESMFineTuner - Fold {fold + 1} Results:")
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

    print(f"\nESMFineTuner - 10-Fold Cross Validation Summary:")
    print("Metric\t\tMean ± Std")
    print("-" * 30)
    for key in ['SN', 'SP', 'ACC', 'MCC', 'F1', 'AUC', 'Precision']:
        mean_val = cv_final_results[f'{key}_mean']
        std_val = cv_final_results[f'{key}_std']
        print(f"{key}\t\t{mean_val:.4f} ± {std_val:.4f}")

    print(f"\nESMFineTuner INDEPENDENT TEST RESULTS")
    test_results = esm_independent_test(PPT1, PPT2, test_pos, test_neg, esm_hidden_dim, dropout)

    return cv_final_results, fold_results, test_results


def esm_independent_test(train_pos, train_neg, test_pos, test_neg, esm_hidden_dim=1280, dropout=0.2):


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
    model = ESMFineTuner(input_dim=input_dim, esm_hidden_dim=esm_hidden_dim, dropout=dropout)

    print(f"ESMFineTuner Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, patience=10
    )

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
    print("TESTING ESMFineTuner MODEL")
    print("=" * 80)

    try:

        esm_hidden_dim = 1280
        dropout = 0.2

        cv_final_results, cv_fold_results, test_results = esm_cv_train_test(
            oneHotPos, oneHotNeg, test_pos, test_neg,
            esm_hidden_dim=esm_hidden_dim, dropout=dropout
        )

        print("\n" + "=" * 60)
        print("ESMFineTuner FINAL RESULTS")
        print("=" * 60)

        print("\n10-Fold Cross Validation Results:")
        print("Metric\t\tMean ± Std")
        print("-" * 30)
        for key in ['SN', 'SP', 'ACC', 'MCC', 'F1', 'AUC', 'Precision']:
            mean_val = cv_final_results[f'{key}_mean']
            std_val = cv_final_results[f'{key}_std']
            print(f"{key}\t\t{mean_val:.4f} ± {std_val:.4f}")

        print("\nIndependent Test Results:")
        for metric, value in test_results.items():
            print(f"{metric}: {value:.4f}")

        cv_df = pd.DataFrame(cv_fold_results)
        cv_df.to_csv('esm_finetuner_10fold_results.csv', index=False)

        summary_data = {
            'Model': ['ESMFineTuner'],
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
        summary_df.to_csv('esm_finetuner_summary.csv', index=False)

        print(f"\nResults saved to:")
        print(f"- esm_finetuner_10fold_results.csv")
        print(f"- esm_finetuner_summary.csv")

    except Exception as e:
        print(f"Error testing ESMFineTuner: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()