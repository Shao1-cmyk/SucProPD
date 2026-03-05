import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, \
    matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


torch.manual_seed(42)
np.random.seed(42)


class SimpleRNN(nn.Module):
    """RNN"""

    def __init__(self, input_dim, hidden_size=128, num_layers=2, dropout=0.2):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.seq_length = self._calculate_seq_length(input_dim)
        self.feature_per_step = input_dim // self.seq_length


        actual_input_dim = self.seq_length * self.feature_per_step
        if actual_input_dim != input_dim:

            self.input_adjust = nn.Linear(input_dim, actual_input_dim)
        else:
            self.input_adjust = nn.Identity()

        self.rnn = nn.RNN(
            input_size=self.feature_per_step,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity='tanh'
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _calculate_seq_length(self, input_dim):

        possible_lengths = [16, 32, 64, 128]
        for length in possible_lengths:
            if input_dim % length == 0:
                return length
        return 32

    def forward(self, x):
        batch_size = x.size(0)

        x = self.input_adjust(x)
        #  [batch_size, seq_length, features_per_step]
        x = x.view(batch_size, self.seq_length, self.feature_per_step)


        rnn_out, hidden = self.rnn(x)

        last_output = rnn_out[:, -1, :]

        output = self.classifier(last_output)
        return output.squeeze()


class LSTMNet(nn.Module):
    """LSTM"""

    def __init__(self, input_dim, hidden_size=256, num_layers=2, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.seq_length = self._calculate_seq_length(input_dim)
        self.feature_per_step = input_dim // self.seq_length

        actual_input_dim = self.seq_length * self.feature_per_step
        if actual_input_dim != input_dim:
            self.input_adjust = nn.Linear(input_dim, actual_input_dim)
        else:
            self.input_adjust = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=self.feature_per_step,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _calculate_seq_length(self, input_dim):
        possible_lengths = [16, 32, 64, 128]
        for length in possible_lengths:
            if input_dim % length == 0:
                return length
        return 32

    def forward(self, x):
        batch_size = x.size(0)

        x = self.input_adjust(x)

        x = x.view(batch_size, self.seq_length, self.feature_per_step)

        lstm_out, (hidden, cell) = self.lstm(x)

        last_output = lstm_out[:, -1, :]

        output = self.classifier(last_output)
        return output.squeeze()


class BiLSTM(nn.Module):
    """BiLSTM"""

    def __init__(self, input_dim, hidden_size=128, num_layers=2, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.seq_length = self._calculate_seq_length(input_dim)
        self.feature_per_step = input_dim // self.seq_length


        actual_input_dim = self.seq_length * self.feature_per_step
        if actual_input_dim != input_dim:
            self.input_adjust = nn.Linear(input_dim, actual_input_dim)
        else:
            self.input_adjust = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=self.feature_per_step,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_size = hidden_size * 2  # 双向

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _calculate_seq_length(self, input_dim):
        possible_lengths = [16, 32, 64, 128]
        for length in possible_lengths:
            if input_dim % length == 0:
                return length
        return 32

    def forward(self, x):
        batch_size = x.size(0)

        x = self.input_adjust(x)

        x = x.view(batch_size, self.seq_length, self.feature_per_step)

        lstm_out, (hidden, cell) = self.lstm(x)

        last_output = lstm_out[:, -1, :]

        output = self.classifier(last_output)
        return output.squeeze()


class GRUNet(nn.Module):
    """GRU"""

    def __init__(self, input_dim, hidden_size=256, num_layers=2, dropout=0.2):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.seq_length = self._calculate_seq_length(input_dim)
        self.feature_per_step = input_dim // self.seq_length

        actual_input_dim = self.seq_length * self.feature_per_step
        if actual_input_dim != input_dim:
            self.input_adjust = nn.Linear(input_dim, actual_input_dim)
        else:
            self.input_adjust = nn.Identity()

        self.gru = nn.GRU(
            input_size=self.feature_per_step,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _calculate_seq_length(self, input_dim):
        possible_lengths = [16, 32, 64, 128]
        for length in possible_lengths:
            if input_dim % length == 0:
                return length
        return 32

    def forward(self, x):
        batch_size = x.size(0)

        x = self.input_adjust(x)

        x = x.view(batch_size, self.seq_length, self.feature_per_step)

        gru_out, hidden = self.gru(x)

        last_output = gru_out[:, -1, :]

        output = self.classifier(last_output)
        return output.squeeze()


class BiGRU(nn.Module):
    """BiGRU"""

    def __init__(self, input_dim, hidden_size=128, num_layers=2, dropout=0.2):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.seq_length = self._calculate_seq_length(input_dim)
        self.feature_per_step = input_dim // self.seq_length

        actual_input_dim = self.seq_length * self.feature_per_step
        if actual_input_dim != input_dim:
            self.input_adjust = nn.Linear(input_dim, actual_input_dim)
        else:
            self.input_adjust = nn.Identity()

        self.gru = nn.GRU(
            input_size=self.feature_per_step,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        gru_output_size = hidden_size * 2  # bi

        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _calculate_seq_length(self, input_dim):

        possible_lengths = [16, 32, 64, 128]
        for length in possible_lengths:
            if input_dim % length == 0:
                return length
        return 32

    def forward(self, x):
        batch_size = x.size(0)

        x = self.input_adjust(x)

        x = x.view(batch_size, self.seq_length, self.feature_per_step)

        gru_out, hidden = self.gru(x)

        last_output = gru_out[:, -1, :]

        output = self.classifier(last_output)
        return output.squeeze()
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
        "SN": sn,
        "SP": sp,
        "ACC": acc,
        "MCC": mcc,
        "F1": f1,
        "AUC": roc_auc,
        "Precision": precision
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


def rnn_10fold(PPT1, PPT2, model_type='lstm'):

    X = np.concatenate([PPT1, PPT2], axis=0)
    y = np.concatenate([np.ones(len(PPT1)), np.zeros(len(PPT2))])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics = {"SN": [], "SP": [], "ACC": [], "MCC": [], "F1": [], "AUC": [], "Precision": []}

    fold_results = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1}/10 ---")

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
        if model_type == 'rnn':
            model = SimpleRNN(input_dim)
        elif model_type == 'lstm':
            model = LSTMNet(input_dim)
        elif model_type == 'bilstm':
            model = BiLSTM(input_dim)
        elif model_type == 'gru':
            model = GRUNet(input_dim)
        elif model_type == 'bigru':
            model = BiGRU(input_dim)
        elif model_type == 'mlp':
            model = MLP(input_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        print(f"Training {model_type.upper()} model for fold {fold + 1}...")
        model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100)

        y_pred, y_probs, y_true = evaluate_model(model, test_loader)
        fold_metrics = calculate_all_metrics(y_true, y_pred, y_probs)

        for key in metrics:
            metrics[key].append(fold_metrics[key])

        fold_result = {'Fold': fold + 1}
        fold_result.update(fold_metrics)
        fold_results.append(fold_result)

        print(f"{model_type.upper()} - Fold {fold + 1} Results:")
        for metric, value in fold_metrics.items():
            print(f"  {metric}: {value:.4f}")

    final_results = {}
    for key in metrics:
        values = metrics[key]
        final_results[f'{key}_mean'] = np.mean(values)
        final_results[f'{key}_std'] = np.std(values)

    print(f"\n{model_type.upper()} - 10-Fold Cross Validation Summary:")
    print("Metric\t\tMean ± Std")
    print("-" * 30)
    for key in ['SN', 'SP', 'ACC', 'MCC', 'F1', 'AUC', 'Precision']:
        mean_val = final_results[f'{key}_mean']
        std_val = final_results[f'{key}_std']
        print(f"{key}\t\t{mean_val:.4f} ± {std_val:.4f}")

    return final_results, fold_results


def rnn_test(train_pos, train_neg, test_pos, test_neg, model_type='lstm'):

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
    if model_type == 'rnn':
        model = SimpleRNN(input_dim)
    elif model_type == 'lstm':
        model = LSTMNet(input_dim)
    elif model_type == 'bilstm':
        model = BiLSTM(input_dim)
    elif model_type == 'gru':
        model = GRUNet(input_dim)
    elif model_type == 'bigru':
        model = BiGRU(input_dim)
    elif model_type == 'mlp':
        model = MLP(input_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    print(f"Training {model_type.upper()} model for independent test...")
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

    rnn_models = ['lstm','gru','mlp','rnn','bilstm','bigru',]

    all_10fold_results = {}
    all_test_results = {}

    for model_type in rnn_models:
        print("\n" + "=" * 80)
        print(f"TESTING {model_type.upper()} MODEL")
        print("=" * 80)

        try:

            print(f"\n10-FOLD CROSS VALIDATION RESULTS ({model_type.upper()})")
            cv_final_results, cv_fold_results = rnn_10fold(oneHotPos, oneHotNeg, model_type=model_type)

            print(f"\nINDEPENDENT TEST RESULTS ({model_type.upper()})")
            test_results = rnn_test(oneHotPos, oneHotNeg, test_pos, test_neg, model_type=model_type)

            all_10fold_results[model_type] = cv_final_results
            all_test_results[model_type] = test_results

        except Exception as e:
            print(f"Error testing {model_type}: {e}")
            continue


    if all_10fold_results:

        cv_summary_data = []
        test_data = []

        for model_name, cv_results in all_10fold_results.items():

            cv_row = {'Model': model_name}
            for metric in ['SN', 'SP', 'ACC', 'MCC', 'F1', 'AUC', 'Precision']:
                cv_row[f'{metric}_mean'] = cv_results[f'{metric}_mean']
                cv_row[f'{metric}_std'] = cv_results[f'{metric}_std']
            cv_summary_data.append(cv_row)


            test_row = {'Model': model_name}
            test_row.update(all_test_results[model_name])
            test_data.append(test_row)

        cv_summary_df = pd.DataFrame(cv_summary_data)
        test_df = pd.DataFrame(test_data)

     
        cv_summary_df.to_csv('rnn_models_10fold_summary.csv', index=False, float_format='%.4f')
        test_df.to_csv('rnn_models_independent_test.csv', index=False, float_format='%.4f')

        print("\n" + "=" * 100)
        print("FINAL RESULTS SUMMARY")
        print("=" * 100)

        print("\n10-FOLD CROSS VALIDATION (Mean ± Std):")
        print("Model\t\tSN\t\tSP\t\tACC\t\tMCC\t\tF1\t\tAUC\t\tPrecision")
        print("-" * 120)
        for model_name in rnn_models:
            if model_name in all_10fold_results:
                cv = all_10fold_results[model_name]
                print(f"{model_name:8}\t"
                      f"{cv['SN_mean']:.4f}±{cv['SN_std']:.3f}\t"
                      f"{cv['SP_mean']:.4f}±{cv['SP_std']:.3f}\t"
                      f"{cv['ACC_mean']:.4f}±{cv['ACC_std']:.3f}\t"
                      f"{cv['MCC_mean']:.4f}±{cv['MCC_std']:.3f}\t"
                      f"{cv['F1_mean']:.4f}±{cv['F1_std']:.3f}\t"
                      f"{cv['AUC_mean']:.4f}±{cv['AUC_std']:.3f}\t"
                      f"{cv['Precision_mean']:.4f}±{cv['Precision_std']:.3f}")

        print("\nINDEPENDENT TEST:")
        print("Model\t\tSN\t\tSP\t\tACC\t\tMCC\t\tF1\t\tAUC\t\tPrecision")
        print("-" * 100)
        for model_name in rnn_models:
            if model_name in all_test_results:
                test = all_test_results[model_name]
                print(f"{model_name:8}\t"
                      f"{test['SN']:.4f}\t"
                      f"{test['SP']:.4f}\t"
                      f"{test['ACC']:.4f}\t"
                      f"{test['MCC']:.4f}\t"
                      f"{test['F1']:.4f}\t"
                      f"{test['AUC']:.4f}\t"
                      f"{test['Precision']:.4f}")

        print(f"\nResults saved to:")
        print(f"10-fold CV summary: rnn_models_10fold_summary.csv")
        print(f"Independent test: rnn_models_independent_test.csv")


if __name__ == "__main__":
    main()