import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import joblib
import time


def load_features():
    features = {}

    features['ptrain_ksp'] = np.load('ptrain_KSP.npy')
    features['ntrain_ksp'] = np.load('ntrain_KSP.npy')
    features['ptest_ksp'] = np.load('ptest_KSP.npy')
    features['ntest_ksp'] = np.load('ntest_KSP.npy')

    features['ptrain_prott5'] = np.load('ptrainProtT5.npy')
    features['ntrain_prott5'] = np.load('ntrainProtT5.npy')
    features['ptest_prott5'] = np.load('ptestProtT5.npy')
    features['ntest_prott5'] = np.load('ntestProtT5.npy')

    return features


def combine_and_save_features(features, output_dir='combined_features_final'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ptrain_concat = np.concatenate([features['ptrain_prott5'], features['ptrain_ksp']], axis=1)
    ntrain_concat = np.concatenate([features['ntrain_prott5'], features['ntrain_ksp']], axis=1)
    ptest_concat = np.concatenate([features['ptest_prott5'], features['ptest_ksp']], axis=1)
    ntest_concat = np.concatenate([features['ntest_prott5'], features['ntest_ksp']], axis=1)

    X_train_all = np.vstack([ptrain_concat, ntrain_concat])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_all)

    target_dim = 512
    if ptrain_concat.shape[1] < target_dim:
        target_dim = ptrain_concat.shape[1]

    pca = PCA(n_components=target_dim, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)

    ptrain_pca = pca.transform(scaler.transform(ptrain_concat))
    ntrain_pca = pca.transform(scaler.transform(ntrain_concat))
    ptest_pca = pca.transform(scaler.transform(ptest_concat))
    ntest_pca = pca.transform(scaler.transform(ntest_concat))

    pca_scaled_dir = os.path.join(output_dir, 'pca_scaled_512')
    os.makedirs(pca_scaled_dir, exist_ok=True)

    np.save(os.path.join(pca_scaled_dir, 'ptrain_combined1.npy'), ptrain_pca)
    np.save(os.path.join(pca_scaled_dir, 'ntrain_combined1.npy'), ntrain_pca)
    np.save(os.path.join(pca_scaled_dir, 'ptest_combined1.npy'), ptest_pca)
    np.save(os.path.join(pca_scaled_dir, 'ntest_combined1.npy'), ntest_pca)

    model_dir = os.path.join(output_dir, 'pca_models')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, 'standard_scaler.pkl'))
    joblib.dump(pca, os.path.join(model_dir, 'pca_model_512.pkl'))

    pca_info_dir = os.path.join(output_dir, 'pca_info')
    os.makedirs(pca_info_dir, exist_ok=True)
    np.save(os.path.join(pca_info_dir, 'pca_components.npy'), pca.components_)
    np.save(os.path.join(pca_info_dir, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(pca_info_dir, 'scaler_scale.npy'), scaler.scale_)

    print(f"Done")
   


def main():
    features = load_features()
    combine_and_save_features(features)


if __name__ == "__main__":
    main()