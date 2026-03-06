import numpy as np
import os
from tqdm import tqdm
from itertools import product


def readFastaFile(fastaFile):
    """Read FASTA file"""
    with open(fastaFile, 'r') as f:
        lines = f.readlines()

    sequences = []
    sequence_ids = []
    sequence = ''

    has_header = any(line.startswith('>') for line in lines)

    if has_header:
        for line in lines:
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
                sequence_ids.append(line.strip()[1:])
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
    else:
        for line in lines:
            if line.strip():
                sequences.append(line.strip())
                sequence_ids.append(f"seq_{len(sequences)}")

    return sequences, sequence_ids


def extract_ksp_features(sequences, k=3):
    """Extract k-spaced amino acid pair features"""
    print(f"Extracting {k}-spaced amino acid pair features...")
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    ksp_features = []

    for seq in tqdm(sequences, desc=f"KSP{k} feature extraction"):
        seq = seq.upper()
        ksp_vector = []

        # Calculate frequencies of amino acid pairs with different gaps
        for gap in range(1, k + 1):
            for aa1 in amino_acids:
                for aa2 in amino_acids:
                    count = 0
                    for i in range(len(seq) - gap - 1):
                        if seq[i] == aa1 and seq[i + gap + 1] == aa2:
                            count += 1
                    total_pairs = len(seq) - gap - 1
                    frequency = count / total_pairs if total_pairs > 0 else 0
                    ksp_vector.append(frequency)

        ksp_features.append(ksp_vector)

    return np.array(ksp_features)


def process_dataset_ksp(sequences, output_prefix):
    """
    Process a single dataset and save KSP features
    """
    print(f"Processing {len(sequences)} sequences...")

    if sequences:
        # Extract KSP features
        print(f"Extracting KSP features...")
        ksp_features = extract_ksp_features(sequences, k=3)
        np.save(f'{output_prefix}_KSP.npy', ksp_features)
        print(f"Saved {output_prefix}_KSP.npy, feature dimension: {ksp_features.shape}")

        return ksp_features
    else:
        print(f"No valid sequences to process")
        return None


def test_ksp_feature_extraction():
    """Test KSP feature extraction functionality"""
    test_sequences = [
        "ACDEFGHIK",
        "ACDEFGHIKLMNPQRSTVWY",
        "MKTVRQERLKSIVRILERSK"
    ]

    print("Testing KSP feature extraction...")
    try:
        ksp_features = extract_ksp_features(test_sequences, k=3)

        # Check feature variance
        if len(ksp_features) >= 2:
            diff_ksp = np.abs(ksp_features[0] - ksp_features[1]).mean()
            print(f"KSP feature mean difference: {diff_ksp:.6f}")

            if diff_ksp < 1e-6:
                print("⚠️ Warning: Some features show low variance")
            else:
                print("✓ All feature extraction working normally")

        return True
    except Exception as e:
        print(f"Test extraction failed: {e}")
        return False


# Main program
if __name__ == "__main__":
    # Read sequence data
    print("Reading training positive samples...")
    trainPos, _ = readFastaFile('trainP_mirror.txt')
    print("Reading training negative samples...")
    trainNeg, _ = readFastaFile('trainN_mirror.txt')
    print("Reading test positive samples...")
    testPos, _ = readFastaFile('testP_mirror.txt')
    print("Reading test negative samples...")
    testNeg, _ = readFastaFile('testN_mirror.txt')

    print("\nStarting KSP feature extraction...")

    # First test feature extraction functionality
    if not test_ksp_feature_extraction():
        print("Feature extraction test failed, exiting program")
        exit(1)

    # Process all data
    print("\nProcessing training positive samples...")
    process_dataset_ksp(trainPos, 'ptrain')

    print("\nProcessing training negative samples...")
    process_dataset_ksp(trainNeg, 'ntrain')

    print("\nProcessing test positive samples...")
    process_dataset_ksp(testPos, 'ptest')

    print("\nProcessing test negative samples...")
    process_dataset_ksp(testNeg, 'ntest')

    print("\nKSP feature extraction completed!")
    print("Generated files:")
    print("KSP features: ptrain_KSP.npy, ntrain_KSP.npy, ptest_KSP.npy, ntest_KSP.npy")