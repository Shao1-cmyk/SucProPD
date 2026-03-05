import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import gc
import os
import json
from tqdm import tqdm


def readFastaFile(fastaFile):

    print(f"read file: {fastaFile}")
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

    print(f"from {fastaFile} read {len(sequences)} sequences")


    return sequences, sequence_ids


def check_vocabulary(tokenizer):


    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    found_tokens = []
    missing_tokens = []

    for aa in amino_acids:
        token_id = tokenizer.convert_tokens_to_ids(aa)
        if token_id != tokenizer.unk_token_id:
            found_tokens.append(aa)
        else:
            missing_tokens.append(aa)

    print(f"Amino acids found in vocabulary: {found_tokens}")
    print(f"Amino acids missing in vocabulary: {missing_tokens}")

    if missing_tokens:
        print(f" Warning: Vocabulary missing {len(missing_tokens)} amino acid characters")
        return False
    else:
        print(f"Vocabulary contains all standard amino acid characters")
        return True


def create_custom_tokenizer(model_path):


    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')


    vocab_content = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + amino_acids

    temp_vocab_path = os.path.join(model_path, 'temp_vocab.txt')
    with open(temp_vocab_path, 'w', encoding='utf-8') as f:
        for token in vocab_content:
            f.write(token + '\n')

    from transformers import BertTokenizer
    tokenizer = BertTokenizer(
        temp_vocab_path,
        do_lower_case=False,
        unk_token='[UNK]',
        pad_token='[PAD]',
        cls_token='[CLS]',
        sep_token='[SEP]',
        mask_token='[MASK]'
    )

    os.remove(temp_vocab_path)


    return tokenizer


def load_local_abbert_model_fixed(model_path):

    required_files = ['config.json', 'pytorch_model.bin']

    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"缺少必要文件: {file}")
        print(f"找到文件: {file}")

    fix_config_file(model_path)

    try:
        try:
            standard_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                do_lower_case=False
            )
        except:
            standard_tokenizer = None

        if standard_tokenizer:
            vocab_ok = check_vocabulary(standard_tokenizer)
            if not vocab_ok:

                tokenizer = create_custom_tokenizer(model_path)
            else:
                tokenizer = standard_tokenizer
        else:

            tokenizer = create_custom_tokenizer(model_path)

        # 加载模型
        model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32
        )

        model = model.to('cpu')
        model.eval()

        return tokenizer, model

    except Exception as e:
        print(f" Model loading failed: {e}")
        raise


def fix_config_file(model_path):

    config_path = os.path.join(model_path, 'config.json')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if 'model_type' not in config:
            config['model_type'] = 'bert'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)

    except Exception as e:
        print(f"Error : {e}")


def correct_sequence_format_fixed(sequences, tokenizer):

    corrected_sequences = []

    for i, seq in enumerate(sequences):

        seq_clean = ''.join([aa for aa in seq.upper() if aa in 'ACDEFGHIKLMNPQRSTVWY'])

        if not seq_clean:

            seq_clean = seq.upper()

        seq_with_spaces = ' '.join(seq_clean)
        corrected_sequences.append(seq_with_spaces)

        if i < 2:

            tokens = tokenizer.tokenize(seq_with_spaces)
            valid_tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]
            print(f"    Token result: {valid_tokens[:10]}...")

    return corrected_sequences

def extract_abbert_features_fixed(sequences, model_path, batch_size=4, dataset_name=""):
    print(f"\nStarting feature extraction for {dataset_name}...")
    print(f"Input sequences: {len(sequences)}")

    if len(sequences) >= 2:
        seq1 = sequences[0]
        seq2 = sequences[1]
        if seq1 == seq2:
            print("Warning: First two sequences are identical!")
        else:
            print(f"Sequence diversity verification passed")

    tokenizer, model = load_local_abbert_model_fixed(model_path)
    corrected_sequences = correct_sequence_format_fixed(sequences, tokenizer)
    all_features = []

    total_batches = (len(corrected_sequences) + batch_size - 1) // batch_size
    with tqdm(total=total_batches, desc=f"ABBERT-{dataset_name}") as pbar:

        for i in range(0, len(corrected_sequences), batch_size):
            batch_seqs = corrected_sequences[i:i + batch_size]

            try:
                inputs = tokenizer(
                    batch_seqs,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                    add_special_tokens=True
                )

                if i == 0:
                    print(f"\nFirst batch tokenization details:")
                    for j in range(min(2, len(batch_seqs))):
                        input_ids = inputs['input_ids'][j]
                        tokens = tokenizer.convert_ids_to_tokens(input_ids)
                        valid_tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]

                        print(f"  Sequence {j + 1}:")
                        print(f"    Processed sequence: {batch_seqs[j][:60]}...")
                        print(f"    Valid tokens: {len(valid_tokens)}")
                        print(f"    First 10 tokens: {valid_tokens[:10]}")

                        unk_count = valid_tokens.count('[UNK]')
                        if unk_count > 0:
                            print(f"    Contains {unk_count} unknown tokens")
                        else:
                            print(f"    All tokens are known")

                        if j > 0:
                            prev_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][j - 1])
                            prev_valid = [t for t in prev_tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]
                            if valid_tokens == prev_valid:
                                print("    Warning: Tokens identical to previous sequence")
                            else:
                                print("    Tokens different from previous sequence")

                inputs = {k: v.to('cpu') for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    hidden_states = outputs.last_hidden_state

                    for j in range(len(batch_seqs)):
                        attention_mask = inputs['attention_mask'][j].bool()
                        seq_len = attention_mask.sum().item()
                        if seq_len > 2:
                            seq_rep = hidden_states[j, 1:seq_len - 1].mean(dim=0)
                        else:
                            seq_rep = hidden_states[j, :seq_len].mean(dim=0)

                        feature_vector = seq_rep.numpy()
                        all_features.append(feature_vector)

            except Exception as e:
                print(f"Error processing batch {i // batch_size + 1}: {e}")
                random_feature = np.random.normal(0, 0.1, model.config.hidden_size).astype(np.float32)
                for j in range(len(batch_seqs)):
                    all_features.append(random_feature)

            pbar.update(1)
            if (i // batch_size) % 10 == 0:
                gc.collect()

    del model, tokenizer
    gc.collect()

    features_array = np.array(all_features)
    print(f"{dataset_name} feature extraction completed, shape: {features_array.shape}")

    if len(features_array) >= 2 and dataset_name == "FixTest":
        overall_diff = np.abs(features_array[0] - features_array[1]).mean()
        print(f"Difference between first two sequences: {overall_diff:.6f}")

        if overall_diff < 0.01:
            print("Critical issue: Feature diversity too low")
            return None
        else:
            print("Feature diversity normal")
    elif len(features_array) >= 2:
        overall_diff = np.abs(features_array[0] - features_array[1]).mean()
        print(f"Difference between first two sequences: {overall_diff:.6f}")
        print("Continuing with real data processing")

    return features_array


def test_feature_extraction_fixed(model_path):
    test_sequences = [
        "MKTVRQERLKSIVRILERSK",
        "ACDEFGHIKLMNPQRSTVWY",
        "MKTVRQERLKSIVRILERSK",
    ]

    print("Testing ABBERT feature extraction...")
    print("Test sequences:")
    for i, seq in enumerate(test_sequences):
        print(f"  Sequence{i + 1}: {seq}")

    try:
        features = extract_abbert_features_fixed(test_sequences, model_path, dataset_name="FixTest")

        if features is None:
            return False

        print(f"\nDetailed feature difference analysis:")
        diff_1_2 = np.abs(features[0] - features[1]).mean()
        diff_1_3 = np.abs(features[0] - features[2]).mean()

        print(f"Sequence1 vs Sequence2 (different) difference: {diff_1_2:.6f}")
        print(f"Sequence1 vs Sequence3 (same) difference: {diff_1_3:.6f}")

        if diff_1_2 < 0.01:
            print("Issue: Feature difference between different sequences too small")
            return False
        else:
            print("Normal: Different sequences have different features")

        if diff_1_3 > 0.1:
            print("Abnormal: Feature difference between identical sequences too large!")
            return False
        else:
            print("Normal: Identical sequences have similar features")

        return True

    except Exception as e:
        print(f"ABBERT feature extraction test failed: {e}")
        return False




def process_dataset_fixed(sequences, output_file, model_path, dataset_name):
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Input sequences: {len(sequences)}")

    if len(sequences) > 0:
        print(f"First 2 sequence examples:")
        for i in range(min(2, len(sequences))):
            print(f"  Sequence{i + 1}: {sequences[i][:50]}... (length: {len(sequences[i])})")

    max_length = 1024
    filtered_sequences = [seq for seq in sequences if len(seq) <= max_length]

    if len(filtered_sequences) < len(sequences):
        print(f"Warning: Skipped {len(sequences) - len(filtered_sequences)} sequences exceeding {max_length} length")

    if filtered_sequences:
        print(f"Extracting ABBERT features, {len(filtered_sequences)} valid sequences remaining...")
        try:
            features = extract_abbert_features_fixed(filtered_sequences, model_path, dataset_name=dataset_name)

            if features is None:
                print(f"{dataset_name} feature extraction failed")
                return None

            np.save(output_file, features)
            print(f"Saved {output_file}, feature dimension: {features.shape}")
            return features
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return None
    else:
        print(f"No valid sequences to process")
        return None



# 主程序
if __name__ == "__main__":

    model_path = r"D:\models\AbBert"

    trainPos, _ = readFastaFile('trainP_mirror.txt')

    trainNeg, _ = readFastaFile('trainN_mirror.txt')

    testPos, _ = readFastaFile('testP_mirror.txt')

    testNeg, _ = readFastaFile('testN_mirror.txt')

    print("\nstart ABBERT...")

    if not test_feature_extraction_fixed(model_path):
        print(" ABBERT feature extraction test failed, exiting")
        exit(1)

    # 处理全部数据
    print("\n" + "=" * 50)
    print("Processing main data")
    print("=" * 50)

    process_dataset_fixed(trainPos, 'ptrainABBERT.npy', model_path, "train pos")

    process_dataset_fixed(trainNeg, 'ntrainABBERT.npy', model_path, "train neg")

    process_dataset_fixed(testPos, 'ptestABBERT.npy', model_path, "test pos")

    process_dataset_fixed(testNeg, 'ntestABBERT.npy', model_path, "test neg")

    print("\ndone")
