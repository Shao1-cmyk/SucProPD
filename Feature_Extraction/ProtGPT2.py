import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import os
from tqdm import tqdm


def readFastaFile(fastaFile):

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


def is_valid_sequence(seq):
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
    return all(char in valid_chars for char in seq.upper())


def load_local_protgpt2_model(model_path):

    print(f"load ProtGPT2: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"error path: {model_path}")

    # 检查必要文件
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            print(f"miss file {file}")

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        model = model.to('cpu')
        model.eval()

        print(f"ProtGPT2 load success,dimension: {model.config.hidden_size}")
        return tokenizer, model

    except Exception as e:
        print(f"unsucess load: {e}")
        raise


def extract_protgpt2_features_single(sequences, model_path):

    tokenizer, model = load_local_protgpt2_model(model_path)
    all_features = []

    for i, seq in enumerate(tqdm(sequences, desc=" extract ProtGPT2 feature")):
        try:

            inputs = tokenizer(
                seq,
                padding=False,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
                add_special_tokens=True
            )

            inputs = {k: v.to('cpu') for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )

                hidden_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]

                seq_len = hidden_states.shape[1]
                if seq_len > 1:

                    seq_rep = hidden_states[0, 1:].mean(dim=0)
                else:
                    seq_rep = hidden_states[0].mean(dim=0)

                all_features.append(seq_rep.numpy())

            if (i + 1) % 100 == 0:
                gc.collect()

        except Exception as e:
            print(f"error: {e}")

            all_features.append(np.zeros(model.config.hidden_size))
            continue

    del model, tokenizer
    gc.collect()

    features_array = np.array(all_features)
    print(f"done ,shape: {features_array.shape}")
    return features_array


def extract_protgpt2_features_batch(sequences, batch_size=1, model_path=None):

    tokenizer, model = load_local_protgpt2_model(model_path)
    all_features = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="batch feature extraction"):
        batch_sequences = sequences[i:i + batch_size]

        try:
            # Tokenize batch
            inputs = tokenizer(
                batch_sequences,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
                add_special_tokens=True
            )

            inputs = {k: v.to('cpu') for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )

                hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

                batch_features = []
                for j in range(len(batch_sequences)):
                    attention_mask = inputs['attention_mask'][j]
                    seq_len = attention_mask.sum().item()

                    if seq_len > 1:
                        seq_rep = hidden_states[j, 1:seq_len].mean(dim=0)
                    else:
                        seq_rep = hidden_states[j].mean(dim=0)

                    batch_features.append(seq_rep.numpy())

                all_features.extend(batch_features)

        except Exception as e:
            print(f"process batch {i // batch_size + 1} error: {e}")

            for _ in range(len(batch_sequences)):
                all_features.append(np.zeros(model.config.hidden_size))
            continue

        del inputs, outputs, hidden_states
        gc.collect()

    del model, tokenizer
    gc.collect()

    features_array = np.array(all_features)
    print(f"done，shape: {features_array.shape}")
    return features_array


def check_feature_quality(features, name):

    print(f"\n{name}quality check:")
    print(f"shape: {features.shape}")
    print(f"range: [{features.min():.6f}, {features.max():.6f}]")
    print(f"mean: {features.mean():.6f}")
    zero_ratio = (np.abs(features) < 1e-10).mean()
    print(f"proportion to 0 {zero_ratio:.4f}")

    if zero_ratio > 0.9:
        print("feature invalid")
    else:
        print("quality mormal")


def process_dataset(sequences, output_file, model_path, batch_size=1, use_single=True):

    print(f"processing,{len(sequences)}sequence...")

    max_length = 1024
    filtered_sequences = [seq for seq in sequences if len(seq) <= max_length]

    if filtered_sequences:

        try:
            if use_single:
                features = extract_protgpt2_features_single(filtered_sequences, model_path)
            else:
                features = extract_protgpt2_features_batch(filtered_sequences, batch_size, model_path)

            check_feature_quality(features, "ProtGPT2")

            np.save(output_file, features)
            print(f"saved {output_file}, shape: {features.shape}")
            return features

        except Exception as e:
            print(f"unsuccess: {e}")
            return None
    else:
        print(f"error")
        return None

if __name__ == "__main__":

    model_path = r"D:\models\protGPT2"

    trainPos, _ = readFastaFile('trainP_mirror.txt')

    trainNeg, _ = readFastaFile('trainN_mirror.txt')

    testPos, _ = readFastaFile('testP_mirror.txt')

    testNeg, _ = readFastaFile('testN_mirror.txt')

    print("\nstart ProtGPT2...")

    test_sequences = trainPos[:3]
    print("testing...")
    try:
        test_features = extract_protgpt2_features_single(test_sequences, model_path)
        check_feature_quality(test_features, "test")
        print("test success！")
    except Exception as e:
        print(f"error: {e}")
        exit(1)

    process_dataset(trainPos, 'ptrainProtGPT2.npy', model_path, use_single=True)

    process_dataset(trainNeg, 'ntrainProtGPT2.npy', model_path, use_single=True)

    process_dataset(testPos, 'ptestProtGPT2.npy', model_path, use_single=True)

    process_dataset(testNeg, 'ntestProtGPT2.npy', model_path, use_single=True)

    print("ProtGPT2 done")