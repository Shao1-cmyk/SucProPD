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


def load_local_progen2_model(model_path):

    print(f"loading model: {model_path}")

    required_files = ['config.json', 'model.safetensors', 'tokenizer.json']
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"miss file: {file}")
        print(f"fine file: {file}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32
        ).to("cpu")

        model.eval()
        print("load success")
        return tokenizer, model

    except Exception as e:
        print(f"load unsuccess {e}")
        raise

def extract_progen2_features_batch(sequences, batch_size=1, model_path="./progen2-base"):

    try:
        tokenizer, model = load_local_progen2_model(model_path)
    except Exception as e:
        print(f"error {e}")

        raise

    all_features = []
    total_sequences = 0

    for i in tqdm(range(0, len(sequences), batch_size), desc="CPU "):
        batch_sequences = sequences[i:i + batch_size]
        total_sequences += len(batch_sequences)

        print(
            f"batch process {i // batch_size + 1}/{(len(sequences) - 1) // batch_size + 1}, sequences {total_sequences}/{len(sequences)}")

        try:

            inputs = tokenizer(
                batch_sequences,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )

            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )

                hidden_states = outputs.hidden_states[-1]

                batch_features = []
                for j in range(len(batch_sequences)):
                    attention_mask = inputs['attention_mask'][j]
                    non_padding_tokens = attention_mask.nonzero(as_tuple=True)[0]

                    if len(non_padding_tokens) > 0:
                        seq_rep = hidden_states[j, non_padding_tokens].mean(dim=0)
                    else:
                        seq_rep = hidden_states[j].mean(dim=0)

                    batch_features.append(seq_rep.cpu().numpy())

                all_features.extend(batch_features)

                feature_mean = np.mean(batch_features[0])
                if abs(feature_mean) < 1e-10:
                    print(f"batch {i // batch_size + 1} error（mean: {feature_mean}）")

        except Exception as e:
            print(f"batch {i // batch_size + 1} error: {e}")
            raise

        del inputs, outputs, hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    features_array = np.array(all_features)
    print(f"done，shape: {features_array.shape}")

    check_feature_quality(features_array, "final")

    return features_array


def check_feature_quality(features, name):

    print(f"\n{name}feature quality check:")
    print(f"shape: {features.shape}")
    print(f"range: [{features.min():.6f}, {features.max():.6f}]")
    print(f"mean: {features.mean():.6f}")

    zero_ratio = (np.abs(features) < 1e-10).mean()
    print(f"A proportion close to 0: {zero_ratio:.4f}")

    if zero_ratio > 0.9:
        print("feature invalid")
    else:
        print("quality normal")


def process_dataset(sequences, output_file, batch_size=4, model_path="./progen2-base"):

    print(f"processing，{len(sequences)}sequences...")

    valid_sequences = []
    for i, seq in enumerate(sequences):
        valid_sequences.append(seq)

    max_length = 1024
    filtered_sequences = [seq for seq in valid_sequences if len(seq) <= max_length]

    if filtered_sequences:

        try:
            features = extract_progen2_features_batch(filtered_sequences, batch_size=batch_size, model_path=model_path)
            np.save(output_file, features)
            print(f"saved {output_file}, dimension: {features.shape}")
            return features
        except Exception as e:
            print(f" {e}")
            return None
    else:
        print(f"no sequence")
        return None


# 主程序
if __name__ == "__main__":

    model_path = r"D:\models\progen2"

    batch_size = 2

    trainPos, _ = readFastaFile('trainP_mirror.txt')

    trainNeg, _ = readFastaFile('trainN_mirror.txt')

    testPos, _ = readFastaFile('testP_mirror.txt')

    testNeg, _ = readFastaFile('testN_mirror.txt')

    print("\nstart ProGen2-base...")

    test_sequences = trainPos[:10]
    print("testing...")
    try:
        test_features = extract_progen2_features_batch(test_sequences, batch_size=2, model_path=model_path)
        print("test success")
    except Exception as e:
        print(f"test unsuccess: {e}")
        exit(1)

    # 如果测试成功，处理全部数据
    process_dataset(trainPos, 'ptrainProGen2.npy', batch_size, model_path)
    process_dataset(trainNeg, 'ntrainProGen2.npy', batch_size, model_path)
    process_dataset(testPos, 'ptestProGen2.npy', batch_size, model_path)
    process_dataset(testNeg, 'ntestProGen2.npy', batch_size, model_path)

    print("Done")