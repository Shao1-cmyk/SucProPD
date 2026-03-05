import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel
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


def load_local_prott5_model(model_path):

    print(f"loading ProtT5 model: {model_path}")

    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"path not exist: {model_path}")

    # 检查必要文件
    required_files = ['config.json', 'pytorch_model.bin', 'spiece.model']
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"missing file: {file}")
        print(f"fine file: {file}")

    try:
        # load tokenizer -  use SentencePiece
        tokenizer = T5Tokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            do_lower_case=False
        )

        model = T5EncoderModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32
        )

        model = model.to('cpu')
        model.eval()

        print(f"ProtT5 model load success,dimensionality: {model.config.hidden_size}")
        return tokenizer, model

    except Exception as e:
        print(f"error: {e}")
        raise


def extract_prott5_features_single(sequences, model_path):

    tokenizer, model = load_local_prott5_model(model_path)
    all_features = []

    for i, seq in enumerate(tqdm(sequences, desc="extract ProtT5 feature")):
        try:

            sequence_processed = " ".join(seq)

            inputs = tokenizer(
                sequence_processed,
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
                    return_dict=True
                )

                hidden_states = outputs.last_hidden_state

                seq_len = hidden_states.shape[1]
                if seq_len > 2:

                    seq_rep = hidden_states[0, 1:seq_len - 1].mean(dim=0)
                else:
                    seq_rep = hidden_states[0].mean(dim=0)

                all_features.append(seq_rep.numpy())

            if (i + 1) % 50 == 0:
                gc.collect()

        except Exception as e:
            print(f"sequence {i} error: {e}")

            all_features.append(np.zeros(model.config.hidden_size))
            continue

    # clean
    del model, tokenizer
    gc.collect()

    features_array = np.array(all_features)
    print(f"ProtT5 success,shape: {features_array.shape}")
    return features_array


def check_feature_quality(features, name):


    print(f"shape: {features.shape}")
    print(f"range: [{features.min():.6f}, {features.max():.6f}]")
    print(f"mean: {features.mean():.6f}")
    print(f"standard deviation: {features.std():.6f}")

    zero_ratio = (np.abs(features) < 1e-10).mean()
    print(f"proportion close to 0: {zero_ratio:.4f}")

    if zero_ratio > 0.9:
        print("invalid feature")
    else:
        print("success")


def process_dataset(sequences, output_file, model_path):

    print(f"processing,{len(sequences)}sequence...")


    max_length = 1024
    filtered_sequences = [seq for seq in sequences if len(seq) <= max_length]

    if len(filtered_sequences) < len(sequences):
        print(f" skip{len(sequences) - len(filtered_sequences)} sequences whose length exceed{max_length}")

    if filtered_sequences:
        print(f"extracting ProtT5 feature，remain{len(filtered_sequences)}sequences...")
        try:
            features = extract_prott5_features_single(filtered_sequences, model_path)
            check_feature_quality(features, "ProtT5")
            np.save(output_file, features)
            print(f"saved to {output_file}, dimensionality: {features.shape}")
            return features
        except Exception as e:
            print(f"error: {e}")
            return None
    else:
        print(f"no sequences")
        return None


# 主程序
if __name__ == "__main__":
    # 设置模型路径
    model_path = r"D:\models\ProtT5"


    trainPos, _ = readFastaFile('trainP_mirror.txt')

    trainNeg, _ = readFastaFile('trainN_mirror.txt')

    testPos, _ = readFastaFile('testP_mirror.txt')

    testNeg, _ = readFastaFile('testN_mirror.txt')

    print("\nstart to extract ProtT5 feature...")

    # test
    test_sequences = trainPos[:2]

    try:
        test_features = extract_prott5_features_single(test_sequences, model_path)
        check_feature_quality(test_features, "测试")
        print("test success")
    except Exception as e:
        print(f"error: {e}")
        exit(1)


    process_dataset(trainPos, 'ptrainProtT5.npy', model_path)

    process_dataset(trainNeg, 'ntrainProtT5.npy', model_path)

    process_dataset(testPos, 'ptestProtT5.npy', model_path)

    process_dataset(testNeg, 'ntestProtT5.npy', model_path)

    print("ProtT5 done!")