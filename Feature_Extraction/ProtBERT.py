import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import gc
import os
import json
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


def fix_config_file(model_path):

    config_path = os.path.join(model_path, 'config.json')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if 'model_type' not in config:
            config['model_type'] = 'bert'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            print("recovered config.json file")
        else:
            print("config.json normal")

    except Exception as e:
        print(f"error: {e}")


def load_local_protbert_model(model_path):

    print(f"load ProtBERT model: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"error {model_path}")

    required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt']
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"miss file: {file}")
        print(f"find: {file}")

    fix_config_file(model_path)

    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                do_lower_case=False
            )
            model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.float32
            )
        except:

            tokenizer = BertTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                do_lower_case=False
            )
            model = BertModel.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.float32
            )

        model = model.to('cpu')
        model.eval()

        print(f"ProtBERT load success,shape: {model.config.hidden_size}")
        return tokenizer, model

    except Exception as e:
        print(f"error {e}")
        return load_protbert_manually(model_path)


def load_protbert_manually(model_path):

    try:
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            config = json.load(f)

        from transformers import BertConfig
        bert_config = BertConfig(
            vocab_size=config.get('vocab_size', 30),
            hidden_size=config.get('hidden_size', 1024),
            num_hidden_layers=config.get('num_hidden_layers', 30),
            num_attention_heads=config.get('num_attention_heads', 16),
            intermediate_size=config.get('intermediate_size', 4096),
            hidden_act=config.get('hidden_act', 'gelu'),
            hidden_dropout_prob=config.get('hidden_dropout_prob', 0.1),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob', 0.1),
            max_position_embeddings=config.get('max_position_embeddings', 512),
            type_vocab_size=config.get('type_vocab_size', 2),
            initializer_range=config.get('initializer_range', 0.02),
            layer_norm_eps=config.get('layer_norm_eps', 1e-12),
            pad_token_id=config.get('pad_token_id', 0),
        )

        tokenizer = BertTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            do_lower_case=False
        )

        model = BertModel(bert_config)
        state_dict = torch.load(
            os.path.join(model_path, 'pytorch_model.bin'),
            map_location='cpu',
            weights_only=True
        )
        model.load_state_dict(state_dict)

        model = model.to('cpu')
        model.eval()

        print("success")
        return tokenizer, model

    except Exception as e:
        print(f"unsuccess: {e}")
        raise


def correct_sequence_format(sequences):

    corrected_sequences = []
    for seq in sequences:
        seq = ''.join([aa for aa in seq.upper() if aa in 'ACDEFGHIKLMNPQRSTVWY'])
        seq_with_spaces = ' '.join(seq)
        corrected_sequences.append(seq_with_spaces)
    return corrected_sequences


def extract_protbert_features_corrected(sequences, model_path, batch_size=4):

    tokenizer, model = load_local_protbert_model(model_path)
    corrected_sequences = correct_sequence_format(sequences)
    all_features = []

    total_batches = (len(corrected_sequences) + batch_size - 1) // batch_size
    with tqdm(total=total_batches, desc="batch process") as pbar:

        for i in range(0, len(corrected_sequences), batch_size):
            batch_seqs = corrected_sequences[i:i + batch_size]
            batch_original = sequences[i:i + batch_size]

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
                    print(f"\nfirse batch tokenization test:")
                    for j in range(min(2, len(batch_seqs))):
                        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
                        valid_tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]
                        print(f"sequence {j + 1}: {len(valid_tokens)}token")

                inputs = {k: v.to('cpu') for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    hidden_states = outputs.last_hidden_state

                    for j in range(len(batch_seqs)):
                        attention_mask = inputs['attention_mask'][j].bool()
                        special_mask = (inputs['input_ids'][j] != tokenizer.cls_token_id) & \
                                       (inputs['input_ids'][j] != tokenizer.sep_token_id)
                        valid_mask = attention_mask & special_mask

                        if valid_mask.sum() > 0:
                            seq_rep = hidden_states[j, valid_mask].mean(dim=0)
                        else:
                            seq_rep = hidden_states[j, 0]

                        all_features.append(seq_rep.numpy())

            except Exception as e:
                print(f"process batch {i // batch_size + 1} error: {e}")
                for j in range(len(batch_seqs)):
                    all_features.append(np.zeros(model.config.hidden_size))

            pbar.update(1)

            if (i // batch_size) % 10 == 0:
                gc.collect()

    del model, tokenizer
    gc.collect()

    features_array = np.array(all_features)
    print(f"success,shape: {features_array.shape}")
    return features_array
def check_feature_quality(features, name):

    print(f"\n{name}feature quality check:")
    print(f"shape: {features.shape}")
    print(f"range: [{features.min():.6f}, {features.max():.6f}]")
    print(f"mean: {features.mean():.6f}")

    zero_ratio = (np.abs(features) < 1e-10).mean()
    print(f":proportion to 0 {zero_ratio:.4f}")

    if zero_ratio > 0.9:
        print("feature invalid")
    else:
        print("quality normal")


def process_dataset(sequences, output_file, model_path):

    print(f"processing，共{len(sequences)}sequence...")

    max_length = 1024
    filtered_sequences = [seq for seq in sequences if len(seq) <= max_length]


    if filtered_sequences:

        try:
            features = extract_protbert_features_corrected(filtered_sequences, model_path)
            check_feature_quality(features, "ProtBERT")
            np.save(output_file, features)
            print(f"saved {output_file},shape: {features.shape}")
            return features
        except Exception as e:
            print(f"error: {e}")
            return None
    else:
        print(f"error")
        return None


def test_feature_extraction(model_path):

    test_sequences = [
        "ACDEFGHIK",
        "ACDEFGHIKLMNPQRSTVWY",
        "MKTVRQERLKSIVRILERSK"
    ]

    print("testing...")
    try:
        features = extract_protbert_features_corrected(test_sequences, model_path)
        check_feature_quality(features, "test")

        if len(features) >= 2:
            diff = np.abs(features[0] - features[1]).mean()
            print(f"mean: {diff:.6f}")

            if diff < 1e-6:
                print("invalid")
            else:
                print("feature normal")

        return True
    except Exception as e:
        print(f"error: {e}")
        return False

if __name__ == "__main__":

    model_path = r"D:\models\ProtBERT"

    trainPos, _ = readFastaFile('trainP_mirror.txt')

    trainNeg, _ = readFastaFile('trainN_mirror.txt')

    testPos, _ = readFastaFile('testP_mirror.txt')

    testNeg, _ = readFastaFile('testN_mirror.txt')

    print("\nstart ProtBERT...")

    if not test_feature_extraction(model_path):
        print("error")
        exit(1)

    process_dataset(trainPos, 'ptrainProtBERT_2.npy', model_path)

    process_dataset(trainNeg, 'ntrainProtBERT_2.npy', model_path)

    process_dataset(testPos, 'ptestProtBERT_2.npy', model_path)

    process_dataset(testNeg, 'ntestProtBERT_2.npy', model_path)

    print("ProtBERT done")