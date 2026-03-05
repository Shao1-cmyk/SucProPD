def mirror_fill_sequence_jia_method(sequence, delta=10):

    if 'X' not in sequence:
        return sequence

    k_positions = [i for i, aa in enumerate(sequence) if aa == 'K']

    if not k_positions:

        return sequence.replace('X', 'A')

    seq_list = list(sequence)

    for k_pos in k_positions:

        for i in range(max(0, k_pos - delta), k_pos):
            if seq_list[i] == 'X':

                mirror_offset = k_pos - i
                mirror_index = k_pos + mirror_offset
                if mirror_index < len(seq_list):
                    seq_list[i] = seq_list[mirror_index]
                else:
                    seq_list[i] = 'A'

        for i in range(k_pos + 1, min(len(seq_list), k_pos + delta + 1)):
            if seq_list[i] == 'X':

                mirror_offset = i - k_pos
                mirror_index = k_pos - mirror_offset
                if mirror_index >= 0:
                    seq_list[i] = seq_list[mirror_index]
                else:
                    seq_list[i] = 'A'

    for i in range(len(seq_list)):
        if seq_list[i] == 'X':
            seq_list[i] = 'A'

    return ''.join(seq_list)


def process_file(input_file, output_file, delta=10):

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if line.startswith('>'):
                f_out.write(line + '\n')
            else:
                filled_seq = mirror_fill_sequence_jia_method(line, delta)
                f_out.write(filled_seq + '\n')


files_to_process = [
    ('trainPos.txt', 'trainP_mirror.txt'),
    ('trainNeg.txt', 'trainN_mirror.txt'),
    ('testPos.txt', 'testP_mirror.txt'),
    ('testNeg.txt', 'testN_mirror.txt')
]

if __name__ == "__main__":


    for input_file, output_file in files_to_process:
        print(f"process file: {input_file} -> {output_file}")
        process_file(input_file, output_file)


