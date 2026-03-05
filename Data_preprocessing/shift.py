def split_samples(input_file, pos_output, neg_output):
    with open(input_file, 'r') as infile, \
            open(pos_output, 'w') as pos_file, \
            open(neg_output, 'w') as neg_file:

        lines = infile.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('>'):
                label = line[1:].strip()
                seq = lines[i + 1].strip() if i + 1 < len(lines) else ''
                if label == '1':
                    pos_file.write(seq + '\n')
                elif label == '0':
                    neg_file.write(seq + '\n')
                i += 2
            else:
                i += 1


# train.fast
split_samples('train.fasta', 'trainPos.txt', 'trainNeg.txt')

# test.fast
split_samples('test.fasta', 'testPos.txt', 'testNeg.txt')