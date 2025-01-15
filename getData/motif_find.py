import numpy as np
from tqdm import tqdm
import random

x_or_y = "x"
cell_line = 'K562'


def motif_compare(seq, motif, threshold):
    num = len(seq)
    seq = seq.upper()
    motif_length = motif.shape[0]
    cnt = 0
    len_start = []
    # len_start.append(motif_length)
    for i in range(num - motif_length + 1):
        score = 0
        seq_list = seq[i:i + motif_length]
        for j in range(len(seq_list)):
            if seq_list[j] == 'A':
                score += motif[j][0]
            elif seq_list[j] == 'C':
                score += motif[j][1]
            elif seq_list[j] == 'G':
                score += motif[j][2]
            elif seq_list[j] == 'T':
                score += motif[j][3]
        if score >= threshold:
            cnt = cnt + 1
            len_start.append(i)
    return cnt, np.array(len_start), motif_length


def generate_random_sequence(length):
    return ''.join(random.choices('ATGC', k=length))


def replace_subsequences(dna_sequence, index_list, sub_length):
    dna_list = list(dna_sequence)
    for index in index_list:
        random_sequence = generate_random_sequence(sub_length)
        dna_list[index:index+sub_length] = random_sequence
    return ''.join(dna_list)


fmotif = open("../motif/" + "HOCOMOCOv11_core_pwms_HUMAN_mono.txt", 'r')
motifs = {}
for line in fmotif.readlines():
    if line[0] != ' ':
        if line[0] == '>':
            key = line.strip('>').strip('\n')
            a = []
        if line[0] != '>':
            a.append(list(line.upper().strip('\n').split("\t")))
            motifs[key] = a

for key in motifs.keys():
    motifs[key] = np.array(motifs[key], dtype="float64")


fthre = open("../motif/" + "HOCOMOCOv11_core_HUMAN_mono_homer_format_0.0001.txt", 'r')
thresholds = {}
key_val = []
for line in fthre.readlines():
    if line[0] != ' ':
        if line[0] == '>':
            key_val = list(line.strip('\n').split("\t"))
            key = key_val[1]
            thresholds[key] = key_val[2]
for key in thresholds.keys():
    thresholds[key] = np.array(thresholds[key], dtype="float64")


input_file = '../data/' + cell_line + '/' + x_or_y + '.fasta'
fseq = open(input_file, 'r')
sequences = []
counts = []
all_cnt = []
count = 0
for line in fseq.readlines():
    if line[0] != ' ':
        if line[0] != '>':
            sequences.append(line.upper().strip('\n'))


for index, key in enumerate(motifs.keys()):
    # for number in range(len(sequences)):
    for number in tqdm(range(len(sequences)), desc="Processing data", unit="item"):
        len_starts = []
        sequence = sequences[number]
        motif = motifs[key]
        threshold = thresholds[key]
        count, len_start, motif_length = motif_compare(sequence, motif, threshold)
        new_sequence = replace_subsequences(sequence, len_start, motif_length)
        sequences[number] = new_sequence
        # len_starts.append(len_start)
        counts.append(count)
    with open('../change_motif_data/' + cell_line + '/' + key + '_' + x_or_y + '.fasta', "w") as file:
        for line in sequences:
            file.write(line + "\n")
    all_cnt.append(counts)
    counts = []
all_cnt = np.array(all_cnt)
np.save('../motif/' + cell_line + '_' + x_or_y + '.npy', all_cnt)