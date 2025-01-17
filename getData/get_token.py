import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from collections import Counter, defaultdict
import os
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
rand_mat = lambda n, m: np.random.rand(n, m)
centralize = lambda mat: mat - mat.mean(axis=1).reshape(-1, 1)
get_lengths = lambda mat: np.sqrt((mat**2).sum(axis=1))


def remove_elements_from_array(arr, target_element, count_to_remove):
    count_removed = 0
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] == target_element:
            arr.pop(i)
            count_removed += 1
            if count_removed == count_to_remove:
                break
    return np.array(arr)


def filter_top_num_frequent_elements(arr, top_num):
    counts = Counter(arr)
    sorted_elements_by_freq = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    retained_elements = set()
    cumulative_count = 0
    need_delete_num = 0
    nedd_delete_element = 0

    for elem, count in sorted_elements_by_freq:
        if cumulative_count + count <= top_num:
            retained_elements.add(elem)
            cumulative_count += count
        else:
            retained_elements.add(elem)
            cumulative_count += count
            need_delete_num = cumulative_count - top_num
            nedd_delete_element = elem
            break
    filtered_arr = [elem for elem in arr if elem in retained_elements]
    return_arr = remove_elements_from_array(filtered_arr, nedd_delete_element, need_delete_num)
    return return_arr


def get_top_indexes(arr, top_n):
    indexed_arr = [(value, index) for index, value in enumerate(arr)]
    indexed_arr.sort(reverse=True)
    top_indexes = [index for value, index in indexed_arr[:top_n]]
    new_arr = [arr[index] for index in sorted(top_indexes)]
    return new_arr


def mutate_dna_sequence(sequence, mutation_rate=0.05):
    bases = ['A', 'T', 'C', 'G']
    new_sequence = []
    for base in sequence:
        if random.random() < mutation_rate:
            new_base = random.choice([b for b in bases if b != base])
            new_sequence.append(new_base)
        else:
            new_sequence.append(base)
    return ''.join(new_sequence)


def only_get_tokenizer_id(filename, tokenizer):
    sequence_forward = []
    f = open(filename, 'r')
    for line in f.readlines():
        if line[0] != ' ':
            if line[0] != '>':
                sequence_forward.append(line.upper().strip('\n'))
    f.close()
    length = len((sequence_forward))
    seq_id_list = []
    for number in range(length):
        print('forward ' + str(number) + '/' + str(length))
        inputs_id = tokenizer(sequence_forward[number], return_tensors='pt')["input_ids"]
        input_array = np.squeeze(inputs_id.numpy())
        top = 500
        # filtered_array = filter_top_num_frequent_elements(input_array, top)
        filtered_array = get_top_indexes(input_array, top)
        seq_id_list.append(filtered_array)
    seq_array = np.array(seq_id_list)

    return seq_array


def only_get_tokenizer_id_r(filename, tokenizer):
    sequence_forward = []
    f = open(filename, 'r')
    for line in f.readlines():
        if line[0] != ' ':
            if line[0] != '>':
                sequence_forward.append(line.strip().upper()[::-1])
    f.close()
    length = len((sequence_forward))
    seq_id_list = []
    for number in range(length):
        print('forward ' + str(number) + '/' + str(length))
        inputs_id = tokenizer(sequence_forward[number], return_tensors='pt')["input_ids"]
        input_array = np.squeeze(inputs_id.numpy())
        top = 500
        # filtered_array = filter_top_num_frequent_elements(input_array, top)
        filtered_array = get_top_indexes(input_array, top)
        seq_id_list.append(filtered_array)
    seq_array = np.array(seq_id_list)
    return seq_array


def compute():
    path = '../DNABERT-S'
    tokenizer = AutoTokenizer.from_pretrained(path)
    cell_lines = ['K562', 'GM12878', 'IMR90']

    for cell_line in cell_lines:
        # for set in ['train', 'test']:
        x_filename = '../data/' + cell_line + '/x' + '.fasta'
        y_filename = '../data/' + cell_line + '/y' + '.fasta'
        # x_filename = '../train_test/' + cell_line + '/x_test.fasta'
        x_feature_f = only_get_tokenizer_id_r(x_filename, tokenizer)
        # np.save('../token/' + cell_line + '/xf_tokens.npy', x_feature_f)
        np.save('../token/' + cell_line + '/xzr_tokensid500' + '.npy', x_feature_f)
        x_feature_f = only_get_tokenizer_id_r(y_filename, tokenizer)
        # np.save('../token/' + cell_line + '/xf_tokens.npy', x_feature_f)
        np.save('../token/' + cell_line + '/yzr_tokensid500' + '.npy', x_feature_f)


if __name__ == '__main__':
    compute()
