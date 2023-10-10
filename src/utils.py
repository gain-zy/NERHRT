# coding: utf-8
import json
import copy
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from parameter import device


PAD_token = 0
PAD, CLS = '[PAD]', '[CLS]'

'''
data utils
'''
def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file


def compute_decimal(input_num):
    target_list = list()
    if "." not in str(input_num):
        for index, every_char in enumerate(input_num):
            target_list.append((every_char, len(input_num) - 1 - index))
    else:
        split_str = str(input_num).split(".")
        reverse_over_zero = split_str[0][::-1]
        for index, every_char in enumerate(reverse_over_zero):
            target_list.append((every_char, index))
        lower_zero = split_str[1]
        for index, every_char in enumerate(lower_zero):
            target_list.append((every_char, 0 - index - 1))
    digits = []
    positions = []
    for t in target_list:
        digits.append(int(t[0]))
        # 因为我们考虑的(-9,9)一共19个数字，从0开始下标
        positions.append(t[1]+9)
    return target_list,digits,positions


# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print('-' * 40, "Reading Row Datas", '-' * 40)
    f = open(filename)
    js = ""
    data = []
    for i, s in tqdm.tqdm(enumerate(f)):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            idx, ori_sent, question, equation, answer = data_d['id'], data_d['original_text'], data_d[
                'segmented_text'].strip(), data_d['equation'], data_d['ans']
            if "千米/小时" in equation:
                '去掉id:10431的错误'
                equation = equation.replace('千米/小时', '')
            equation = equation[2:]

            # 去除19573的数据问题
            if idx == '19573':
                ori_sent = ori_sent[:-3]
                question = question[:-6]
            # 23088
            if idx == '23088' or idx == '10619':
                ori_sent = ori_sent[:-4]
                question = question[:-8]
            js = ""
            # 丢掉 不合格的数据
            if idx == '18737':
                continue
            data.append((idx, ori_sent, question, equation, answer))
    print('-' * 40, "      Done  ", '-' * 40)
    return data


def get_train_test_fold(ori_path,prefix,data,pairs,group):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []
    # 这里
    for item,pair,g in zip(data, pairs, group):
        pair = list(pair)
        pair.append(g['group_num'])
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res


def change_num(num):
    new_num = []
    for item in num:
        if '/' in item:
            new_str = item.split(')')[0]
            new_str = new_str.split('(')[1]
            a = float(new_str.split('/')[0])
            b = float(new_str.split('/')[1])
            value = a/b
            new_num.append(value)
        elif '%' in item:
            value = float(item[0:-1])/100
            new_num.append(value)
        else:
            new_num.append(float(item))
    return new_num


def get_adjacency_matrices_num(batch_nums, batch_num_pos, max_n_nodes):
    '''
    :param batch_nums: [['5', '1', '50', '21'], ...]
    :param batch_num_pos: [[0, 2, 7, 13], ...]
    :param max_n_nodes: 4
    :param device:
    :return:
    '''
    batch_size = len(batch_nums)
    num_graph = torch.zeros((batch_size, max_n_nodes, max_n_nodes))
    attention_mask = torch.zeros((batch_size, max_n_nodes, max_n_nodes))
    attention_mask += -1e9
    for b, nums in enumerate(batch_nums):
        num_list = change_num(nums)
        length_num = len(num_list)
        attention_mask[b, :length_num, :length_num] = 0
        for i in range(length_num):
            for j in range(length_num):
                if float(num_list[i]) > float(num_list[j]):
                    num_graph[b, i, j] = 1
                else:
                    num_graph[b, j, i] = 1
    return num_graph, attention_mask


def get_adjacency_matrices_token(batch_lengths, batch_num_pos, max_len, group_batches,
                                 parse_tree_batches, seg_batches):
    '''
    '''
    # print('batch_lengths: ', batch_lengths)
    # print('parse_tree_batches: ', parse_tree_batches)
    # print('seg_batches: ', seg_batches)
    # print('batch_num_pos: ', batch_num_pos)
    # print('group_batches: ', group_batches)
    batch_size = len(batch_lengths)
    token_graph = torch.zeros((batch_size, max_len, max_len))
    attention_mask = torch.zeros((batch_size, max_len, max_len))
    attention_mask += -1e9

    for b, length in enumerate(batch_lengths):
        attention_mask[b, :length, :length] = 0
        for i in range(length):
            token_graph[b, i, i] = 1
        for i in batch_num_pos[b]:
            for j in group_batches[b]:
                if i < max_len and j < max_len and j not in batch_num_pos[b] and abs(i - j) < 4:
                    token_graph[b, i, j] = 1
                    token_graph[b, j, i] = 1
        for i in group_batches[b]:
            for j in group_batches[b]:
                if i < max_len and j < max_len:
                    if seg_batches[b][i] == seg_batches[b][j]:
                        token_graph[b, i, j] = 1
                        token_graph[b, j, i] = 1
        for p in range(len(parse_tree_batches[b])):
            if parse_tree_batches[b][p] != -1:
                token_graph[b, parse_tree_batches[b][p], p] = 1
                token_graph[b, p, parse_tree_batches[b][p]] = 1
    return token_graph, attention_mask


# attribute between graph
def get_attribute_between_graph(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list, k_hop=1, contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    #quantity_cell_list = quantity_cell_list.extend(id_num_list)
    if not contain_zh_flag:
        return graph
    for i in id_num_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                graph[i][j] = 1
                graph[j][i] = 1
    for i in quantity_cell_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len:
                if input_batch[i] == input_batch[j]:
                    graph[i][j] = 1
                    graph[j][i] = 1
    if k_hop == 1:
        return graph


def Q_set(max_len, id_num_list, quantity_cell_list, htop=1):
    temp = -1
    group_dict = {}
    for i in id_num_list:
        temp_list = []
        for j in quantity_cell_list:
            if i < max_len and j < max_len and j not in id_num_list and temp != j and abs(i-j) < 4:
                temp_list.append(j)
            temp = j
        group_dict[i] = temp_list
    return group_dict


def get_two_hop_graph(input_batch, max_len, sentence_length, id_num_list, quantity_cell_list, parse_tree, k_hop=2, contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    # quantity_cell_list = quantity_cell_list.extend(id_num_list)
    if not contain_zh_flag:
        return graph

    quantity2word = Q_set(max_len, id_num_list, quantity_cell_list)
    for q in quantity2word:
        for w in quantity2word[q]:
            graph[q][w] = 1
            graph[w][q] = 1
    '''
    for i in id_num_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                graph[i][j] = 1
                graph[j][i] = 1
    '''
    for i in quantity_cell_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len:
                if input_batch[i] == input_batch[j]:
                    graph[i][j] = 1
                    graph[j][i] = 1

    if k_hop == 1:
        return graph
    else:
        # 构建了基础的相邻。
        for q in quantity2word:
            for w in quantity2word[q]:
                for i, p in enumerate(parse_tree):
                    # 如果下标单词是w， 对应的不是根，对应的不允许是数字 , p != q
                    if i == w and p!=-1 and p not in id_num_list:
                        graph[q][p] = 1
                        graph[p][q] = 1
        return graph


def get_number_word_graph(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list, parse_tree, k_hop=1, contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    # quantity_cell_list = quantity_cell_list.extend(id_num_list)
    if not contain_zh_flag:
        return graph
    for i in id_num_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len and j not in id_num_list and abs(i - j) < 4:
                graph[i][j] = 1
                graph[j][i] = 1
    for i in quantity_cell_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len:
                if input_batch[i] == input_batch[j]:
                    graph[i][j] = 1
                    graph[j][i] = 1


def get_greater_num_graph(max_len, sentence_length, num_list, id_num_list, k_hop=1, contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    num_list = change_num(num_list)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    if not contain_zh_flag:
        return graph
    for i in range(len(id_num_list)):
        for j in range(len(id_num_list)):
            if float(num_list[i]) > float(num_list[j]):
                graph[id_num_list[i]][id_num_list[j]] = 1
            else:
                graph[id_num_list[j]][id_num_list[i]] = 1
    if k_hop == 1:
        return graph


def get_comparison_graph(max_len, num_list, id_num_list, k_hop=1, contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    num_list = change_num(num_list)

    graph = np.diag(diag_ele)
    if not contain_zh_flag:
        return graph
    for i in range(len(id_num_list)):
        for j in range(len(id_num_list)):
            if float(num_list[i]) > float(num_list[j]):
                graph[id_num_list[i]][id_num_list[j]] = 1
            else:
                graph[id_num_list[j]][id_num_list[i]] = 1
    if k_hop == 1:
        return graph


def get_parse_graph_batch(max_len, input_length, parse_tree, k_hop=1):
    '''
    #  版本是graph的版本,也就是上面的生成方法
    # batch_size个：input长度， input中[每个词]和[下标的词]关系
    diag_ele = [1] * input_length + [0] * (max_len - input_length)
    # 区别
    graph = np.diag(diag_ele) + np.diag(diag_ele[1:], 1) + np.diag(diag_ele[1:], -1)
    for j in range(len(parse_tree)):
        if parse_tree[j] != -1:
            graph[parse_tree[j], j] = 1
    return graph
    '''
    '''
    # 版本是multi_decoder的版本
    '''
    # batch_size个：input长度， input中[每个词]和[下标的词]关系
    diag_ele = [1] * input_length + [0] * (max_len - input_length)
    # 区别,相邻的也是1.
    graph = np.diag([1]*max_len) + np.diag(diag_ele[1:], 1) + np.diag(diag_ele[1:], -1)
    # 双向的
    for j in range(len(parse_tree)):
        if parse_tree[j] != -1:
            graph[parse_tree[j], j] = 1
            graph[j, parse_tree[j]] = 1
    if k_hop == 1:
        return graph


# attribute between graph
def get_HG_attribute_between_graph(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list,contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    #for i in range(sentence_length):
    #    diag_ele[i] = 1
    graph = np.diag(diag_ele)
    #quantity_cell_list = quantity_cell_list.extend(id_num_list)
    if not contain_zh_flag:
        return graph
    for i in id_num_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                graph[i][j] = 1
                graph[j][i] = 1
    for i in quantity_cell_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len:
                if input_batch[i] == input_batch[j]:
                    graph[i][j] = 1
                    graph[j][i] = 1
    return graph


def get_HG_greater_num_graph(max_len, sentence_length, num_list, id_num_list,contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    num_list = change_num(num_list)
    #for i in range(sentence_length):
    #    diag_ele[i] = 1
    graph = np.diag(diag_ele)
    if not contain_zh_flag:
        return graph
    for i in range(len(id_num_list)):
        for j in range(len(id_num_list)):
            if float(num_list[i]) > float(num_list[j]):
                graph[id_num_list[i]][id_num_list[j]] = 1
            else:
                graph[id_num_list[j]][id_num_list[i]] = 1
    return graph


def get_HG_parse_graph_batch(max_len, input_length, parse_tree):
    '''
    #  版本是graph的版本,也就是上面的生成方法
    # batch_size个：input长度， input中[每个词]和[下标的词]关系
    diag_ele = [1] * input_length + [0] * (max_len - input_length)
    # 区别
    graph = np.diag(diag_ele) + np.diag(diag_ele[1:], 1) + np.diag(diag_ele[1:], -1)
    for j in range(len(parse_tree)):
        if parse_tree[j] != -1:
            graph[parse_tree[j], j] = 1
    return graph
    '''
    '''
    # 版本是multi_decoder的版本
    '''
    # batch_size个：input长度， input中[每个词]和[下标的词]关系
    diag_ele = np.zeros(max_len)
    # 区别
    graph = np.diag([1]*max_len) + np.diag(diag_ele[1:], 1) + np.diag(diag_ele[1:], -1)
    for j in range(len(parse_tree)):
        if parse_tree[j] != -1:
            graph[parse_tree[j], j] = 1
    return graph


'''
model utils
'''
# Graph Module
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def gen_single_group_num(input_seq, max_seq, num_pos):
    # max_seq : 这个 seq的 最大长度
    punctuation = [",", "：", "；", "？", "！", "，", "“", "”", ",", ".", "?", "，", "。", "？", "．", "；", "｡"]
    group = []
    # seq_list=pair[0]
    # id_=pair[4]
    # num_pos=pair[3]
    for num_id in num_pos:
        if input_seq[num_id] == "NUM":
            if num_id - 1 >= 0 and input_seq[num_id - 1] not in punctuation:
                group.append(num_id - 1)
            group.append(num_id)
            if num_id + 1 < max_seq and input_seq[num_id + 1] not in punctuation:
                group.append(num_id + 1)

    last_p_pos = 0
    for id_ in range(0, max_seq - 2):
        if input_seq[id_] in punctuation:
            if id_ > last_p_pos:
                last_p_pos = id_
    keyword_list = ["多", "少", "多少", "How", "how", "what", "What"]
    for num_id in range(last_p_pos + 1, max_seq):
        if input_seq[num_id] in keyword_list:
            if num_id - 1 >= 0 and input_seq[num_id - 1] not in punctuation:
                group.append(num_id - 1)
            group.append(num_id)
            if num_id + 1 < max_seq and input_seq[num_id + 1] not in punctuation:
                group.append(num_id + 1)
    return group


# S x B x H; B x num_seq; B; max_num_len; H
# num_size：该batch最大不等式数字长度
def get_all_question_encoder_outputs(encoder_outputs, question_pos, batch_size, question_max_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0) # S
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        question_start = question_pos[b][0]
        question_end = question_pos[b][-1]
        for i in range(question_start, question_end):
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(question_end - question_start, question_max_size)]
        masked_index += [temp_1 for _ in range(question_end - question_start, question_max_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, question_max_size, hidden_size)
    if device:
        indices = indices.to(device)
        masked_index = masked_index.to(device)
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))
    all_question = all_embedding.index_select(0, indices)
    all_question = all_question.view(batch_size, question_max_size, hidden_size)
    return all_question.masked_fill(masked_index.bool(), 0.0)


def gen_word_char_mat(ori_seg, tokenizer):
    length = 1
    word_list = []
    chars = []
    for i, w in enumerate(ori_seg):
        w_char = tokenizer.tokenize(w)
        chars.extend(w_char)
        word_list.append((i, w, w_char, length, length + len(w_char)))
        length += len(w_char)
    #print(word_list)
    max_pos = 0
    pos_s = np.zeros((len(ori_seg),), dtype=np.int64)
    # [0, .., .., ], 单个 最多 latt 的句子
    pos_e = np.zeros((len(ori_seg),), dtype=np.int64)
    mat = np.zeros((len(word_list), length + 1), dtype=np.int64) # len(input) * length_char + 2
    forward_position = np.zeros(len(ori_seg))
    # [0.0, .., .., 0.0], 单个 最多 token 的句子
    backward_position = np.zeros(len(ori_seg))
    #print(mat.shape, len(word_list), length)
    for i, index in enumerate(word_list):
        s = index[-2]
        e = index[-1]
        pos_s[i] = s
        pos_e[i] = e
        forward_position[i] = s
        backward_position[i] = e
        max_pos = e if e > max_pos else max_pos
        for j in range(s, e):
            mat[i][j] = 1
    return mat, forward_position, backward_position, pos_s, pos_e, chars


def get_num_char_mat(num_pos, word_char_mat):
    mat = copy.deepcopy(word_char_mat)
    num_char_mat = []
    for n in num_pos:
        char_list = mat[n]
        temp_index = []
        for i, c in enumerate(char_list):
            if c == 1:
                temp_index.append(i)
        num_char_mat.append((n, temp_index))
    # [], len(num_pos) 个 (pos, [char_index, char_index,.. ])
    return num_char_mat


def pad_word_char_mat(ori_mat, input_len, char_len):
    unified_mat = np.zeros((input_len, char_len), dtype=np.int64)

    shape_raw = ori_mat.shape

    unified_mat[:shape_raw[0], :shape_raw[1]] = ori_mat

    return unified_mat


def pad_char(chars, max_len, tokenizer):
    # max_len 是  最长 +2 后了
    token = [CLS] + chars # 最长的也是小2

    mask = [1] * len(token_ids) + [0] * (max_len - len(token))


def array_seq(ori_seg, num_pos):
    seg = ''
    start = 0
    for index in num_pos:
        if start < len(ori_seg):
            seg += ''.join(ori_seg[start:index]) + ' ' + ori_seg[index]
            start = index + 1
    return seg