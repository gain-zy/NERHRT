# coding: utf-8
import re
import random
import tqdm
import copy
import numpy as np
from .expressions_transfer import from_infix_to_prefix
from .utils import indexes_from_sentence, pad_seq, get_adjacency_matrices_num, get_adjacency_matrices_token,\
    get_attribute_between_graph, get_parse_graph_batch, gen_word_char_mat, pad_word_char_mat, \
    get_num_char_mat, get_comparison_graph, compute_decimal, change_num


def transfer_num(data, unk2word_vocab):  # transfer num into "NUM"
    print("-"*40, ' Transfer numbers...', "-"*40)
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0

    unk2word = {}
    with open(unk2word_vocab, 'r', encoding='utf-8') as unk:
        for word in unk.readlines():
            unk2word[word.strip().split("###")[0]] = word.strip().split("###")[1]

    for d in tqdm.tqdm(data, desc=' 遍历 中 :  '):
        idx = d[0]
        nums = []
        input_seq = []
        ori_seg = []
        segment = d[2].strip().lower()
        for unk in unk2word:
            if unk in segment:
                segment = segment.replace(unk, unk2word[unk])
        seg = segment.split(' ')
        #ori_seq = copy.deepcopy(seg)

        equations = d[3]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                ori_seg.append(s[pos.start(): pos.end()])
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
                    ori_seg.append(s[pos.end():])
            else:
                if len(s) > 0:
                    input_seq.append(s)
                    ori_seg.append(s)
                else:
                    continue
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # input_seq, out_seq, nums, num_pos
        pairs.append((idx, input_seq, out_seq, nums, num_pos, ori_seg))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    print('-' * 40, "      Done  ", '-' * 40)
    return pairs, temp_g, copy_nums


def clean_question_data(pairs, logger):
    logger.info('训练、测试、验证数据集的总数：{} '.format(len(pairs)))
    count_comma = 0
    count_period = 0
    count_question = 0
    count_brackets = 0
    count_special_char = 0
    count_other = 0
    all_punc = ["．", "？", "（", "）", ",", "：", "；", "？", "！", "，", "“", "”", ",", ".", "?", "，", "。", "？", "．", "；", "｡"]
    comma = [",", "，", ",", "，"]
    period = ["｡", "．", "。"]
    question = ["？", "﹖"]
    brackets = ["）", ")"]
    special_char = ["=", "%", "NUM"]
    for p in pairs:
        idx = p['id']
        seg = p['ltp_seg']
        if seg[-1] in comma:
            count_comma += 1
        elif seg[-1] in period:
            # 90260 个 中文的 ？； 26个 "﹖"
            count_period += 1
        elif seg[-1] in question:
            # 22981 "？"  0 "﹖"
            count_question += 1
        elif seg[-1] in brackets:
            # 　180个 ")"
            # print(seg)
            count_brackets += 1
        elif seg[-1] in special_char:
            # 136 个 "="; 0个 "%"； 54 个 "NUM"
            count_special_char += 1
        else:
            # 基本是 符号和单词结尾。
            count_other += 1
    logger.info('逗号结尾的问题有: {} '.format(count_comma))
    logger.info('句号结尾的问题有: {}'.format(count_period))
    logger.info('问号结尾的问题有: {}'.format(count_question))
    logger.info('括号结尾的问题有: {}'.format(count_brackets))
    logger.info('特殊结尾的问题有: {}'.format(count_special_char))
    logger.info('其他结尾的问题有: {}'.format(count_other))
    logger.info('问题总数是否相等: {}'.format(count_comma + count_period + count_question + count_brackets + count_special_char + count_other == len(pairs)))

    temp_pairs = []
    for t in pairs:
        idx = t['id']
        ltp_seg = t['ltp_seg']
        parse = t['parse']
        seg = t['num_seg']
        equ = t['equation']
        num_exp = t['num_exp']
        nums = t['nums']
        num_pos = t['num_pos']
        ans = t['ans']

        punc = []
        punc_index = []
        for i, s in enumerate(seg):
            if s in all_punc:
                punc.append(s)
                punc_index.append(i)
        temp_pairs.append((idx, seg, parse, num_exp, nums, num_pos, ltp_seg, punc, punc_index))

    count = 0
    count_1 = 0
    count_2 = 0
    question_pairs = []
    for p in temp_pairs:
        idx = p[0]
        seg = p[1]
        parse = p[2]
        equ = p[3]
        num = p[4]
        num_pos = p[5]
        ori_seq = p[6]
        punc = p[7]
        punc_index = p[8]
        assert punc != punc_index

        if punc[-1] in ["？"]:
            count += 1
            if len(punc_index) >= 2:
                question_start = punc_index[-2] + 1
                question_end = punc_index[-1]
                # print(seg, question_start, question_end)from src.expressions_transfer import from_infix_to_prefix
                question_pairs.append((idx, seg, parse, equ, from_infix_to_prefix(equ), num, num_pos, ori_seq, punc,
                                       punc_index, (question_start, question_end)))
            else:
                question_start = 0
                question_end = punc_index[-1]
                question_pairs.append((idx, seg, parse, equ, from_infix_to_prefix(equ), num, num_pos, ori_seq, punc,
                                       punc_index, (question_start, question_end)))
        elif punc[-1] in ["）"]:
            brack_indices = punc_index[len(punc) - 1 - punc[::-1].index('（')]  # 找到正序的第一个"("
            # print('brack_indices', brack_indices)
            if len(punc) >= 4:
                count_1 += 1
                # 如果 （ 左边是其他符号
                if seg[brack_indices - 1] in punc:
                    # 怎么找到 这个其他符号的 更上面的一个符号？
                    temp_seg = seg[:brack_indices]
                    temp_punc = []
                    for i, t in enumerate(temp_seg):
                        if t in all_punc:
                            temp_punc.append(i)
                    question_start = temp_punc[-2] + 1
                    question_end = punc_index[-1]
                    question_pairs.append(
                        (idx, seg, parse, equ, from_infix_to_prefix(equ), num, num_pos, ori_seq, punc, punc_index,
                         (question_start, question_end)))
                    # print(idx, p[1], punc, punc_index, question_start, question_end)
                else:
                    # 和最近的 符号 距离 不等于1，就是在 远方
                    temp_seg = seg[:brack_indices]
                    temp_punc = []
                    for i, t in enumerate(temp_seg):
                        if t in all_punc:
                            temp_punc.append(i)
                    question_start = temp_punc[-1] + 1
                    question_end = punc_index[-1]
                    question_pairs.append(
                        (idx, seg, parse, equ, from_infix_to_prefix(equ), num, num_pos, ori_seq, punc, punc_index,
                         (question_start, question_end)))
                    # print('***',p[1],punc, punc_index, question_start, question_end)
            else:
                count_2 += 1
                # 相邻 而且 小于3, 整个句子都是
                if seg[brack_indices - 1] in punc:
                    question_pairs.append((idx, seg, parse, equ, from_infix_to_prefix(equ), num, num_pos, ori_seq, punc,
                                           punc_index, (0, punc_index[-1])))
                    # print('***',(idx, seg, equ, num, num_pos, punc, punc_index, (0, punc_index[-1])))
                elif len(punc) <= 2:
                    question_pairs.append((idx, seg, parse, equ, from_infix_to_prefix(equ), num, num_pos, ori_seq, punc,
                                           punc_index, (0, punc_index[-1])))
                    # print((idx, seg, equ, num, num_pos, punc, punc_index, (0, punc_index[-1])))
                # 不相邻 且 小于 3
                else:
                    temp_seg = seg[:brack_indices]
                    temp_punc = []
                    for i, t in enumerate(temp_seg):
                        if t in all_punc:
                            temp_punc.append(i)
                    question_start = temp_punc[-1] + 1
                    question_end = punc_index[-1]
                    question_pairs.append(
                        (idx, seg, parse, equ, from_infix_to_prefix(equ), num, num_pos, ori_seq, punc, punc_index,
                         (question_start, question_end)))
                    # print('===',(idx, seg, equ, num, num_pos, punc, punc_index, (question_start, question_end)))
    # for p in question_pairs:
    #     if p[-1][0] == 0:
    #         print(p)
    return question_pairs


class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            if re.search("N\d+|NUM|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print(' <----------------------------  keep_words %s / %s = %.4f  ----------------------------> ' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count, outlang_vocab):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
        '2022- 3- 13 这里加入了 output_lang的'
        self.index2word = outlang_vocab + ["PAD", "NUM", "UNK"] + self.index2word
        input_lang_vocab = []
        for word_ in self.index2word:
            if word_ not in input_lang_vocab:
                input_lang_vocab.append(word_)
        self.index2word = input_lang_vocab

        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
                          ["SOS", "UNK"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)

        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i


def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, cutting, tree=False):
    # input :
    # input_seg(NUM), pre_exp, num, num_pos, par_tree, group, ori_input, (q_s, q_e)
    #     p[0],        p[1],   p[2], p[3]  ,    p[4],   p[5],    p[6]       p[7]
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    # 之前: input, pre_exp, num, num_pos, par_tree, group, ori_input
    #        p[0]   p[1]    p[2]   p[3]      p[4]    p[5]    p[6]
    # 现在：
    # input_seg(NUM), pre_exp, num, num_pos, par_tree, group, ori_input, (q_s, q_e)
    #     p[0],        p[1],   p[2], p[3]  ,   p[4],   p[5],    p[6]       p[7]
    print('-'*40, " Indexing words... ", '-'*40)
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    '2022- 3- 13 加入了 output_lang'
    input_lang.build_input_lang(trim_min_count, output_lang.index2word)

    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)
            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])
        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        mat, _, _, _, _, _ = gen_word_char_mat(ori_seg=pair[6], tokenizer=cutting)
        # [], len(num_pos) 个 (pos, [char_index, char_index,.. ])
        num_dict = get_num_char_mat(pair[3], mat)

        assert mat.shape[0] == len(input_cell)
        # input : input, pre_exp, num, num_pos, par_tree, group, ori_input
        # input_seg, input_length, 每个词的词性, 每个词和哪个有关系， 中缀表达式， 后缀表达式，nums, num_pos
        train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                            pair[2], pair[3], num_stack, pair[4], pair[5], pair[7], pair[6], mat, num_dict))

    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    print('output_lang.index2word: ', output_lang.index2word)

    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)

        mat, _, _, _, _, _ = gen_word_char_mat(ori_seg=pair[6], tokenizer=cutting)
        # [], len(num_pos) 个 (pos, [char_index, char_index,.. ])
        num_dict = get_num_char_mat(pair[3], mat)

        assert mat.shape[0] == len(input_cell)
        test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                           pair[2], pair[3], num_stack, pair[4], pair[5], pair[7], pair[6], mat, num_dict))

    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs


def prepare_valid_data(input_lang, output_lang, valid_fold, cutting, tree=False):
    valid_pairs = []
    for pair in valid_fold:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)

        mat, _, _, _, _, _ = gen_word_char_mat(ori_seg=pair[6], tokenizer=cutting)
        # [], len(num_pos) 个 (pos, [char_index, char_index,.. ])
        num_dict = get_num_char_mat(pair[3], mat)

        assert mat.shape[0] == len(input_cell)
        valid_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                           pair[2], pair[3], num_stack, pair[4], pair[5], pair[7], pair[6], mat, num_dict))

    print('Number of testind data %d' % (len(valid_pairs)))
    return valid_pairs


def get_single_batch_graph(input_batch, input_length, group, num_value, num_pos, parse_tree_batch, seg_batch, k_hop=2):

    max_len = max(input_length)

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)

    num_graph, num_attention_mask = get_adjacency_matrices_num(num_value, num_pos, num_size)

    tok_graph, tok_attention_mask = get_adjacency_matrices_token(input_length, num_pos, max_len, group,
                                                                 parse_tree_batch, seg_batch)

    batch_graph = (num_graph, tok_graph)
    batch_attention_mask = (num_attention_mask, tok_attention_mask)

    # print(input_length)
    # print(group)
    # print(num_value)
    # print(num_pos)
    # print(parse_tree_batch)
    # print(seg_batch)
    # print(batch_graph)
    # print(batch_attention_mask)

    return batch_graph, batch_attention_mask


def get_single_example_graph(input_batch, input_length, group, num_value, num_pos, parse_tree_batch, seg_batch):

    num_size = len(num_pos)

    num_graph, num_attention_mask = get_adjacency_matrices_num([num_value], [num_pos], num_size)

    tok_graph, tok_attention_mask = get_adjacency_matrices_token([input_length], [num_pos], input_length, [group],
                                                                 [parse_tree_batch], [seg_batch])

    batch_graph = (num_graph, tok_graph)
    batch_attention_mask = (num_attention_mask, tok_attention_mask)

    return batch_graph, batch_attention_mask


def adj_to_bias(adj):
    # adj -> [1,3025,3025]
    # sizes -> [3025]
    nb_graphs = adj.shape[0] # 1
    sizes = [a.shape[0] for a in adj]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        # 对角元素上变为1
        mt[g] = np.eye(adj.shape[1])
        #for _ in range(1):
        mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    # [b, s, s]
    return -1e9 * (1.0 - mt)


def adj_bias(adj):
    # adj -> [1,3025,3025]
    # sizes -> [3025]
    mt = np.eye(adj.shape[0])
    mt = np.matmul(mt, (adj + np.eye(adj.shape[0])))
    np.where(mt > 0, 1.0, 0.0)
    # [b, s, s]
    return -1e9 * (1.0 - mt)


def get_single_word_num_mat(input_length, num_pos):
    graph = np.zeros((input_length, len(num_pos)))

    for i, index in enumerate(num_pos):
        graph[index][i] = 1

    return graph


def get_word_num_mat(input_len_max, num_pos_batch, max_num_len):
    batch_word2num = []

    for i in range(len(num_pos_batch)):
        graph = np.zeros((input_len_max, max_num_len))

        for j, index in enumerate(num_pos_batch[i]):
            graph[index][j] = 1

        batch_word2num.append(graph)

    return batch_word2num


# prepare the batches
def prepare_train_batch(pairs_to_batch, batch_size):
    # input_seg, input_length, output, output_length, num, num_pos, num_stack, parse_tree,
    # group, (q_s,q_e), ori, mat, num_mat
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    char_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    group_batches = []
    parse_tree_batches = []
    graph_batches = []
    num_value_batches = []
    mask_batches = []
    question_pos_batches = []
    question_size_batches = []
    content_pos_batches = []
    content_size_batches = []
    mat_batches = []
    seg_batches = []
    ori_batches = []
    num_dict_batches = []
    n_broadcast_2_w_batches = []
    nums_digit_value_batches = []
    nums_digit_pos_batches = []

    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos + batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        char_length = []
        # num_dict : [], len(num_pos) 个 (pos, [char_index, char_index,.. ])
        for _, i, _, j, _, _, _, _, _, _, ori_seg, mat, _ in batch:
            input_length.append(i)
            output_length.append(j)
            char_length.append(mat.shape[-1])
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        char_lengths.append(char_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        char_len_max = max(char_length)

        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        group_batch = []
        num_value_batch = []
        parse_tree_batch = []
        question_batch = []
        question_size_batch = []
        content_batch = []
        content_size_batch = []
        mat_batch = []
        num_char_dict = []
        seg_batch = []
        ori_batch = []
        nums_digit_value = []
        nums_digit_pos = []

        for i, li, j, lj, num, num_pos, num_stack, parse_tree, group, question, ori_seg, small_mat, num_dict in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            num_value_batch.append(num)
            group_batch.append(group)
            parse_tree_batch.append(parse_tree)
            question_batch.append(question)
            question_size_batch.append(question[1] - question[0])
            content_batch.append([0, question[0]])
            content_size_batch.append(question[0])
            mat_batch.append(pad_word_char_mat(small_mat, input_len_max, char_len_max))
            seg_batch.append(ori_seg)
            ori_batch.append(' '.join(ori_seg)) # B 个 seg, [ori_seq, ori_seq, ...]
            num_char_dict.append(num_dict)

            nums_problem_value = []
            nums_problem_pos = []
            nums = []
            temp_nums = change_num(num)
            for t in temp_nums:
                nums.append(str(t))
            for num_word in nums:
                if len(num_word) > 10:
                    num_word = str(round(float(num_word), 9))
                _, value, pos = compute_decimal(num_word)
                nums_problem_value.append(value)
                nums_problem_pos.append(pos)

            nums_digit_value.append(nums_problem_value)
            nums_digit_pos.append(nums_problem_pos)

        max_num_len = max(num_size_batch)
        n_broadcast_2_w = get_word_num_mat(input_len_max, num_pos_batch, max_num_len)

        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        num_value_batches.append(num_value_batch)
        group_batches.append(group_batch)
        parse_tree_batches.append(parse_tree_batch)
        # [b, 3, max_len, max_len]
        graph_batch, mask_batch = get_single_batch_graph(input_batch, input_length, group_batch,
                                                         num_value_batch, num_pos_batch,
                                                         parse_tree_batch, seg_batch)
        graph_batches.append(graph_batch)
        # [b, 3, max_len, max_len]
        mask_batches.append(mask_batch)
        question_pos_batches.append(question_batch)
        question_size_batches.append(question_size_batch)
        content_pos_batches.append(question_batch)
        content_size_batches.append(question_size_batch)
        mat_batches.append(mat_batch)
        seg_batches.append(seg_batch)
        ori_batches.append(ori_batch)
        num_dict_batches.append(num_char_dict)
        n_broadcast_2_w_batches.append(n_broadcast_2_w)
        nums_digit_value_batches.append(nums_digit_value)
        nums_digit_pos_batches.append(nums_digit_pos)

    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
           num_pos_batches, num_size_batches, num_value_batches, graph_batches, mask_batches, parse_tree_batches, \
           question_pos_batches, question_size_batches, mat_batches, char_lengths, seg_batches, \
           ori_batches, num_dict_batches, n_broadcast_2_w_batches, nums_digit_value_batches, nums_digit_pos_batches