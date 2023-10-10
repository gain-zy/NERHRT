# coding: utf-8
import os
import pickle
import tqdm
import re
import json
import time
import random
import numpy as np
import copy

from ltp import LTP
from parameter import *
from src.utils import load_raw_data, gen_single_group_num, array_seq, time_since
from src.train_and_evaluate import train_tree, sta_compute_prefix_tree_result, evaluate_tree
from src.pre_data import transfer_num, clean_question_data, prepare_data, prepare_train_batch, \
    get_single_example_graph, prepare_valid_data, get_single_word_num_mat
from src.expressions_transfer import from_infix_to_prefix
from src.models import EncoderChar, EncoderSeq, QuestionEnc, Prediction, GenerateNode, Merge, FusionReview

bert_epoch = [66]

for last_epoch in bert_epoch:

    if SEED:
        torch.manual_seed(SEED)  # 为CPU设置随机种子
        torch.cuda.manual_seed_all(SEED)  # 为当前GPU设置随机种子
        np.random.seed(SEED)  # 为np设定
        random.seed(SEED)  # 为random设定

        print(' 注意, 这里我们固定了随机种子为     : %d' % SEED)
    else:
        print(' 没有使用          随机种子        No ')

    ltp = LTP(path=ltp_path)


    start = time.time()

    with open('./data/math23k/me/ltp_seg_parse_math23k.json', 'r', encoding = 'utf-8') as f:
        data = json.load(f)    #此时a是一个字典对象
    _, generate_nums, copy_nums = data['let_seg_parse_pairs'], data['math23k_constant'], data['max_expression_length']

    with open('./data/math23k/me/ltp_parse_train.json', 'r', encoding = 'utf-8') as f:
        train_pairs = json.load(f)    #此时a是一个字典对象
    with open('./data/math23k/me/ltp_parse_test.json', 'r', encoding = 'utf-8') as f:
        test_pairs = json.load(f)    #此时a是一个字典对象
    with open('./data/math23k/me/ltp_parse_valid.json', 'r', encoding = 'utf-8') as f:
        valid_pairs = json.load(f)    #此时a是一个字典对象

    pairs = clean_question_data(train_pairs, logger) + clean_question_data(test_pairs, logger) + clean_question_data(valid_pairs, logger)

    print('准备基本数据花时间： ' + time_since(time.time()-start))

    temp_pairs = []
    for p in pairs:
        # return ：
        # input_seg(NUM), pre_exp, num, num_pos, par_tree, group, ori_input, (q_s, q_e)
        #     p[0],        p[1],   p[2], p[3]  ,    p[4],   p[5],    p[6]       p[7]
        temp_pairs.append((p[1], p[4], p[5], p[6], p[2], gen_single_group_num(p[1], len(p[1]), p[6]), p[7], (p[10][0], p[10][1]+1)))

    # print('-'*100)
    # for p in temp_pairs[12314]:
    #     print(p)
    # print('-'*100)

    # 现在获得了， 数据 p[6] 就是
    print('                        随机按 最新论文进行 切分数据                        ')
    pairs = copy.deepcopy(temp_pairs)
    # 结果显示还是得随机，随机的效果好一点。
    random.shuffle(pairs)  # shuffle the pairs
    train_fold = []
    test_fold = []
    valid_fold = []
    fold_size = 1000
    valid_fold.extend(pairs[-fold_size:])
    train_fold.extend(pairs[: -2*fold_size])
    test_fold.extend(pairs[-2*fold_size:-fold_size])

    print('train的数据有 {}, test的数据有 {} , valid 的数据有 {}.'.format(len(train_fold), len(test_fold), len(valid_fold)))
    pairs_trained = train_fold
    pairs_tested = test_fold
    pairs_valid = valid_fold

    print('-'*40 + '   开始设置 模型   ' + '-'*40)

    bert = EncoderChar(bert_path=bert_path, bert_size=bert_hidden, hidden_size=hidden_size, get_word_and_sent=get_trans_flag)

    if os.path.exists(processed_data_path):
        print('-'*40, ' 可以获取input and output 的 vocab ', '-'*40)
        with open(processed_data_path, 'rb') as f:
            input_lang, output_lang, _ = pickle.load(f)
        _, _, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 3, generate_nums,
                                                     copy_nums, bert.tokenizer, tree=True)
        print('-' * 40, ' 获取input_lange, output_lang 结束 ', '-' * 40)
    else:
        print(' * Error: 自己生成 * ')
        input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 3, generate_nums,
                                                                        copy_nums, bert.tokenizer, tree=True)


    encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                         type_list=subgraph_type_list, gru_layers=gru_layers, hop_layers=hir_layers)
    #numStudent = NumberStudent(hidden_size=hidden_size)
    #numGAT = numGAT(hidden_size, heads=1)
    # question = QuestionEnc(hidden_size=hidden_size, n_layers=gru_layers, dropout=0.3)
    fusion = FusionReview(vec_size=hidden_size, kernels=kernels, activation='Relu', sigmoid='sigmoid', softmax='softmax',
                          semantic_use_door=False)
    '2022-3-27 暂时不用 之前的针对num特定模型'
    #enc_num = EncoderNum(input_size=bert_hidden, hidden_size=bert_hidden, word_hidden=hidden_size, gru_layers=gru_layers)
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                         input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=embedding_size, pretrain_emb=op_matrix)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

    print('last_epoch: ', last_epoch)
    if os.path.exists("./models/encoder" + str(last_epoch)):
        encoder.load_state_dict(torch.load("models/encoder" + str(last_epoch), map_location=device))
        fusion.load_state_dict(torch.load("models/fusion" + str(last_epoch), map_location=device))
        bert.load_state_dict(torch.load("models/bert" + str(last_epoch), map_location=device))
        predict.load_state_dict(torch.load("models/predict" + str(last_epoch), map_location=device))
        generate.load_state_dict(torch.load("models/generate" + str(last_epoch), map_location=device))
        merge.load_state_dict(torch.load("models/merge" + str(last_epoch), map_location=device))
    else:
        print(' 没有模型 ')
        assert 0==1
    # Move models to GPU
    if USE_CUDA:
        encoder.to(device=device)
        bert.to(device=device)
        fusion.to(device=device)
        predict.to(device=device)
        generate.to(device=device)
        merge.to(device=device)

    print(' * 调用 最优 模型 完毕 , the process of load models are Done... * ')
    start = time.time()
    valid_datas = prepare_valid_data(input_lang, output_lang, pairs_valid, bert.tokenizer, tree=True)

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    print(' valid 数据集  用时: {}, 总数 {} '.format(time_since(time.time()-start), len(valid_datas)))

    down_ans = []
    value_ac = 0
    equation_ac = 0
    eval_total = 0
    error_total = 0
    start = time.time()
    for test_batch in tqdm.tqdm(test_pairs):
        # print(test_batch)
        # test_batch : input_seg, input_length, output, output_length, num, num_pos, num_stack,
        #                   0         1            2          3         4       5        6
        #              parse_tree, group, (q_s,q_e), ori, mat, num_mat
        #                   7         8         9     10   11     12
        # need : input_batch, input_length, group, num_value, num_pos, parse_tree_batch
        # batch_graph, batch_bias = get_single_example_graph(test_batch[0], test_batch[1], test_batch[8],
        #                                                    test_batch[4], test_batch[5], test_batch[7])
        graph_batch, mask_batch = get_single_example_graph(
            test_batch[0], test_batch[1], test_batch[8], test_batch[4], test_batch[5], test_batch[7], test_batch[10])

        n_broadcast_2_w = get_single_word_num_mat(test_batch[1], test_batch[5])

        # need: input_batch, input_length, generate_nums, encoder, fusion, bert,
        #       predict, generate, merge, output_lang, nums, num_pos, graph_batches, mask_batches,
        #       question_pos, mat, ori_datas, n_broadcast_2_w,
        #       beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):
        test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder,
                                 fusion, bert, predict, generate, merge, output_lang, test_batch[4],
                                 test_batch[5], graph_batch, mask_batch, test_batch[9],
                                 test_batch[11], test_batch[10], n_broadcast_2_w, beam_size=beam_size)
        # need: test_res, test_tar, output_lang, num_list, num_stack
        val_ac, equ_ac, test_equ, tar_equ = sta_compute_prefix_tree_result(
            test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
        if val_ac:
            value_ac += 1
        if equ_ac:
            equation_ac += 1
        if not val_ac:
            error_total += 1
        down_ans.append({'seg':test_batch[10], 'exp':test_batch[2], 'val_acc':val_ac, 'equ_acc':equ_ac, 'gen_euq':test_equ, 'tar_equ':tar_equ})
        eval_total += 1
    down_ans_all = json.dumps(down_ans, ensure_ascii=False, indent=2)
    with open(r'error_ana/ans.json', 'w', encoding='utf-8') as f:
        f.write(down_ans_all)
    print(' * 验证数据集结果如下： ')
    print('equation_ac(表达式准确数量)：{} . value_ac(答案准确数量) : {}; eval_total(测试总数) : {}'.format(equation_ac, value_ac,  eval_total))
    print("test equation acc {:.3}, and value acc {:.3}".format(float(equation_ac) / eval_total, float(value_ac) / eval_total))
    print('which error num: {}'.format(error_total))
    print(" * 验证数据集 所花时间   testing time: {}".format(time_since(time.time() - start)))