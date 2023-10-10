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

# from ltp import LTP
from parameter import *
from src.utils import load_raw_data, gen_single_group_num, array_seq, time_since, compute_decimal, change_num
from src.train_and_evaluate import train_tree, compute_prefix_tree_result, evaluate_tree
from src.pre_data import transfer_num, clean_question_data, prepare_data, prepare_train_batch, \
    get_single_example_graph, prepare_valid_data, get_single_word_num_mat
from src.expressions_transfer import from_infix_to_prefix
from src.models import EncoderChar, EncoderSeq, EncoderNum, Prediction, GenerateNode, Merge, FusionReview


if SEED:
    torch.manual_seed(SEED) # 为CPU设置随机种子
    torch.cuda.manual_seed_all(SEED) # 为当前GPU设置随机种子
    np.random.seed(SEED) # 为np设定
    random.seed(SEED) # 为random设定

    logger.info(' 注意, 这里我们固定了随机种子为     : %d' % SEED)
else:
    logger.info(' 没有使用          随机种子        No ')


# ltp = LTP(path=ltp_path)

start = time.time()

# 23-10-10 省略数据预处理
with open('./data/math23k/me/ltp_seg_parse_math23k.json', 'r', encoding = 'utf-8') as f:
    data = json.load(f)    #此时a是一个字典对象
_, generate_nums, copy_nums = data['let_seg_parse_pairs'], data['math23k_constant'], data['max_expression_length']

with open('./data/math23k/me/ltp_parse_train.json', 'r', encoding = 'utf-8') as f:
    train_pairs = json.load(f)    #此时a是一个字典对象
with open('./data/math23k/me/ltp_parse_test.json', 'r', encoding = 'utf-8') as f:
    test_pairs = json.load(f)    #此时a是一个字典对象
with open('./data/math23k/me/ltp_parse_valid.json', 'r', encoding = 'utf-8') as f:
    valid_pairs = json.load(f)    #此时a是一个字典对象

'''
(0, 'id', '65')
(1, 'ltp_seg', ['张明', '有', '120', '元', '钱', '，', '买', '书', '用', '去', '80%', '，', '买', '文具', '的', '钱', '是', '买', '书', '的', '15%', '．', '买', '文具', '用', '去', '多少', '元', '？'])
(2, 'parse', [1, -1, 3, 4, 1, 1, 8, 6, 1, 8, 8, 1, 15, 12, 12, 16, 1, 20, 17, 17, 16, 1, 24, 22, 1, 24, 27, 24, 1])
(3, 'num_seg', ['张明', '有', 'NUM', '元', '钱', '，', '买', '书', '用', '去', 'NUM', '，', '买', '文具', '的', '钱', '是', '买', '书', '的', 'NUM', '．', '买', '文具', '用', '去', '多少', '元', '？'])
(4, 'equation', '120*80%*15%')
(5, 'num_exp', ['N0', '*', 'N1', '*', 'N2'])
(6, 'nums', ['120', '80%', '15%'])
(7, 'num_pos', [2, 10, 20])
(8, 'ans', '14.4')
'''
# 筛选q-s的下标
pairs = clean_question_data(train_pairs, logger) + clean_question_data(test_pairs, logger) + clean_question_data(valid_pairs, logger)

logger.info('准备基本数据花时间： ' + time_since(time.time()-start))

temp_pairs = []
for p in pairs:
    # return ：
    # input_seg(NUM), pre_exp, num, num_pos, par_tree, group, ori_input, (q_s, q_e)
    #     p[0],        p[1],   p[2], p[3]  ,    p[4],   p[5],    p[6]       p[7]
    temp_pairs.append((p[1], p[4], p[5], p[6], p[2], gen_single_group_num(p[1], len(p[1]), p[6]), p[7], (p[10][0], p[10][1]+1)))

print('-'*100)
for p in temp_pairs[12314]:
    print(p)
print('-'*100)

# 现在获得了， 数据 p[6] 就是
logger.info('                        随机按 最新论文进行 切分数据                        ')
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
# 测试
# valid_fold.extend(pairs[-20:])
# train_fold.extend(pairs[: 128])
# test_fold.extend(pairs[-100:-10])

logger.info('train的数据有 {}, test的数据有 {} , valid 的数据有 {}.'.format(len(train_fold), len(test_fold), len(valid_fold)))
pairs_trained = train_fold
pairs_tested = test_fold
pairs_valid = valid_fold

logger.info('-'*40 + '   开始设置 模型   ' + '-'*40)

bert = EncoderChar(bert_path=bert_path, bert_size=bert_hidden, hidden_size=hidden_size, get_word_and_sent=get_trans_flag)

start = time.time()
input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 3, generate_nums,
                                                                copy_nums, bert.tokenizer, tree=True)
# op_tokens = " ".join(output_lang.index2word[:5])
# temp = bert.tokenizer(op_tokens, padding=True, return_tensors="pt")
# output = bert.model(**temp)
# op_matrix = output.last_hidden_state.squeeze()[1:6].detach().numpy()

logger.info('构建input_vocab 与nv output_vocab 和 train and test 数据集  用时: ' + time_since(time.time()-start))
logger.info(output_lang.index2word)
# for p in train_pairs[12314]:
#     print(p)
# print('-'*100)
for i, p in enumerate(test_pairs[10]):
    print(p)
print('-'*100)

hir_feat = []
hir_vec = torch.FloatTensor([hir_ele for _ in range(hidden_size)])
for i in range(hir_layers):
    hir_feat.append(hir_vec)
    hir_vec = hir_vec*2

encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     type_list=subgraph_type_list, gru_layers=gru_layers, hop_layers=hir_layers)
#numStudent = NumberStudent(hidden_size=hidden_size)
#numGAT = numGAT(hidden_size, heads=1)
# question = QuestionEnc(hidden_size=hidden_size, n_layers=gru_layers, dropout=0.3)
# fusion = FusionReview(vec_size=hidden_size, kernels=kernels, activation='Relu', sigmoid='sigmoid', softmax='softmax',
#                      semantic_use_door=False)
'2022-3-27 暂时不用 之前的针对num特定模型'
number_enc = EncoderNum(hidden_size=hidden_size)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size, pretrain_emb=op_matrix)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
#numStudent_optimizer = torch.optim.Adam(numStudent.parameters(), lr=learning_rate, weight_decay=weight_decay)
number_optimizer = torch.optim.Adam(number_enc.parameters(), lr=learning_rate, weight_decay=weight_decay)
# param_optimizer = list(encoder_char.named_parameters())
# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
bert_optimizer = torch.optim.AdamW(bert.parameters(), lr=bert_learning_rate, weight_decay=bert_weight_decay)
# question_optimizer = torch.optim.Adam(question.parameters(), lr=learning_rate, weight_decay=weight_decay)
#fusion_optimizer = torch.optim.Adam(fusion.parameters(), lr=learning_rate, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
#numStudent_scheduler = torch.optim.lr_scheduler.StepLR(numStudent_optimizer, step_size=20, gamma=0.5)
number_scheduler = torch.optim.lr_scheduler.StepLR(number_optimizer, step_size=20, gamma=0.5)
bert_scheduler = torch.optim.lr_scheduler.StepLR(bert_optimizer, step_size=20, gamma=0.5)
# question_scheduler = torch.optim.lr_scheduler.StepLR(question_optimizer, step_size=20, gamma=0.5)
#fusion_scheduler = torch.optim.lr_scheduler.StepLR(fusion_optimizer, step_size=20, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)


# Move models to GPU
if device:
    encoder.to(device=device)
    bert.to(device=device)
    number_enc.to(device=device)
    predict.to(device=device)
    generate.to(device=device)
    merge.to(device=device)

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

# print(generate_num_ids)
# for p in train_pairs[1]:
#     print(p)
'''
[5, 6]
[24, 38, 24, 39, 28, 35, 40, 24, 30, 31, 41, 35, 42, 24, 30, 31, 43, 44, 45, 32, 34, 46, 37]
23
[0, 2, 9, 10, 9]
5
['5', '1', '50', '21']
[0, 2, 7, 13]
[]
[4, 2, 4, 2, 5, -1, 8, 8, 5, 5, 11, 5, 14, 14, 11, 5, 17, 5, 19, 21, 19, 17, 5]
[0, 1, 1, 2, 3, 6, 7, 8, 12, 13, 14]
(16, 23)
['5', '（', '1', '）', '班', '有', '学生', '50', '人', '，', '其中', '有', '女生', '21', '人', '，', '男生', '占', '全班', '人数', '的', '几分之几', '？']
[[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]]
[(0, [1]), (2, [3]), (7, [9]), (13, [17])]
'''
final_acc_fold = []
down_ans_all = []
loss_list = []
equ_acc_list = []
val_acc_list = []
last_value_acc = 0.
last_epoch = 0
temp_value_acc = 0.

for epoch in range(n_epochs):
    loss_total = 0
    start = time.time()
    '''
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
    num_pos_batches, num_size_batches, num_value_batches, graph_batches, mask_batches, parse_tree_batches, \
    question_pos_batches, question_size_batches, mat_batches, char_lengths, seg_batches, \
    ori_batches, num_dict_batches, n_broadcast_2_w_batches
    '''
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
    num_pos_batches, num_size_batches, num_value_batches, graph_batches, mask_batches, parse_tree_batches, \
    question_pos_batches, question_size_batches, mat_batches, char_lengths, seg_batches, \
    ori_batches, num_dict_batches, n_broadcast_2_w_batches, nums_digit_value_batches,\
    nums_digit_pos_batches = prepare_train_batch(train_pairs, batch_size)

    # print(' 这是一个 batch 的数据 ')
    # print(' input_batches[0]: ', input_batches[0])
    # print(' input_lengths[0]: ', input_lengths[0])
    # print(' output_batches[0]: ', output_batches[0])
    # print(' output_lengths[0]: ', output_lengths[0])
    # print(' nums_batches[0]: ', nums_batches[0])
    # print(' num_value_batches[0]: ', num_value_batches[0])
    # print(' num_stack_batches[0]: ', num_stack_batches[0])
    # print(' num_pos_batches[0]: ', num_pos_batches[0])
    # print(' num_size_batches[0]: ', num_size_batches[0])
    # print(' parse_tree_batches[0]: ', parse_tree_batches[0])
    # print(' seg_batches[0]: ', seg_batches[0])
    # print(' ori_batches[0]: ', ori_batches[0])
    # print(' graph_batches[0]: ', graph_batches[0])
    # print(' graph_batches[0][1][1]: ', graph_batches[0][1][1])
    # print(' mask_batches[0]: ', mask_batches[0])
    # print(' mask_batches[0][1][1]: ', mask_batches[0][1][1])
    # assert 0==1

    # # print(len(input_batches[2][3]), input_lengths[2][3], mat_batches[2][3].shape, mat_batches[2][4].shape)

    logger.info('-' * 20 + "每个epochs准备数据的时长: " + time_since(time.time() - start) + '-' * 20)
    logger.info("epoch: {} ".format(epoch + 1))
    start = time.time()
    for idx in tqdm.tqdm(range(len(input_lengths))):
        loss = train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, number_enc, bert,
            predict, generate, merge, encoder_optimizer, number_optimizer, bert_optimizer, predict_optimizer,
            generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx], num_value_batches[idx],
            graph_batches[idx], mask_batches[idx], parse_tree_batches[idx], question_pos_batches[idx],
            question_size_batches[idx], mat_batches[idx], char_lengths[idx], seg_batches[idx], ori_batches[idx],
            num_dict_batches[idx], n_broadcast_2_w_batches[idx], nums_digit_value_batches[idx],
            nums_digit_pos_batches[idx], hir_feat, device)
        loss_total += loss

    encoder_scheduler.step()
    number_scheduler.step()
    bert_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()

    logger.info("loss: {} ".format(loss_total / len(input_lengths)))
    loss_list.append(loss_total / len(input_lengths))
    logger.info("training time： {} ".format(time_since(time.time() - start)))
    logger.info("-" * 100)
    if epoch % 6 == 0 or epoch > n_epochs - 5:
        down_ans = []
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        for test_batch in tqdm.tqdm(test_pairs):
            # print(test_batch)
            # test_batch : input_seg, input_length, output, output_length, num, num_pos, num_stack,
            #                   0         1            2          3         4       5        6
            #              parse_tree, group, (q_s,q_e), ori, mat, num_mat
            #                   7         8         9     10   11     12
            # need : input_batch, input_length, group, num_value, num_pos, parse_tree_batch, seg_batch
            graph_batch, mask_batch = get_single_example_graph(
                test_batch[0], test_batch[1], test_batch[8], test_batch[4], test_batch[5], test_batch[7],
                test_batch[10])

            nums = []
            temp_nums = change_num(test_batch[4])
            for t in temp_nums:
                nums.append(str(t))
            nums_problem_value = []
            nums_problem_pos = []
            for num_word in nums:
                if len(num_word) > 10:
                    num_word = str(round(float(num_word), 9))
                _, value, pos = compute_decimal(num_word)
                nums_problem_value.append(value)
                nums_problem_pos.append(pos)

            n_broadcast_2_w = get_single_word_num_mat(test_batch[1], test_batch[5])

            # need: input_batch, input_length, generate_nums, encoder, fusion, bert,
            #       predict, generate, merge, output_lang, nums, num_pos, graph_batches, mask_batches,
            #       question_pos, mat, ori_datas, n_broadcast_2_w,
            #       beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, number_enc,
                                     bert, predict, generate, merge, output_lang, test_batch[4],
                                     test_batch[5], graph_batch, mask_batch, test_batch[9],
                                     test_batch[11], test_batch[10], n_broadcast_2_w, hir_feat, nums_problem_value,
                                     nums_problem_pos, beam_size=beam_size)
            '''
            input_batch, input_length, generate_nums, encoder, fusion, bert,
                  predict, generate, merge, output_lang, nums, num_pos, group_batches, parse_tree_batches,
                  hir_feat, question_pos, mat,
                  ori_datas, num_dict, word2num, beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):
            '''
            # need: test_res, test_tar, output_lang, num_list, num_stack
            val_ac, equ_ac, test_equ, tar_equ = compute_prefix_tree_result(
                test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            if not val_ac or not equ_ac:
                down_ans.append((test_batch[1], test_batch[3], val_ac, equ_ac, test_equ, tar_equ))
            eval_total += 1
        down_ans_all = down_ans
        logger.info('equation_ac(表达式准确数量)：{} . value_ac(答案准确数量) : {}; eval_total(测试总数) : {}'.format(equation_ac, value_ac,  eval_total))
        equ_ac_percent = float(equation_ac) / eval_total
        val_ac_percent = float(value_ac) / eval_total
        logger.info("test equation acc {:.4}, and value acc {:.4}".format(equ_ac_percent, val_ac_percent))
        logger.info('which error num: {} '.format(len(down_ans)))
        logger.info("testing time {} ".format(time_since(time.time() - start)))
        if last_value_acc < val_ac_percent:
            last_value_acc = val_ac_percent
            logger.info('\n * 目前, 最优的 结果 -->[[[准确度]]]<-- 是 :  {} * '.format(last_value_acc))
        if val_ac_percent >= target_ans_acc or epoch == n_epochs-1:
            logger.info('\n * 模型正在保存! * ')
            torch.save(encoder.state_dict(), "./models/encoder" + str(epoch))
            torch.save(bert.state_dict(), "./models/bert" + str(epoch))
            torch.save(number_enc.state_dict(), "./models/number" + str(epoch))
            torch.save(predict.state_dict(), "./models/predict" + str(epoch))
            torch.save(generate.state_dict(), "./models/generate" + str(epoch))
            torch.save(merge.state_dict(), "./models/merge" + str(epoch))
            logger.info('\n * 模型保存完毕! * ')
            if temp_value_acc <= val_ac_percent:
                last_epoch = epoch
                logger.info('\n * 目前 结果 准确度 最优 的 -->[[[epoch]]]<-- 是: {}'.format(last_epoch))
                temp_value_acc = val_ac_percent

        equ_acc_list.append((epoch, equation_ac))
        val_acc_list.append((epoch, value_ac))

        if epoch == n_epochs - 1:
            final_acc_fold.append((equation_ac, value_ac, eval_total))
        logger.info("-" * 80)

a, b, c = 0, 0, 0
for bl in range(len(final_acc_fold)):
    a += final_acc_fold[bl][0]
    b += final_acc_fold[bl][1]
    c += final_acc_fold[bl][2]
    logger.info('最终acc的准确度 ： {}'.format(final_acc_fold[bl]))
logger.info('结果分别 : {}, {} '.format(a / float(c), b / float(c)))

# down_ans_all.append(output_lang)
# with open('ReadMe/final_error_data.pkl', 'wb') as f:
#    pickle.dump(down_ans_all, f)

logger.info('+'*100)
logger.info('+'*100)
logger.info('='*30 + '开始测试 valid 数据... ' + '='*30)
logger.info('+'*100)
logger.info('+'*100)

if os.path.exists("./models/encoder" + str(last_epoch)):
    logger.info(' * 调用最优 epoch 的模型，进行加载  , epoch 是 ： {}'.format( last_epoch))
    encoder.load_state_dict(torch.load("models/encoder" + str(last_epoch)))
    bert.load_state_dict(torch.load("models/bert" + str(last_epoch)))
    number_enc.load_state_dict(torch.load("models/number" + str(last_epoch)))
    predict.load_state_dict(torch.load("models/predict" + str(last_epoch)))
    generate.load_state_dict(torch.load("models/generate" + str(last_epoch)))
    merge.load_state_dict(torch.load("models/merge" + str(last_epoch)))
else:
    logger.info(' * 没有保存最优的 epoch, 调用最后的 epoch 的参数学习. ')

# Move models to GPU
if USE_CUDA:
    bert.to(device=device)
    number_enc.to(device=device)
    predict.to(device=device)
    generate.to(device=device)
    merge.to(device=device)

start = time.time()
valid_datas = prepare_valid_data(input_lang, output_lang, pairs_valid, bert.tokenizer, tree=True)

logger.info('                      valid 数据集  用时: {}, 总数 {}                       '.format(time_since(time.time()-start), len(valid_datas)))
for p in valid_datas[1]:
    print(p)
print('-'*100)

down_ans = []
value_ac = 0
equation_ac = 0
eval_total = 0
start = time.time()
for test_batch in tqdm.tqdm(test_pairs):
    # print(test_batch)
    # test_batch : input_seg, input_length, output, output_length, num, num_pos, num_stack,
    #                   0         1            2          3         4       5        6
    #              parse_tree, group, (q_s,q_e), ori, mat, num_mat
    #                   7         8         9     10   11     12
    # need : input_batch, input_length, group, num_value, num_pos, parse_tree_batch, seg_batch
    graph_batch, mask_batch = get_single_example_graph(
        test_batch[0], test_batch[1], test_batch[8], test_batch[4], test_batch[5], test_batch[7],
        test_batch[10])

    nums = []
    temp_nums = change_num(test_batch[4])
    for t in temp_nums:
        nums.append(str(t))
    nums_problem_value = []
    nums_problem_pos = []
    for num_word in nums:
        if len(num_word) > 10:
            num_word = str(round(float(num_word), 9))
        _, value, pos = compute_decimal(num_word)
        nums_problem_value.append(value)
        nums_problem_pos.append(pos)

    n_broadcast_2_w = get_single_word_num_mat(test_batch[1], test_batch[5])

    # need: input_batch, input_length, generate_nums, encoder, fusion, bert,
    #       predict, generate, merge, output_lang, nums, num_pos, graph_batches, mask_batches,
    #       question_pos, mat, ori_datas, n_broadcast_2_w,
    #       beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):
    test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, number_enc,
                             bert, predict, generate, merge, output_lang, test_batch[4],
                             test_batch[5], graph_batch, mask_batch, test_batch[9],
                             test_batch[11], test_batch[10], n_broadcast_2_w, hir_feat, nums_problem_value,
                             nums_problem_pos, beam_size=beam_size)
    '''
    input_batch, input_length, generate_nums, encoder, fusion, bert,
          predict, generate, merge, output_lang, nums, num_pos, group_batches, parse_tree_batches,
          hir_feat, question_pos, mat,
          ori_datas, num_dict, word2num, beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):
    '''
    # need: test_res, test_tar, output_lang, num_list, num_stack
    val_ac, equ_ac, test_equ, tar_equ = compute_prefix_tree_result(
        test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
    if val_ac:
        value_ac += 1
    if equ_ac:
        equation_ac += 1
    if not val_ac or not equ_ac:
        down_ans.append((test_batch[1], test_batch[3], val_ac, equ_ac, test_equ, tar_equ))
    eval_total += 1
down_ans_all = down_ans
print(' * 验证数据集结果如下： ')
print('equation_ac(表达式准确数量)：{} . value_ac(答案准确数量) : {}; eval_total(测试总数) : {}'.format(equation_ac, value_ac,  eval_total))
print("test equation acc {:.3}, and value acc {:.3}".format(float(equation_ac) / eval_total, float(value_ac) / eval_total))
print('which error num: {}'.format(len(down_ans)))
print(" * 验证数据集 所花时间   testing time: {}".format(time_since(time.time() - start)))

if len(os.listdir('./models')) != 0:
    logger.info('-'*100)
    logger.info('                 保存数据                  ')
    with open(processed_data_path, 'wb') as f:
         pickle.dump((input_lang, output_lang, last_epoch), f)
else:
    logger.info('         -----------> Failed <---------          ')