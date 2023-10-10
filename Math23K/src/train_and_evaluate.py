# coding: utf-8
import math
import torch
import torch.optim
import torch.nn.functional as f
import copy
import torch.nn as nn
import random
import numpy as np

from src.masked_cross_entropy import masked_cross_entropy
from src.utils import PAD_token, get_all_question_encoder_outputs, change_num
from src.expressions_transfer import out_expression_list, compute_postfix_expression, compute_prefix_expression
from src.models import TreeNode
from parameter import device


MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def sta_compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, out_expression_list(test_res, output_lang, num_list), out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, contexts, goals, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)
        self.contexts = copy_list(contexts)
        self.goals = copy_list(goals)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if device:
        indices = indices.to(device)
        masked_index = masked_index.to(device)
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill(masked_index.bool(), 0.0)


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    # max_out × B 选一个; list有 B × [op + O];  num_stack； 5； unk
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        # 如果label 是未知的,
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start # 用stack替换
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    # B 个 label, unk 替换成stack的数字; B 个 label, 除了5个op 全部是0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def devide_numtype(num_str):
    #integer, decimal, fraction, percentage
    num_type = 1
    if "%" in num_str:
        num_type = 4
    elif "/" in num_str:
        num_type = 3
    elif "." in num_str:
        num_type = 2
    else:
        num_type = 1
    return num_type


def num_pre():
    distance_loss, compare_loss, cate_loss = 0., 0., 0.
    return distance_loss, compare_loss, cate_loss


def get_adjacency_matrices_token(batch_lengths, batch_num_pos, max_len, group_batches,
                                 parse_tree_batches, seg_batches, device: torch.device):
    '''
    '''
    # print('batch_lengths: ', batch_lengths)
    # print('parse_tree_batches: ', parse_tree_batches)
    # print('seg_batches: ', seg_batches)
    # print('batch_num_pos: ', batch_num_pos)
    # print('group_batches: ', group_batches)
    batch_size = len(batch_lengths)
    token_graph = torch.zeros((batch_size, max_len, max_len))
    attention_mask = torch.zeros((batch_size, max_len, max_len), device=device)
    attention_mask += -1e9

    for b, length in enumerate(batch_lengths):

        attention_mask[b, :, :length] = 0

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
    return token_graph.to(device), attention_mask.to(device)


def train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, number_enc, bert, predict, generate, merge, encoder_optimizer, number_optimizer,
               bert_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_pos, nums, graph_batches, mask_batches,
               parse_tree_batches, question_pos, question_size, mat, char_length, seg_batches, ori_datas,
               num_dict, n_broadcast_2_w, nums_digit_batch, nums_pos_batch, hir_feat, english=False):
    # graph tuple -> (B x N x N, B x S x S)
    # mask_batches -> (B x N x N, B x S x S)
    # mat是一个list, 里面是[S, C_s]
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask) # B × max_len

    # print("max_len : ", max_len)
    # print('seq_mask : ', seq_mask)

    max_nums = max(num_size_batch)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)
    mat = torch.FloatTensor(np.array(mat))
    # B x S x N --> B x N x S
    n_extract_f_w = torch.FloatTensor(np.array(n_broadcast_2_w)).transpose(1, 2)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    num_padding_hidden = torch.FloatTensor([0.0 for _ in range(encoder.hidden_size)])

    num_graph = graph_batches[0]
    tok_graph = graph_batches[1]
    num_attention_mask = mask_batches[0]
    tok_attention_mask = mask_batches[1]

    encoder.train()
    number_enc.train()
    bert.train()
    predict.train()
    generate.train()
    merge.train()

    if device:
        # input_var = input_var.to(device)
        seq_mask = seq_mask.to(device)
        padding_hidden = padding_hidden.to(device)
        num_mask = num_mask.to(device)
        mat = mat.to(device)
        hir_feat = torch.stack(hir_feat).to(device)
        n_extract_f_w = n_extract_f_w.to(device)
        num_graph = num_graph.to(device)
        tok_graph = tok_graph.to(device)
        num_attention_mask = num_attention_mask.to(device)
        tok_attention_mask = tok_attention_mask.to(device)
        num_padding_hidden = num_padding_hidden.to(device)

    encoder_optimizer.zero_grad()
    number_optimizer.zero_grad()
    bert_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()

    word_emb, sent_emb = bert(ori_datas, char_length, mat, True) # B x S x 512

    num_emb = torch.matmul(n_extract_f_w, word_emb) # B x N x H

    question_max_size = max(question_size)

    decimal_num_emb = number_enc(batch_size, nums_digit_batch, nums_pos_batch, max_nums, num_padding_hidden)

    '2022-3-28 放进enecoder中'
    # encoder_outputs.shape = S * B * H
    # return : B x H; S x B x H; S x B x 2H
    '''
    input_seqs, input_lengths, max_len, num_emb, word_emb, num_graph, num_attention_mask,
    tok_graph, tok_attention_mask, question_lengths, question_pos, question_max_size, hidden=None, sorted=False
    '''
    problem_outputs, inherit_root, encoder_outputs, question_feature = encoder(
        input_var, input_length, max_len, num_emb, decimal_num_emb, word_emb, num_graph, num_attention_mask,
        tok_graph, tok_attention_mask, batch_size, question_size, question_pos, question_max_size)

    all_nums_encoder_outputs = torch.matmul(n_extract_f_w, encoder_outputs.transpose(0,1)) # B x N x H

    max_target_length = max(target_length)
    #print(max_target_length)

    all_node_outputs = []
    num_start = output_lang.num_start
    # Prepare input and output variables
    'B × H ---> 1 × H'
    node_stacks = [[TreeNode(_)] for _ in problem_outputs.split(1, dim=0)]
    # inherit_stacks = [[parent_know] for parent_know in inherit_root.split(1, dim=0)]
    # print('len(node_stacks), len(inherit_stacks): ', len(node_stacks), len(inherit_stacks))
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    contexts = []
    goals = []

    for t in range(max_target_length):
        ' score(cat([goal vector], [环境向量c]) + e(y|P));   B × op;  '
        ' [goal vector] b × 1 × h;   [环境向量c] B x 1 x H; 数字emb B x O x H'
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        contexts.append(current_context)
        goals.append(current_embeddings)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1) # B × [op + O]
        all_node_outputs.append(outputs)
        # target.shape = max_out × B, 选一个; list有 B × [op + O];  num_stack； 5； unk
        # return : B 个 label, unk 替换成stack的数字; B 个 label, 除了5个op 全部是0
        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t # 通过 outputs 填充，所以有学习痕迹
        if device:
            generate_input = generate_input.to(device)
        # [goal vector] b × 1 × h；  B 个 label, 除了5个op 全部是0； [环境向量c] B x 1 x H
        # return: h_l, B x H ; h_r, B x H ; B x E, B 个 label, 除了5个op， 其他全部是0 转换的 emb
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context, contexts, goals, hir_feat)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            # id @ 1 x H @ 1 x H @  node_stack @ 1个 label, unk 替换成stack的数字 @ []
            'o 好像是 op 的存储， node_stack 存储的是 leaf'

            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            # 当 label 是 op 时候，这里需要捕捉额外的信息 和 利用 父亲流动节点 融合
            if i < num_start:
                # 最终融合的 s x b x h ---> 1 x 1 x S x H; 融合数字之前的 q x b x h.
                #l, r, child_information = fusion(encoder_outputs[:, idx, :].unsqueeze(0).unsqueeze(0), node.embedding,
                #                                 question_feature[:, idx, :].unsqueeze(0), r, l)
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                # append 1 x E, false 继续
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                # 当 label 是 [叶子结点]=数字 时候；选出了学习到的 num_emb
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                # 迭代 到 o 空 或者 ：emb_stack[op]不是空, 最后一个 node  flag=True[ 叶子节点 ]
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop() # 拿出 别的 op, 其实是子树,一般是左子树
                    op = o.pop() # [op]， 母树
                    # op; 左子树； 最近节点;
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            # emb_stack[op]不是空, 最后一个 node  flag=True[ 叶子节点 ]
            ' 当左子树 已经结束了， 开始右节点时候，右边节点通过上述方法生成了，这时候，left_chailds 就开始记录了左边子树的 embedding'
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if device:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.to(device)
        target = target.to(device)

    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    loss.backward()

    encoder_optimizer.step()
    number_optimizer.step()
    bert_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()


def evaluate_tree(input_batch, input_length, generate_nums, encoder, number_enc, bert,
                  predict, generate, merge, output_lang, nums, num_pos, graph_batches, mask_batches,
                  question_pos, mat, ori_datas, n_broadcast_2_w, hir_feat, nums_digit_single, nums_pos_single,
                  beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # batch_graph = torch.LongTensor(batch_graph)
    # batch_bias = torch.FloatTensor(batch_bias)

    mat = torch.FloatTensor(np.array([mat]))
    max_nums = len(nums)

    # 1 * S * N ---> 1 x N x S
    n_extract_f_w = torch.FloatTensor(np.array([n_broadcast_2_w])).transpose(1, 2)

    num_padding_hidden = torch.FloatTensor([0.0 for _ in range(encoder.hidden_size)])

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    num_graph = graph_batches[0]
    tok_graph = graph_batches[1]
    num_attention_mask = mask_batches[0]
    tok_attention_mask = mask_batches[1]

    batch_size = 1

    # Set to not-training mode to disable dropout
    encoder.eval()
    number_enc.eval()
    bert.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    if device:
        input_var = input_var.to(device)
        seq_mask = seq_mask.to(device)
        padding_hidden = padding_hidden.to(device)
        num_mask = num_mask.to(device)
        mat = mat.to(device)
        hir_feat = torch.stack(hir_feat).to(device)
        n_extract_f_w = n_extract_f_w.to(device)
        num_graph = num_graph.to(device)
        tok_graph = tok_graph.to(device)
        num_attention_mask = num_attention_mask.to(device)
        tok_attention_mask = tok_attention_mask.to(device)
        num_padding_hidden = num_padding_hidden.to(device)

    # Run words through encoder
    word_emb, sent_emb = bert(' '.join(ori_datas), [mat.shape[-1]], mat, out_all_hidden=True)  # B[1] * char_len_max * 512

    num_emb = torch.matmul(n_extract_f_w, word_emb)  # B[1] x N x H

    question_size = [question_pos[-1] - question_pos[0] + 1]
    question_max_size = max(question_size)

    decimal_num_emb = number_enc(batch_size, [nums_digit_single], [nums_pos_single], max_nums, num_padding_hidden)

    '2022-3-28 放进enecoder中'
    # encoder_outputs.shape = S * B * H
    # return : B x H; S x B x H; Q * B * 2H
    problem_outputs, inherit_root, encoder_outputs, word_feature = encoder(
        input_var, [input_length], input_length, num_emb, decimal_num_emb, word_emb, num_graph, num_attention_mask,
        tok_graph, tok_attention_mask, batch_size, question_size, [question_pos], question_max_size, sorted=True)

    all_nums_encoder_outputs = torch.matmul(n_extract_f_w, encoder_outputs.transpose(0, 1))

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_outputs.split(1, dim=0)]

    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], [], [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs  # [None, for ...]
            contexts = b.contexts
            goals = b.goals
            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            with torch.no_grad():
                contexts.append(current_context)
                goals.append(current_embeddings)

            # b x (op + O)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
            # 拿出前五个
            topv, topi = out_score.topk(beam_size)

            # 5 个 [1 x 1]; 5 个 [1 x 1]
            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:

                    generate_input = torch.LongTensor([out_token])
                    if device:
                        generate_input = generate_input.to(device)
                    #if len(contexts)>50:
                    #    print(' '.join(ori_datas))
                    #print('len(contexts), len(goals):',len(contexts), len(goals))
                    # 1 x H ; 1 x H
                    left_child, right_child, node_label = generate(current_embeddings, generate_input,
                                                                   current_context, contexts, goals, hir_feat)
                    # 1 x 1 x S x H
                    #l, r, child_information = fusion(encoder_outputs.permute(1, 0, 2).unsqueeze(0),
                    #                                 node.embedding,
                    #                                 word_feature.permute(1, 0, 2), right_child, left_child)
                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, contexts, goals, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out

