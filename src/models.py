# coding: utf-8
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel
from .GNN import Graph_Moulde, GraphConvolution, Node_Attn_head, GCN
from parameter import *


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


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


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        # input: H × B; S × B × H; B × max_len

        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len  # S × 1
        hidden = hidden.repeat(*repeat_dims)  # S * H x B
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill(seq_mask.bool(), -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S
        return attn_energies.unsqueeze(1)


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        # leaf B × 1 × 2H； B × O × H; B × max_N
        max_len = num_embeddings.size(1)  # O
        repeat_dims = [1] * hidden.dim()  # [1, 1, 1]
        repeat_dims[1] = max_len  # [1, O, 1]
        hidden = hidden.repeat(*repeat_dims)  # B × O x 2H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)  # B * O × 3H
        score = self.score(torch.tanh(self.attn(energy_in)))  # B * O × H -> B * O × 1
        score = score.squeeze(1)  # B * O
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill(num_mask.bool(), -1e12)
        return score


class _NumberEnc(nn.Module):
    def __init__(self, input_size, hidden_size, word_hidden, gru_layers, dropout=0.2):
        super(_NumberEnc, self).__init__()
        self.hidden_size = hidden_size
        self.c_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(input_size, hidden_size, gru_layers, dropout=dropout, bidirectional=True)
        self.FFC = nn.Linear(hidden_size, word_hidden)

    def forward(self, char_emb, flag=False, hidden=None):
        #  max_char_num_size * len(num_pos) * bert_hidden; len(num_pos) 个 数字的 list
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(char_emb, pade_hidden)
        nums_gru_final = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        outputs = self.c_dropout(self.FFC(nums_gru_final.squeeze(0)))
        if flag:
            print(outputs.shape)
        # len(num_pos), hidden
        return outputs


class NumberEnc(nn.Module):
    def __init__(self, input_size, hidden_size, word_hidden, gru_layers, dropout=0.2):
        super(NumberEnc, self).__init__()
        self.hidden_size = hidden_size
        self.c_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(input_size, hidden_size, gru_layers, dropout=dropout, bidirectional=True)
        self.FFC = nn.Linear(hidden_size, word_hidden)

    def forward(self, char_emb, flag=False, hidden=None):
        #  max_char_num_size * len(num_pos) * bert_hidden; len(num_pos) 个 数字的 list
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(char_emb, pade_hidden)
        nums_gru_final = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        outputs = self.c_dropout(self.FFC(nums_gru_final.squeeze(0)))
        if flag:
            print(outputs.shape)
        # len(num_pos), hidden
        return outputs


class Dense(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, bias=False, dropout=None):
        super(Dense, self).__init__()
        self.FC = nn.Linear(in_dim, out_dim, bias=bias)

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None

        self.dropout = None if dropout is None else nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.FC.weight)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)

        y = self.FC(x)

        if self.activation is not None:
            y = self.activation(y)

        return y


class GNNReasoning(nn.Module):
    def __init__(self, in_features, out_features, activation=False, bias=False, dropout=None):
        super(GNNReasoning, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        self.H_Dense = Dense(in_dim=in_features, out_dim=out_features, bias=bias)
        self.A_Dense = Dense(in_dim=in_features, out_dim=out_features, bias=bias)
        self.att = Dense(in_dim=in_features, out_dim=1, activation='tanh', bias=False)

        self.res_Dense = Dense(in_dim=in_features, out_dim=out_features, bias=bias)

        self.activation = nn.ReLU() if activation else None

        self.gc = GraphConvolution(hidden_size, hidden_size, bias=bias)

    def forward(self, inputs, adj):
        'inputs: B * S * H; adj: B * S * S'
        # H = self.H_Dense(inputs)

        A = self.A_Dense(inputs)
        att = self.att(A)
        alpha = torch.sigmoid(att)

        neighbor_h = alpha * A

        num_emb = self.gc(neighbor_h, adj) / adj.sum(dim=-1, keepdim=True)

        if self.activation:
            outputs = self.activation(self.res_Dense(inputs) + num_emb)
        else:
            outputs = self.res_Dense(inputs) + num_emb

        return outputs


class _numGAT(nn.Module):
    # 512, 8, 3个
    def __init__(self, in_features, out_features, activation=False, bias=False, dropout=None):
        super(_numGAT, self).__init__()

        self.numUpdate = GNNReasoning(in_features=in_features, out_features=out_features, activation=activation,
                                      bias=bias)

    def getNumEmb(self, word_emb, num_pos, way='multiple'):
        'input: (B * S * H)'

        batch_size = word_emb.size(0)
        max_len = word_emb.size(1)
        hidden_size = word_emb.size(2)

        if way == 'multiple':
            value = 0
        else:
            value = -1e12

        temp_1 = [1 for _ in range(hidden_size)]
        temp_0 = [0 for _ in range(hidden_size)]

        masked_index = []
        for b in range(batch_size):
            mask_one = [temp_1 for _ in range(max_len)]
            for i in num_pos[b]:
                mask_one[i] = temp_0
            masked_index.append(mask_one)

        if device:
            masked_index = torch.LongTensor(masked_index).to(device)

        nums_emb_masked = word_emb.masked_fill_(masked_index.bool(), value)

        return nums_emb_masked

    def forward(self, inputs, graphs, num_pos):
        'inputs: B * S * H; adj: B * 3 * S * S'
        adj = graphs.float()
        adj_list = [adj[:, 0, :], adj[:, 1, :], adj[:, 2, :]]

        outputs = self.numUpdate(inputs, adj_list[1])

        # num_emb = self.getNumEmb(outputs, num_pos)

        return outputs


class back_numGAT(nn.Module):
    # 512, 8, 3个
    def __init__(self, hidden_dim, heads):
        super(numGAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.heads_dim = int(hidden_dim / heads)

        # hete_multi_gat 需要一个 GCN 卷她 更新节点
        self.map_features_function = GraphConvolution(hidden_dim, hidden_dim, bias=True)

        # 然后需要 一个 Node_Attn_head(self.hidden_dim, self.heads_dim) 做 attention
        self.node_function = Node_Attn_head(self.hidden_dim, self.heads_dim)

    def forward(self, inputs, graphs, biases_batch, ):
        'inputs: B * S * H; adj: B * 3 * S * S'

        nbatches = inputs.size(0)
        gbatches = graphs.size(0)
        seq_len = graphs.size(2)

        adj = graphs.float()
        biases = biases_batch.float()
        # 3. split the adj_list to [nw, nn, ww] three type-subgraphs
        # and corroperansing biases
        adj_list = [adj[:, 0, :], adj[:, 1, :], adj[:, 2, :], adj[:, 3, :]]  # [b,s,s]
        bias_list = [biases[:, 0, :], biases[:, 1, :], biases[:, 2, :], biases[:, 3, :]]  # [b,s,s]

        # 1. 需要复制一个 Hete_multi_GAT(self.hidden_dim, self.heads, self.type_list, self.dropout)
        #    然后传入 gbatches, seq_len, inputs, adj_list, bias_list

        # 2. hete_multi_gat 需要一个 GCN 卷她

        embed_list = []

        map_feats = self.map_features_function(inputs, adj_list[3])

        attns = []
        for _ in range(self.heads):
            # input: [b,s,h]; [b,s,s]
            # ouput: [b,s,head_dim]
            attns.append(self.node_function(map_feats, bias_list[3]))

        h = torch.cat(attns, dim=-1)

        return h


class numGAT(nn.Module):
    # 512, 8, 3个
    def __init__(self, hidden_dim, heads, att_drop=0.0, coef_drop=0.0, activation=True):
        super(numGAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.heads_dim = int(hidden_dim / heads)

        # hete_multi_gat 需要一个 GCN 卷她 更新节点
        self.mapFunction = GraphConvolution(hidden_dim, hidden_dim, bias=True)

        self.att_drop = att_drop
        self.coef_drop = coef_drop

        self.att_dropout = nn.Dropout(att_drop)
        self.coef_dropout = nn.Dropout(coef_drop)

        self.conv1 = nn.Conv1d(hidden_dim, self.heads_dim, 1, bias=False)
        self.conv2_1 = nn.Conv1d(self.heads_dim, 1, 1, bias=False)
        self.conv2_2 = nn.Conv1d(self.heads_dim, 1, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.LeakyReLU()

    def forward(self, inputs, graphs, biases_batch):
        'inputs: B * S * H; adj: B * 3 * S * S'
        nbatches = inputs.size(0)
        gbatches = graphs.size(0)
        seq_len = graphs.size(2)

        adj = graphs.float()
        biases = biases_batch.float()
        # 3. split the adj_list to [nw, nn, ww] three type-subgraphs
        # and corroperansing biases
        adj_list = [adj[:, 0, :], adj[:, 1, :], adj[:, 2, :], adj[:, 3, :]]  # [b,s,s]
        bias_list = [biases[:, 0, :], biases[:, 1, :], biases[:, 2, :], biases[:, 3, :]]  # [b,s,s]

        # 1. 需要复制一个 Hete_multi_GAT(self.hidden_dim, self.heads, self.type_list, self.dropout)
        #    然后传入 gbatches, seq_len, inputs, adj_list, bias_list

        # 2. hete_multi_gat 需要一个 GCN 卷她

        map_feats = self.mapFunction(inputs, adj_list[3])

        attns = []
        for _ in range(self.heads):
            # input: [b,s,h]; [b,s,s]
            # ouput: [b,s,head_dim]
            seq = map_feats.float()
            if self.att_drop != 0.0:
                seq = self.att_dropout(inputs)
                seq = seq.float()
            seq_fts = self.conv1(seq.permute(0, 2, 1))  # [b, out_sz, seq_len]
            # print("seq_fts.shape:",seq_fts.shape)
            f_1 = self.conv2_1(seq_fts)  # [b, 1, seq_len]
            # print(f_1.shape)
            f_2 = self.conv2_1(seq_fts)  # [b, 1, seq_len]
            # print(f_2.shape)
            logits = f_1 + torch.transpose(f_2, 2, 1)  # [b, s, s]
            logits = self.leakyrelu(logits)
            # print("logis.shape:",logits.shape)
            coefs = self.softmax(logits + bias_list[3].float())  # [b,s,s]
            # print("coefs.shape:",coefs.shape)
            if self.coef_drop != 0.0:
                coefs = self.coef_dropout(coefs)
            if self.att_drop != 0.0:
                seq_fts = self.att_dropout(seq_fts)
            ret = torch.matmul(coefs, torch.transpose(seq_fts, 2, 1))  # [b,s,s] * [b,s,h] -> [b,s,h]
            # print(ret.shape)
            # ret = torch.transpose(ret, 2, 1) # [b,h,s]
            # print("ret.shape:",ret.shape)
            attns.append(self.activation(ret))

        h = torch.cat(attns, dim=-1)

        return h


class NumberStudent(nn.Module):
    def __init__(self, hidden_size):
        super(NumberStudent, self).__init__()

        self.hidden_size = hidden_size

        self.sort_fun_1 = nn.Linear(self.hidden_size, self.hidden_size)
        # 操作符，
        self.sort_fun_2 = nn.Linear(self.hidden_size, 5)

        self.compare_fun = nn.Linear(self.hidden_size, 1)

    def forward(self, word_emb):
        # B * S * H -->  B * S * H
        type_hidden = self.sort_fun_1(word_emb)
        #  B * S * H  -->  B * S * 5
        type_num = torch.sigmoid(self.sort_fun_2(type_hidden))  # B*S*5

        num_pair_score = self.compare_fun(word_emb).squeeze(2)  # B * S

        return type_num, num_pair_score


class EncoderChar(nn.Module):
    def __init__(self, bert_path, bert_size, hidden_size, get_word_and_sent=False):
        super(EncoderChar, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.model = BertModel.from_pretrained(bert_path)
        self.flag = 0
        for param in self.model.parameters():
            param.requires_grad = True
        self.trans_word = get_word_and_sent
        if self.trans_word:
            self.small_fc = nn.Linear(4 * bert_size, bert_size)
            self.md_softmax = MultiDimEncodingSentence(bert_size, hidden_size)

    def forward(self, inputs, char_len, matrix, out_all_hidden=False):
        input_all = self.tokenizer(inputs, return_tensors="pt", padding=True).to(device)
        # input就是 ' '.join(ori_seg)
        # return
        output = self.model(**input_all, output_hidden_states=out_all_hidden)
        if self.trans_word and out_all_hidden is False:
            sent_emb, word_emb = self.md_softmax(output.last_hidden_states, char_len, matrix)
            return word_emb, sent_emb
        elif self.trans_word and out_all_hidden:
            if self.flag == 0:
                print('   ***************************输出所有的潜在向量，并且做优化******************************   ')
                self.flag += 1
            all_hidden = output.hidden_states[1:]
            concatenate_pooling = torch.cat(
                (all_hidden[-1], all_hidden[-2], all_hidden[-3], all_hidden[-4]), -1
            )
            o = self.small_fc(concatenate_pooling)
            sent_emb, word_emb = self.md_softmax(o, char_len, matrix)
            return word_emb, sent_emb
        else:
            return output.last_hidden_states, output.pooler_output


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)


        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

        self.attn_bias_linear = nn.Linear(1, self.num_heads)

    def forward(self, q, k, v, attn_bias=None, attention_mask=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size

        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k) # [b, q_len, H] ->[b, q_len, H]->[b,q,head,d_k ]
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            # 相关的都是1
            attn_bias = attn_bias.unsqueeze(-1).permute(0, 3, 1, 2)
            # [b, h, q_len, k_len]
            attn_bias = attn_bias.repeat(1, self.num_heads, 1, 1)
            x += attn_bias

        if attention_mask is not None:
            # b x n x n x 1 --> b x 1 x n x n. 0 和 -1e9
            attention_mask = attention_mask.unsqueeze(-1).permute(0, 3, 1, 2)
            # b x head x n x n
            attention_mask = attention_mask.repeat(1, self.num_heads, 1, 1)
            x += attention_mask

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        # # [b, h, q_len, k_len] mat [b, h, v_len, d_v] -> [b, h, q_len, attn]
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, attn_bias=None):
        super(EncoderLayer, self).__init__()
        '512, 512, 0.1, 0.1, 6'
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = PositionwiseFeedForward(hidden_size, ffn_size, hidden_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, attention_mask=None):
        y = self.self_attention_norm(x)
        out = self.self_attention(y, y, y, attn_bias, attention_mask)
        y = self.self_attention_dropout(out)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


# normal position embedding
class Position_Embedding(nn.Module):
    def __init__(self, hidden_size):
        super(Position_Embedding, self).__init__()
        self.hidden_size = hidden_size
        self.scale = 1e-5

    def forward(self, x):  # input is encoded spans
        batch_size = x.size(0)
        seq_len = x.size(1)
        embedding = torch.rand(batch_size, seq_len, self.hidden_size) * self.scale

        return embedding.to(x.device)


class EncoderNum(nn.Module):
    def __init__(self, hidden_size):
        super(EncoderNum, self).__init__()
        self.hidden_size = hidden_size
        # numbers
        self.digit_value = nn.Embedding(10, int(self.hidden_size / 2))
        self.position_value = nn.Embedding(19, int(self.hidden_size / 2))


    def forward(self, batch_size, nums_digit_batch, nums_pos_batch, max_nums, num_padding_hidden):
        nums_hidden = []
        for b in range(batch_size):
            pro_nums = []
            for index in range(len(nums_digit_batch[b])):
                # print(nums_digit_batch[b][index])
                value_list = nums_digit_batch[b][index]
                value_emb = self.digit_value(torch.LongTensor(value_list).to(device))  # c x e

                pos_list = nums_pos_batch[b][index]
                pos_emb = self.position_value(torch.LongTensor(pos_list).to(device))  # c x E

                nums_pro_emb = torch.cat((value_emb, pos_emb), dim=-1)  # c x 2e
                nums_fin = torch.mean(nums_pro_emb, dim=0)  # 2e
                pro_nums.append(nums_fin)
            if len(pro_nums) < max_nums:
                pro_nums.extend([num_padding_hidden] * (max_nums - len(pro_nums)))
            pro_nums = torch.stack(pro_nums)
            nums_hidden.append(pro_nums)

        node_num = torch.stack(nums_hidden)  # b x n x h

        return node_num


class NumberReason(nn.Module):
    def __init__(self, indim, hiddim, outdim, dropout=0.3):
        super(NumberReason, self).__init__()
        self.input_dim = indim
        self.hidden_dim = hiddim
        self.out_dim = outdim

        self.dropout = dropout

        self.h = 2

        self.d_k = outdim//self.h

        #self.graph = self.clone(GCN(self.input_dim, self.hidden_dim, self.out_dim, dropout=self.dropout))

        self.graph = GCN(self.input_dim, self.hidden_dim, self.out_dim, dropout=self.dropout)

        self.norm = LayerNorm(self.out_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.out_dim)
        )

    def clone(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, emb, graph):
        # emb: b x n x h ;graph : B × N × N
        # adj = graph.float()

        temp = self.graph(emb, graph)
        num_fea = self.norm(temp) + emb
        output = self.feed_forward(num_fea) + num_fea

        return output


class EncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, type_list, gru_layers=2, hop_layers=2, multi_head=1,
                 num_dropout=0.5, word_dropout=0.5, all_dropout=0.5):
        super(EncoderSeq, self).__init__()
        # init: type_list = [ , , ]; hop_layers=2;

        # 2022-8-1 don't need .

        # self.input_size = input_size
        # self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        # self.gru_layers = gru_layers
        # self.num_dropout = num_dropout
        # self.word_dropout = word_dropout
        # self.all_dropout = all_dropout

        # self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        # self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(hidden_size, hidden_size, gru_layers, dropout=all_dropout, bidirectional=True)
        #
        # self.linea_p_1 = nn.Linear(2*hidden_size, hidden_size)
        # self.linea_h_1 = nn.Linear(hidden_size, hidden_size)
        # self.num_Dropout = nn.Dropout(num_dropout)
        # self.word_Dropout = nn.Dropout(word_dropout)
        # self.Dropout = nn.Dropout(all_dropout)
        # self.output_l = nn.Linear(2 * hidden_size, hidden_size)
        # self.norm = LayerNorm(2 * hidden_size)
        # self.FFC = PositionwiseFeedForward(2 * hidden_size, hidden_size, hidden_size, all_dropout)

        num_encoders = [EncoderLayer(hidden_size, hidden_size, num_dropout, num_dropout, 4)
                        for _ in range(2)]
        tok_encoders = [EncoderLayer(hidden_size, hidden_size, word_dropout, word_dropout, 4)
                        for _ in range(2)]

        logger.info(' [数字] 的EncoderLayer输入hidden_size {}, 用的dropout: {}, 注意力头数 {}, 层数 {}'.format(hidden_size, num_dropout, 4, 2))
        logger.info(' [单词] 的EncoderLayer输入hidden_size {}, 用的dropout: {}, 注意力头数 {}, 层数 {}'.format(hidden_size, word_dropout, 4, 2))
        self.num_encoder_layers = nn.ModuleList(num_encoders)
        self.tok_encoder_layers = nn.ModuleList(tok_encoders)

        self.num_encoders = NumberReason(hidden_size, hidden_size, hidden_size)

        self.pos_embed = Position_Embedding(hidden_size)
        self.input_dropout = nn.Dropout(all_dropout)
        logger.info(' 所有EncoderLayer用的 输入 dropout {}'.format(all_dropout))

        self.final_ln = nn.LayerNorm(hidden_size)

        self.self_att = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)

        self.FFC = PositionwiseFeedForward(hidden_size, hidden_size, hidden_size, all_dropout)

        self.dropout = nn.Dropout(all_dropout)
        self.concat_log_sem = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_log_sem_g = nn.Linear(hidden_size * 2, hidden_size)

        self.norm = LayerNorm(2 * hidden_size)
        self.feed_foward = PositionwiseFeedForward(2 * hidden_size, hidden_size, hidden_size, num_dropout)

    def get_question_emb(self, H, question_pos, batch_size, question_max_size):

        indices = list()
        sen_len = H.size(0)  # S
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
        all_outputs = H.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, H.size(2))
        all_question = all_embedding.index_select(0, indices)
        all_question = all_question.view(batch_size, question_max_size, hidden_size)
        return all_question.masked_fill(masked_index.bool(), 0.0)

    def forward(self, input_seqs, input_lengths, max_len, num_emb, decimal_num_emb, word_emb, num_graph,
                num_attention_mask, tok_graph, tok_attention_mask, batch_size, question_lengths, question_pos,
                question_max_size, hidden=None, sorted=False):
        ' word_emb: (b * s * h); batch_graph: b 个 k 个  s x s;  '
        ' num_emb : B x M_nums x H; num_graph: B x M_nums x M_nums; num_graph: B x M_nums x M_nums '

        # decimal = self.dropout(decimal_num_emb)
        # num_embeddings = self.norm(torch.cat((decimal, num_emb), dim=-1))
        #
        # num_encoded = self.feed_foward(num_embeddings) + num_emb
        #
        # n_encoded = num_encoded + self.pos_embed(num_encoded)
        # node = self.input_dropout(n_encoded)
        # num_layer_outputs = []
        # for enc_layer in self.num_encoder_layers:
        #     attn_bias = num_graph
        #     # attention_mask: b x max_node x max_node
        #     node = enc_layer(node, attn_bias, num_attention_mask)
        #     num_layer_outputs.append(node)
        # node_num = num_layer_outputs[-1] + num_layer_outputs[-2]
        # # B x N x H -->  N x B x H
        # node_num = self.final_ln(node_num).transpose(0, 1)

        decimal = self.dropout(decimal_num_emb)

        num_output = self.num_encoders(decimal, num_graph).transpose(0, 1)  # b x n x h --> N x B x H

        w_encoded = word_emb + self.pos_embed(word_emb)
        node = self.input_dropout(w_encoded)
        token_layer_outputs = []
        for enc_layer in self.tok_encoder_layers:
            attn_bias = tok_graph
            # attention_mask: b x max_node x max_node
            node = enc_layer(node, attn_bias, tok_attention_mask)
            token_layer_outputs.append(node)
        node_token = token_layer_outputs[-1] + token_layer_outputs[-2]
        # B x S x H -> S x B x H
        node_token = self.final_ln(node_token).transpose(0, 1)

        emb = torch.cat((node_token, num_output), dim=0)

        # (S + N)  x  B  x H
        token_num_outputs = self.self_att(emb)

        token_emb = token_num_outputs[:max_len, :, :] # [S x B x H]

        H = self.FFC(token_emb) + word_emb.transpose(0, 1) # S x B x H

        # 获得 question向量 B x Q x H
        all_question = self.get_question_emb(H, question_pos, batch_size, question_max_size)

        packed = torch.nn.utils.rnn.pack_padded_sequence(all_question, question_lengths, batch_first=True,
                                                         enforce_sorted=sorted)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        # Q * B * 2H
        question_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)
        # B * H
        problem_output = question_outputs[-1, :, :self.hidden_size] + question_outputs[0, :, self.hidden_size:]

        return problem_output, problem_output, H, question_outputs


class QuestionEnc(nn.Module):
    def __init__(self, hidden_size, n_layers=2, dropout=0.5):
        super(QuestionEnc, self).__init__()
        self.hidden_size = hidden_size
        self.question_gru_pade = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.self_att = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)

        # self.content_gru_pade = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

        # self.linea_p_1 = nn.Linear(2 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.norm = LayerNorm(2 * hidden_size)
        self.FFC = PositionwiseFeedForward(2 * hidden_size, hidden_size, hidden_size, dropout)


    def get_question_context_emb(self, H, question_pos, batch_size, question_max_size):
        indices = list()
        sen_len = H.size(0)  # S
        masked_index = []
        temp_1 = [1 for _ in range(hidden_size)]
        temp_0 = [0 for _ in range(hidden_size)]
        # print('batch_size,', batch_size)
        # print('qustion_pos,', question_pos)
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
        all_outputs = H.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, H.size(2))
        all_question = all_embedding.index_select(0, indices)
        all_question = all_question.view(batch_size, question_max_size, hidden_size)
        return all_question.masked_fill(masked_index.bool(), 0.0)

    def forward(self, encoder_outputs, input_length, question, num, question_lengths, word2num, question_pos,
                batch_size, question_max_size, context_lengths, context_pos, context_max_size, hidden=None, sorted=False):
        # input :
        # encoder_outputs [S * B * H]
        # question:  [Q * B * H]
        # num : [ B * N * H ]
        # word2num: [B * S * N]

        # N
        max_num_len = num.size(1)
        # [ B * N * H ] ---> [ N * B * H ]
        num = num.transpose(0, 1)
        # Q
        max_q_len = question.size(0)

        # print(question.shape, num.shape)

        emb = torch.cat((question, num), dim=0)
        H = self.self_att(emb)

        question_emb = H[:max_q_len, :, :]  # [Q * B * H]
        num_emb = H[max_q_len:, :, :]  # [ N * B * H ]

        # B x S x N, B x N x H ---> B x S x H
        all_num_emb = torch.matmul(word2num, num_emb.transpose(0, 1))
        # [S * B * H] --> [B * S * H]; cat ( [B * S * H] + [B * S * H] )
        emb = self.norm(torch.cat((encoder_outputs.transpose(0, 1), all_num_emb), dim=-1))
        # emb : B * S * 2H --->  B * S * H ; transpose(0,1) +  [S * B * H]  ; H : S * B * H
        H = self.FFC(emb).transpose(0, 1) + encoder_outputs
        # H = self.Dropout(F.relu(self.linea_p_1(emb))).transpose(0, 1)

        # print('all_question')
        # B x Q x H -> Q x S x H
        all_question = self.get_question_context_emb(H, question_pos, batch_size, question_max_size).permute(1, 0, 2)
        # print('all_question.shape, ', all_question.shape)
        # print('all_context')
        # B x C x H -> C x B x H
        # all_context = self.get_question_context_emb(H, context_pos, batch_size, context_max_size).permute(1, 0, 2)
        # print('all_context.shape: ', all_context.shape)
        # print('context_lengths: ', context_lengths)

        # emb = torch.cat((word_emb, num_emb), dim=-1)
        # H = self.Dropout(F.relu(self.linea_p_1(emb)))
        packed = torch.nn.utils.rnn.pack_padded_sequence(all_question, question_lengths, enforce_sorted=sorted)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.question_gru_pade(packed, pade_hidden)
        # Q * B * 2H
        question_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)
        # B * H
        problem_output = question_outputs[-1, :, :self.hidden_size] + question_outputs[0, :, self.hidden_size:]

        # context
        # packed = torch.nn.utils.rnn.pack_padded_sequence(all_context, context_lengths, enforce_sorted=sorted)  #
        # pade_hidden = hidden
        # pade_outputs, pade_hidden = self.content_gru_pade(packed, pade_hidden)
        # # 这个 pade_outputs 才是 T
        # pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)  # C x B x 2h; _
        # # [B, H]
        # content = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]

        # # 这是一种信息融合
        # p = self.dropout(problem_output)
        # c = self.dropout(content)
        # g = torch.tanh(self.concat_pc(torch.cat((p, c), 1)))
        # t = torch.sigmoid(self.concat_g_pc(torch.cat((p, c), 1)))
        # inherit_root = g * t

        return problem_output, _, H, question_outputs


class Review(nn.Module):
    def __init__(self, vec_size, kernels='3@5', activation='Relu', review_filter=512):
        super(Review, self).__init__()
        self.vec_size = vec_size
        self.n_grams = kernels.split('@')
        self.filter = int(vec_size / len(self.n_grams))
        print('\n 卷积size: ', self.n_grams)
        self.temp_con = []
        for g in self.n_grams:
            self.temp_con.append(
                nn.Conv2d(in_channels=1,
                          out_channels=self.filter,
                          kernel_size=(int(g), self.vec_size),
                          stride=1,
                          padding=(int(int(g) / 2), 0),
                          dilation=1)
            )
        self.dif_conv2d = nn.ModuleList(self.temp_con)
        self.ffc1 = nn.Linear(self.filter * len(self.n_grams), int(self.filter / 2))

        self.ffc2 = nn.Linear(int(self.filter / 2), 1)

        self.activation = nn.ReLU() if activation else None

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 这里的x输入必须是 B x S x H
        # 然后，必须转变为： B x 1 x S x H
        # 2022-11-20 也可以处理 1 x 1 x S x H --> 1 x S x H; 1 x H
        x = x.view(x.shape[0], 1, -1, self.vec_size)
        con_outputs = []
        for c in self.dif_conv2d:
            out = self.activation(c(x))
            # print('out.shape：', out.shape)
            out = out.squeeze(3)
            # print('2 out.shape：', out.shape)
            out = out.permute(0, 2, 1)
            # print('3 out.shape：', out.shape)
            con_outputs.append(out)
            # print('-'*100)
        outputs = torch.cat(con_outputs, dim=-1)
        # print('outputs：', outputs.shape)
        h = self.activation(self.ffc1(outputs))
        # print('h：', h.shape)
        att = self.ffc2(h).repeat(1, 1, outputs.shape[2])
        # print('att：', att.shape)
        summed_output = torch.sum(outputs * att, dim=1, keepdim=False)
        # print('summed_output:', summed_output.shape)
        # B x S x H; B x H
        return outputs, summed_output


class FusionReview(nn.Module):
    def __init__(self, vec_size, kernels='5', activation='Relu', sigmoid='sig', softmax='soft',
                 semantic_use_door=False, dropout=0.5, review_filter=512):
        super(FusionReview, self).__init__()
        # use_door 是 是否使用 门网络生成。
        self.vec_size = vec_size
        #self.h_size = h_size
        self.kernels = kernels

        print('\n 传入的 kenels : ', self.kernels)

        self.semantic_use_door = semantic_use_door

        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU() if activation else None
        self.sigmoid = nn.Sigmoid() if sigmoid else None

        self.review = Review(self.vec_size, self.kernels, activation)

        if self.semantic_use_door:
            self.concat_semantic = nn.Linear(self.vec_size * 2, self.vec_size)
            self.concat_s_semantic = nn.Linear(self.vec_size * 2, self.vec_size)
        else:
            self.semantic_fc = nn.Linear(4 * self.vec_size, self.vec_size)
        self.semantic_score = nn.Linear(self.vec_size, 1)

        self.word_ffc1 = nn.Linear(2 * self.vec_size, self.vec_size)
        self.row_softmax = nn.Softmax(dim=2) if softmax else None
        self.col_softmax = nn.Softmax(dim=1) if softmax else None

        self.softmax = nn.Softmax(dim=1)

        self.rnn = nn.GRU(input_size=self.vec_size, hidden_size=self.vec_size, batch_first=True, bidirectional=True)
        self.word_ffc2 = nn.Linear(3 * self.vec_size, 1)

        # self.concat_l = nn.Linear(self.vec_size * 2, self.vec_size)
        # self.concat_lg = nn.Linear(self.vec_size * 2, self.vec_size)
        # self.concat_r = nn.Linear(self.vec_size * 2, self.vec_size)
        # self.concat_rg = nn.Linear(self.vec_size * 2, self.vec_size)

        self.norm = LayerNorm(2 * hidden_size)
        self.left_FFC = PositionwiseFeedForward(2 * hidden_size, hidden_size, hidden_size, dropout)
        self.right_FFC = PositionwiseFeedForward(2 * hidden_size, hidden_size, hidden_size, dropout)

        # self.merge = nn.Linear(self.vec_size * 2 + self.vec_size, self.vec_size)
        # self.merge_g = nn.Linear(self.vec_size * 2 + self.vec_size, self.vec_size)

    def forward(self, x, parent_feature, question_feature, r, l):
        # input： 1 x 1 x S x H; 1 x H; 1 x B x H; 1 x H; 1 x H
        # print('x.shape : ', x.shape)
        # print('p.shape : ', p.shape)
        # print('r.l.shape : ', r.shape, l.shape)
        # print('word_feature: ', word_feature.shape)

        parent_information = self.dropout(parent_feature)
        question_feature = self.dropout(question_feature)
        x = self.dropout(x)

        # B x S x H; B x H
        # 实际： 1 x 1 x S x H; 1 x H
        word_conv_matrix, re_knowledge = self.review(x)
        # print(word_conv_matrix.shape, re_knowledge.shape)

        if self.semantic_use_door:
            g = torch.tanh(self.concat_semantic(torch.cat((parent_information, re_knowledge), 1)))
            t = torch.sigmoid(self.concat_s_semantic(torch.cat((parent_information, re_knowledge), 1)))
            semantic = g * t
        else:
            # 1 x 4h;
            semantic = torch.cat(
                [parent_information, re_knowledge, parent_information * re_knowledge,
                 torch.abs(parent_information - re_knowledge)], dim=1
            )
            # 1 x 1
            semantic = self.relu(self.semantic_fc(semantic))
        # 1 x 1
        semantic_score = self.sigmoid(self.semantic_score(semantic))
        # print('semantic_score: ', semantic_score.shape, semantic_score)

        # word-level: 1 x Q x 2H --> 1 x Q x H
        question_vec = self.word_ffc1(question_feature)
        # 1 x C x H bmm 1 x H x S_c ---> 1 x C x S_c
        score_matrix = torch.tanh(torch.bmm(question_vec, word_conv_matrix.permute(0, 2, 1)))
        # print()
        row_S = self.row_softmax(score_matrix)
        col_S = self.col_softmax(score_matrix)  # importance measurement
        # 1 x C
        word_important = torch.sum(col_S, dim=2, keepdim=False)
        # 1 x C x S_c bmm 1 x S_c x H --> 1 x C x H
        intermediate = torch.bmm(row_S, word_conv_matrix)
        # print('intermediate":', intermediate.shape)
        ' 双向的 '
        intermediate, _ = self.rnn(intermediate) # 1 x C x 2H
        # print('intermediate":', intermediate.shape)
        # intermediate = self.word_ffc2(intermediate)
        # 1 x C x 3H --> 1 x C x H
        output, _ = torch.max(torch.cat([intermediate, question_vec], dim=2), dim=1)
        word_score = self.sigmoid(self.word_ffc2(output))

        # print('word_score: ', word_score.shape, word_score)

        # 融合
        # 1 x 1 x h bmm 1 x S x H = 1 x 1 x S ---> 1 x S
        alpha = torch.bmm(parent_information.unsqueeze(1), word_conv_matrix.permute(0, 2, 1)).squeeze(1)
        # print('alpha:, ', alpha.shape)
        # 1 x S
        beta = self.softmax(alpha)
        # print('bera : ', beta, beta.shape)
        # 1 x H
        inter_word_sum = torch.sum(beta.unsqueeze(2).repeat(1, 1, word_conv_matrix.shape[2]) * word_conv_matrix, dim=1)
        child_information = parent_information + (word_score / (semantic_score + word_score)) * re_knowledge + \
                            (semantic_score / (semantic_score + word_score)) * inter_word_sum

        child = self.dropout(child_information)

        # 改成这样就行了？
        l_emb = self.norm(torch.cat((l, child), dim=-1))
        r_emb = self.norm(torch.cat((r, child), dim=-1))
        # emb : B * S * 2H --->  B * S * H ; transpose(0,1) +  [S * B * H]  ; H : S * B * H
        l = self.left_FFC(l_emb) + l
        r = self.right_FFC(r_emb) + r

        # g_l = torch.tanh(self.concat_l(torch.cat((l, parent_information), 1)))
        # t_l = torch.sigmoid(self.concat_lg(torch.cat((l, parent_information), 1)))
        # l = g_l * t_l

        # g_r = torch.tanh(self.concat_r(torch.cat((r, parent_information), 1)))
        # t_r = torch.sigmoid(self.concat_rg(torch.cat((r, parent_information), 1)))
        # r = g_r * t_r

        # sub_child = torch.tanh(self.merge(torch.cat((l, r, child), 1)))
        # sub_child_g = torch.sigmoid(self.merge_g(torch.cat((l, r, child), 1)))
        # child_information = sub_child * sub_child_g

        # print('l.shape, r.shape', l.shape, r.shape)

        return l, r, child_information


class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding
    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        # 这是 常数
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)
        # 常数向量
        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        '1. node_stacks 存储的是 [根向量] '
        current_embeddings = []

        '2. 最近的 [根向量] '
        ' node_stack 在后续中，当母节点是 op, 直接插入的 generate 模块的 左右 子节点'
        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)  # padding_hidden.shape = 1 × H
            else:
                current_node = st[-1]  # 每次只要 node_stack 最后一个的emb, 最后一个就是h_l
                current_embeddings.append(current_node.embedding)  # current_node.embedding = 1 × H； B 个.

        '3. 最近的 [根向量] 生成 [goal vector]'
        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            # 1 × H， 1 × H
            if l is None:
                ' 没有左子树 只有 goal_vec, 用门 生成 左子 根'
                # pre 左子树
                c = self.dropout(c)  # c : 1 × H
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                # pre 右子树
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)
        ' [goal vector] ，当要直接用的时候，就下面加个dropout，千万不要用已经dropout的在进行处理'
        current_node = torch.stack(current_node_temp)  # [goal vector] q, B × 1 × h
        current_embeddings = self.dropout(current_node)
        # B × H; S × B × H; B × max_len
        '4. [goal vector] 生成 环境向量 权重 alpha'
        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)  # return: B × 1 × S
        # B × 1 × S bmm B * S * H；
        # x.bmm(y) 要求 x.dim(0)=y.dim(0) 必须相同, x.dim(2)=y.dim(1)
        '5. alpha 从 h 中获取 [环境向量c]'
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x H

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE
        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x con_num x H
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x H， 数字

        'cat([goal vector], [环境向量c])， '
        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)  # 生成的叶子结点向量，cat(q,c) 后续用来 预测 op还是num --> B × 2H

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        ' score(cat([goal vector], [环境向量c]) + e(y|P)), 数字的选择 考虑 cat([goal vector], [环境向量c]) 结合 数字'
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)  # B × O
        ' 2022-11-10 融入 我们说的知识背景，应该是在 [goal vector], [环境向量c] 这两个下手 '

        # num_score = nn.functional.softmax(num_score, 1)
        ' op 的生成 不考虑 数字向量进去 只考虑 cat([goal vector], [环境向量c]) 这两个向量'
        op = self.ops(leaf_input)  # B × 2H --->  B × op

        # return p_leaf, num_score, op, current_embeddings, current_attn
        ' score(cat([goal vector], [环境向量c]) + e(y|P)); B × op; [goal vector]; [环境向量c]； 数字emb'
        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, pretrain_emb=None, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # 目前来说, 常数 和 op都可以用 bert搞
        self.embeddings = nn.Embedding(op_nums, embedding_size)
        if pretrain_emb is not None:
            print(' ---------- 使用预训练的op 向量 ------------------')
            self.embeddings.weight.data.copy_(torch.from_numpy(pretrain_emb))
        else:
            print(' ---------- 自 定 义 op 向  量 ------------------')

        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

        self.hire_soft = nn.Softmax(dim=-1)
        self.cont_hire_att = nn.Parameter(torch.zeros(self.hidden_size, 1))
        self.goal_hire_att = nn.Parameter(torch.zeros(self.hidden_size, 1))

        self.norm = LayerNorm(hidden_size)

        self.cont_feed_foward = PositionwiseFeedForward(hidden_size, hidden_size, hidden_size, dropout)
        self.goal_feed_foward = PositionwiseFeedForward(hidden_size, hidden_size, hidden_size, dropout)

        self.reset_parameters()

    def reset_parameters(self):
        '''
        initialization params
        :return:
        '''
        nn.init.xavier_uniform_(self.cont_hire_att)
        nn.init.xavier_uniform_(self.goal_hire_att)

    def forward(self, node_embedding, node_label, current_context, contexts, goals, hir_feat):
        # [goal vector] b x 1 x h； B 个 label, 除了5个op， 其他全部是0； [环境向量c] B x 1 x H
        node_label_ = self.embeddings(node_label)  # B x E, B 个 label, 除了5个op， 其他全部是0 转换的 embbeding
        node_label = self.em_dropout(node_label_)

        cur_deep = len(contexts)
        #print('deep', cur_deep)
        # deep个 b x 1 x H
        cont_att_list = []
        goal_att_list = []
        for i in range(cur_deep):
            cont_att_list.append(torch.matmul((contexts[i] + hir_feat[i]), self.cont_hire_att))  # b x 1 x 1
            goal_att_list.append(torch.matmul((goals[i] + hir_feat[i]), self.goal_hire_att))

        # print(cont_att_list[0].shape, goal_att_list[0].shape)

        all_cont = torch.cat(cont_att_list, dim=-1)  # b x 1 x deep
        all_goal = torch.cat(goal_att_list, dim=-1)

        # print(all_cont.shape, all_goal.shape)

        cont_att = self.hire_soft(all_cont)  # b x 1 x deep
        goal_att = self.hire_soft(all_goal)

        final_cont_list = []  # deep 个 b x 1 x h
        final_goal_list = []
        for i in range(cur_deep):
            # b x 1 x 1 -> b x 1 x 1 * b x 1 x h
            final_cont_list.append(cont_att[:, :, i].unsqueeze(-1) * contexts[i])
            final_goal_list.append(goal_att[:, :, i].unsqueeze(-1) * goals[i])

        context_embedding = torch.sum(torch.stack(final_cont_list), dim=0)  # deep 个 b x 1 x h -> b x b
        goal_embedding = torch.sum(torch.stack(final_goal_list), dim=0)  # b x 1 x h

        node_embedding = node_embedding.squeeze(1)  # [goal vector] b x h
        current_context = current_context.squeeze(1)  # [环境向量c] B x H
        context_embedding = context_embedding.squeeze(1)  # [环境向量c] B x H
        goal_embedding = goal_embedding.squeeze(1)  # [环境向量c] B x H
        #node_embedding = self.em_dropout(node_embedding)
        #current_context = self.em_dropout(current_context)

        cont_feature = self.norm(context_embedding) + current_context
        goal_feature = self.norm(goal_embedding) + node_embedding

        att_cont_feature = self.cont_feed_foward(cont_feature) + cont_feature
        att_goal_feature = self.goal_feed_foward(goal_feature) + goal_feature

        l_child = torch.tanh(self.generate_l(torch.cat((att_goal_feature, att_cont_feature, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((att_goal_feature, att_cont_feature, node_label), 1)))
        l_child = l_child * l_child_g

        r_child = torch.tanh(self.generate_r(torch.cat((att_goal_feature, att_cont_feature, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((att_goal_feature, att_cont_feature, node_label), 1)))
        r_child = r_child * r_child_g
        # h_l; h_r ; B x E, B 个 label, 除了5个op， 其他全部是0 转换的 embbeding
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        # op; 左子树； 最近节点;
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


class BayesDropout(nn.Module):
    def __init__(self, dropout):
        super(BayesDropout, self).__init__()
        self.dropout = nn.Dropout(dropout=dropout)

    def forward(self, x):
        return self.dropout(x)


def Mask(inputs, seq_len=None, way='multiple'):
    # seq_len is list [len, len, ...]
    # inputs is tensor , B * S * H
    if seq_len is None:
        return inputs

    if way == 'multiple':
        value = 0
    else:
        value = -1e12

    batch_size = inputs.size(0)
    max_len = inputs.size(1)
    hidden_size = inputs.size(2)

    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]

    for b in range(batch_size):
        mask_one = [temp_1 for _ in range(max_len)]
        for i in range(seq_len[b]):
            mask_one[i] = temp_0
        masked_index.append(mask_one)

    if torch.cuda.is_available():
        masked_index = torch.LongTensor(masked_index).to(device)
    inputs_masked = inputs.masked_fill_(masked_index.bool(), value)

    return inputs_masked


class DropDense(nn.Module):
    def __init__(self, in_dim, out_dim, activation=False, bias=True, dropout=None):
        super(DropDense, self).__init__()
        self.FC = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = nn.ReLU() if activation else None
        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(self, x, x_len=None, way='multiple'):
        if self.dropout is not None:
            x = self.dropout(x)

        y = self.FC(x)

        if self.activation is not None:
            y = self.activation(y)

        if x_len is not None:
            y = Mask(y, x_len, way)

        return y


class MultiDimEncodingSentence(nn.Module):
    def __init__(self, in_dim, out_dim=None, dropout=0.2):
        # 768, 512
        super(MultiDimEncodingSentence, self).__init__()
        if out_dim is None:
            self.out_dim = in_dim
        else:
            self.out_dim = out_dim

        self.input_h = None

        self.DDense_x = DropDense(in_dim, self.out_dim, bias=False)

        self.DDense_s2t_1 = DropDense(in_dim, in_dim, activation=True, bias=True)
        self.DDense_s2t_2_1 = DropDense(in_dim, 1, bias=True)
        self.DDense_s2t_2_2 = DropDense(in_dim, self.out_dim, bias=True)

        self.dropout_H = nn.Dropout(dropout)
        self.dropout_A = nn.Dropout(dropout)
        self.dropout_A_1 = nn.Dropout(dropout)
        self.dropout_A_2 = nn.Dropout(dropout)

    def forward(self, x, x_len, seg_mask=None):
        # x 是 bert 输入 B * C_S * H
        x_H = self.dropout_H(x)
        # in_dim, self.out_dim, bias=False, ac=False, dropout=None;
        # B * C_S * 512
        H = self.DDense_x(x_H, x_len, 'multiple')  # 1. MLP(H) + zero_Mask

        x_A = self.dropout_A(x)
        # in_dim, in_dim, ac=True, bias=True, dropout=None
        # B * C_S * 768
        A = self.DDense_s2t_1(x_A)  # 2. MLP(H) + Relu + Bias
        A = self.dropout_A_1(A)
        # B * C_S * 1
        A_1 = self.DDense_s2t_2_1(A, x_len, 'addition')

        x_A_2 = self.dropout_A_2(x)
        # B * C_s * 512
        A_2 = self.DDense_s2t_2_2(x_A_2, x_len, 'addition')  # 4. MLP + no_zeros_mask + bias
        # B * C_S * 1 +  B * C_s * 512  - B * C_s * 1 = B * C_s * 512
        A = A_1 + A_2 - torch.mean(A_2, dim=-1, keepdims=True)  # 向量的 alpha

        A1 = F.softmax(A, dim=1)  # Attention --> 对每个 Chara 做 attention --> (B * C_s * 512)
        sent_emb = torch.sum(A1 * H, dim=1)  # 把句子中 所有的 char 向量加 得到一个 来表示 句子 B * 512
        A = torch.exp(torch.clip(A, min=-1e12, max=10))  # B * C_s * 512 元素控制在-1e12, 10

        AH = A * H  # B * C_s * 512 ;  B * C_S * 512 ->  B * C_S * 512 attention后的Hid
        #  seg_mask 是 B * W_s * C_S  ; B * C_S * 512  ->  B * W_s * 512
        # 除 attention B * W_s * 512
        word_emb = torch.matmul(seg_mask, AH) / (torch.matmul(seg_mask, A) + 1e-12)

        return sent_emb, word_emb
