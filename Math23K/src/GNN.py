# coding: utf-8
import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        #nfeat为feature大小也就是idx_features_labels[:, 1:-1]第二列到倒数第二列，nhid自己设定
        #for _ in range(nheads)多头注意力机制，这里面nheads为8
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        #创建8个多头注意力机制模块

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)  # 第二层(最后一层)的attention layer
        #nhid * nheads输入维度8*8，输出维度当时分的类，这个数据集里面是7个
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每层attention拼接
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))   # 第二层的attention layer
        return F.log_softmax(x, dim=1)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # 初始化in_features行，out_features列的权重矩阵
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # concat(V,NeigV)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # 初始化α，大小为两个out_features拼接起来

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), hW.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # 每一个节点和所有节点，特征。(Vall, Vall, feature)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # 之前计算的是一个节点和所有节点的attention，其实需要的是连接的节点的attention系数
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # 将邻接矩阵中小于0的变成负无穷
        attention = F.softmax(attention, dim=1)  # 按行求softmax。 sum(axis=1) === 1
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # 聚合邻居函数

        if self.concat:
            return F.elu(h_prime)  # elu-激活函数
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # 复制 (b, s, h) --> (2 * b , s, h)

        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
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
    def __init__(self, d_model, d_ff,d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        # stdv = 1. / math.sqrt(self.weight.size(1))

    def forward(self, inputs, adj):
        #
        temp = torch.matmul(inputs, self.weight)
        output = torch.matmul(adj, temp)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, in_feat_dim, nhid, out_feat_dim, dropout):
        super(GCN, self).__init__()
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        - adjacency matrix (batch_size, K, K)
        ## Returns:
        - gcn_enhance_feature (batch_size, K, out_feat_dim)
        '''
        self.gc1 = GraphConvolution(in_feat_dim, nhid)
        self.gc2 = GraphConvolution(nhid, out_feat_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        # b,s,h
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GAT(nn.Module):
    def __init__(self, in_feat_dim, nhid, out_feat_dim, att_heads, dropout, concat=True):
        super(GAT, self).__init__()
        self.in_features = in_feat_dim
        self.hid_features = nhid
        self.out_feat_dim = out_feat_dim
        self.alpha = alpha
        self.att_heads = att_heads
        self.att_dropout = nn.Dropout(dropout)
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_feat_dim, nhid)))  # 建立都是0的矩阵，大小为（输入维度，输出维度）
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.han_feed = nn.Linear(nhid, out_feat_dim)

    def forward(self, x, adj):
        # print('x.shape', x.shape)
        h = torch.matmul(x, self.W)
        # print('h.shape', h.shape)
        N = h.size()[1]
        batch_size = h.size()[0]
        # a_input = torch.cat([x.repeat(1, 1, N).view(batch_size, N * N, -1), x.repeat(1, N, 1)], dim=-1).view(
        #     batch_size, N, -1, 2 * self.hid_features)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        a_input_1 = torch.matmul(x, self.a[:self.hid_features])
        a_input_2 = torch.matmul(x, self.a[self.hid_features:])

        a_input = a_input_1.repeat(1, 1, N).view(batch_size, N, N) + a_input_2.repeat(1, N, 1).view(batch_size, N, N)
        e = self.leakyrelu(a_input)

        zero_vec = -9e15 * torch.ones_like(e)
        # print(zero_vec.shape)
        attention = torch.where(adj > 0, e, zero_vec)
        # print(attention.shape)
        attention = F.softmax(attention, dim=1)
        attention = self.att_dropout(attention)
        # print(attention.shape)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            hid = F.elu(h_prime)
        else:
            hid = h_prime
        return self.han_feed(hid)


class Node_Attn_head(nn.Module):
    # 512, 64,
    def __init__(self, in_channel, out_sz, in_drop=0.0, coef_drop=0.0, activation=nn.LeakyReLU(),
                 residual=False, return_coef=False):
        super(Node_Attn_head, self).__init__()
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.return_coef = return_coef
        # print(in_channel, out_sz)
        self.conv1 = nn.Conv1d(in_channel, out_sz, 1, bias=False)

        # node_attention
        self.conv2_1 = nn.Conv1d(out_sz, 1, 1, bias=False)
        self.conv2_2 = nn.Conv1d(out_sz, 1, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.in_dropout = nn.Dropout(in_drop)
        self.coef_dropout = nn.Dropout(coef_drop)
        self.activation = activation

    def forward(self, x, bias_mat):
        seq = x.float()
        if self.in_drop != 0.0:
            seq = self.in_dropout(x)
            seq = seq.float()
        seq_fts = self.conv1(seq.permute(0,2,1)) # [b, out_sz, seq_len]
        # print("seq_fts.shape:",seq_fts.shape)
        f_1 = self.conv2_1(seq_fts) # [b, 1, seq_len]
        # print(f_1.shape)
        f_2 = self.conv2_1(seq_fts) # [b, 1, seq_len]
        # print(f_2.shape)
        logits = f_1 + torch.transpose(f_2, 2, 1) # [b, s, s]
        logits = self.leakyrelu(logits)
        # print("logis.shape:",logits.shape)
        coefs = self.softmax(logits + bias_mat.float()) # [b,s,s]
        # print("coefs.shape:",coefs.shape)
        if self.coef_drop != 0.0:
            coefs = self.coef_dropout(coefs)
        if self.in_drop != 0.0:
            seq_fts = self.in_dropout(seq_fts)
        ret = torch.matmul(coefs, torch.transpose(seq_fts, 2, 1)) # [b,s,s] * [b,s,h] -> [b,s,h]
        # print(ret.shape)
        # ret = torch.transpose(ret, 2, 1) # [b,h,s]
        # print("ret.shape:",ret.shape)
        if self.return_coef:
            return self.activation(ret), coefs
        else:
            return self.activation(ret)  # activation


class TypeAttLayer(nn.Module):
    # self.heads * self.heads_dim, 2 * self.hidden_dim, time_major=False, return_alphas=False
    def __init__(self, inputs, attention_size, time_major=False, return_alphas=False):
        super(TypeAttLayer, self).__init__()
        self.hidden_size = inputs
        self.return_alphas = return_alphas
        self.time_major = time_major
        # attention_size 考虑是 hidden_size的两倍。
        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_size, attention_size))
        self.b_omega = nn.Parameter(torch.Tensor(attention_size))
        self.u_omega = nn.Parameter(torch.Tensor(attention_size, 1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_omega)
        nn.init.zeros_(self.b_omega)
        nn.init.xavier_uniform_(self.u_omega)

    def forward(self, x):
        # [b*s,type * 1 , head_dim * head] -> [b*s,type * 1 , 2 * self.hidden_dim]
        v = self.tanh(torch.matmul(x, self.w_omega) + self.b_omega)
        # [b*s,type * 1 , 2 * self.hidden_dim] -> [b*s, type * 1 , 1]
        vu = torch.matmul(v, self.u_omega)

        # [b*s, type * 1 , 1]
        alphas = self.softmax(vu)
        #print('alphas:', alphas.shape)

        #  [b*s,type * 1 , head_dim * head] * [b*s, type * 1 , 1] ->  [b*s, head_dim * head]
        output = torch.sum(x * alphas, dim=1)
        #print('output, ', output.shape)

        # [b*s, head_dim * head]
        if not self.return_alphas:
            return output, None
        else:
            return output, alphas


class Hete_multi_GAT(nn.Module):
    # 512, 8, 3个
    def __init__(self, hidden_dim, heads, dropout):
        super(Hete_multi_GAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.heads_dim = int(hidden_dim/heads)

        self.map_features_function = GraphConvolution(hidden_dim, hidden_dim, bias=True)

        self.node_att_layers = Node_Attn_head(self.hidden_dim, self.heads_dim)

    def forward(self, inputs, adj_list, bias_list):
        embed_list = []
        # i = [0,1,2]
        map_feats = self.map_features_function(inputs, adj_list)

        attns = []
        for _ in range(self.heads):
            # input: [b,s,h]; [b,s,s]
            # ouput: [b,s,head_dim]
            attns.append(self.node_att_layers(map_feats, bias_list))
        # [b, s, head_dim * head]

        h = torch.cat(attns, dim=-1) + inputs

        return h


class Graph_Moulde(nn.Module):
    # hidden_size, hidden_size, hidden_size, 2, 1, type_list
    def __init__(self, indim, hidden, outdim, layers, heads, type,
                 node_attention=True, type_attention=True, dropout=0.3):
        super(Graph_Moulde, self).__init__()
        ' layers = 2, heads = 1 , type_list = [ , , ]  '
        self.input_dim = indim
        self.hidden_dim = hidden
        self.out_dim = outdim
        self.layers = layers
        print('self.layer = ', self.layers)
        self.heads = heads
        self.heads_dim = self.hidden_dim/heads

        if type == 'logic':
            self.type = 1
        else:
            self.type = 0

        self.node_attention = node_attention
        self.type_attention = type_attention

        self.dropout = dropout

        self.hire_soft = nn.Softmax(dim=-1)
        self.hire_att = nn.Parameter(torch.zeros(self.hidden_dim, 1))

        #self.FFN = PositionwiseFeedForward(indim, hidden, outdim, dropout)
        self.norm = LayerNorm(hidden)
        self.feed_foward = PositionwiseFeedForward(indim, hidden, outdim, dropout)

        if node_attention and type_attention:
            self.node_type_Layer = self._make_Hete_layers()

    def _make_Hete_layers(self):
        layers = []
        for _ in range(self.layers):
            layers.append(Hete_multi_GAT(self.hidden_dim, self.heads, self.dropout))

        return nn.Sequential(*list(m for m in layers))


    def reset_parameters(self):
        '''
        initialization params
        :return:
        '''
        nn.init.xavier_uniform_(self.hire_att)


    def forward(self, inputs, graphs, biases_batch):
        # s x b x h; b 个 k 个 s x s; b 个 k 个 s x s
        nbatches = inputs.size(0)
        gbatches = graphs.size(0)
        seq_len = graphs.size(2)
        #print('batch_size:', gbatches)
        #print('seq_len:', seq_len)
        if nbatches != gbatches:
            inputs = inputs.transpose(0, 1)

        # 2.adj -> float
        adj = graphs.float()
        biases = biases_batch.float()
        # 3. split the adj_list to [nw, nn, ww] three type-subgraphs
        # and corroperansing biases

        adj_list = adj[:, self.type, :] # 1 个 [b,s,s]
        bias_list = biases[:, self.type, :] # 1 个 [b,s,s]
        '''
        embed_list = []
        # i = [0,1,2]
        for i in range(len(self.type_list)):
            # 4. features project according the type
            # input: [b,s,h] , [b,s,s]
            # output: [b,s,h]
            map_feats = self.map_features_function[i](inputs, adj_list[i])

            attns = []
            for _ in range(self.heads):
                # input: [b,s,h]; [b,s,s]
                # ouput: [b,s,head_dim]
                attns.append(self.att_layers[i](map_feats, bias_list[i]))
            # [b,s,head_dim * head]
            h = torch.cat(attns, dim=-1)

            # temp_hidden_size = h.size(-1)
            # type 个[b*s, 1, head_dim * head]
            embed_list.append(h.reshape(-1, 1, self.heads * self.heads_dim))
        # [b*s,type * 1 , head_dim * head]
        multi_embed = torch.cat(embed_list, dim=1)
        # [b*s, head_dim * head]
        temp_embed, att_val = self.type_AttLayer(multi_embed)
        final_embed = temp_embed.reshape(gbatches, seq_len, self.hidden_dim) # [b,s,h]
        '''
        # [layers]个[b,s,h]
        final_embed_list = []
        final_embed_list.append(inputs)
        for i in range(self.layers):
            # layer 层, 每层 b x s x h
            inputs = self.node_type_Layer[i](inputs, adj_list, bias_list)
            final_embed_list.append(inputs)

        g_feature = self.norm(final_embed_list[-1]) + final_embed_list[0]
        # print('g_feature')
        # print(g_feature.shape)

        graph_encode_features = self.feed_foward(g_feature) + g_feature

        return graph_encode_features