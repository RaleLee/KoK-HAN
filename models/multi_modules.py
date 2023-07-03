import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from utils.RiSA_dataloader import _cuda

from dgl.nn.pytorch import GATConv


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, attention_out_size, dropout, n_layers=1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.W = nn.Linear(2 * hidden_size + attention_out_size, hidden_size)
        self.attention = SelfAttention(
            hidden_size, hidden_size, attention_out_size, dropout_rate=dropout
        )
        self.sent_attention = UnFlatSelfAttention(
            hidden_size * 2 + attention_out_size, dropout
        )

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long())
        embedded = embedded.view(input_seqs.size() + (embedded.size(-1),))
        embedded = torch.sum(embedded, 2).squeeze(2)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        attention_in = embedded.transpose(0, 1)
        attention_out_hidden = self.attention(attention_in)
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        outputs = outputs.transpose(0, 1)
        cat_hidden = torch.cat([outputs, attention_out_hidden], dim=-1)
        sent_represent = self.sent_attention(cat_hidden, input_lengths)
        cat_hidden, sent_represent = self.W(cat_hidden), self.W(sent_represent)
        return cat_hidden, sent_represent


class GraphEK(nn.Module):
    def __init__(self, vocab, embedding_dim, dropout,
                 num_edges, graph_hidden_size, graph_out_size, num_heads,
                 domain2index: dict):
        super(GraphEK, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
        self.HAN = HAN(num_edges, embedding_dim, graph_hidden_size, graph_out_size, num_heads, self.dropout)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        for idx in domain2index.values():
            node_li = nn.Linear(embedding_dim, embedding_dim)
            self.add_module("node_{}".format(idx), node_li)
        self.node_lis = AttrProxy(self, "node_")

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens, domain_num):
        # full_memory:(bs, node_num, hidden), kb_len:(bs), hiddens:(bs, dh_node_num, hidden)
        for bi in range(full_memory.size(0)):
            start, end = kb_len[bi] + domain_num[bi], kb_len[bi] + domain_num[bi] + conv_len[bi]
            full_memory[bi, start:end, :] = full_memory[bi, start:end, :] + hiddens[bi, :conv_len[bi], :]
        return full_memory

    def node_linear(self, feat_in, span: dict):
        """
        not for batch version
        domain_span: dict. dict[key=domain_idx] = [value=tuple()]
        batch*maxlen*hidden
        kb+history - 100
        node linear 100*128 domain
        domain node
        """
        feat_out = feat_in.clone()
        for idx, sp in span.items():
            node_list = []
            if len(sp) > 1:
                node_list.append(feat_in[sp[0]: sp[0] + 1])
                for kk in sp[1:]:
                    bi, ed = kk
                    node_list.append(feat_in[bi: ed + 1])
                l_in = torch.cat(node_list)
                out = self.node_lis[idx](l_in)
                feat_out[sp[0]] = out[0]
                out_bi = 1
                for kk in sp[1:]:
                    bi, ed = kk
                    feat_out[bi: ed + 1] = out[out_bi: out_bi + ed - bi + 1]
                    out_bi += ed - bi + 1
            else:
                bi, ed = sp[0]
                l_in = feat_in[bi: ed + 1]
                feat_out[bi: ed + 1] = self.node_lis[idx](l_in)
        return feat_out

    def cross_embed(self, feat_in, seq_len: list, look_up: list):
        """
        use seq_len to split the features, use look_up to check if it need to sum
        """
        # feat_in size: batch_size*max_len*4*embedding_size
        sum_up, batch_size = [], len(seq_len)

        for i in range(batch_size):
            sentence = self.embedding(feat_in[i].long())
            sum_up_line, sp_tuple, line_look_up = [], torch.split(sentence, seq_len[i], dim=0), look_up[i]
            len_tuple = len(sp_tuple)
            assert len_tuple == len(line_look_up)
            for k in range(len_tuple):
                if line_look_up[k]:
                    sum_up_line.append(torch.sum(sp_tuple[k], keepdim=True, dim=0))
                else:
                    sum_up_line.append(sp_tuple[k])
            sum_up.append(torch.cat(sum_up_line, dim=0))
        max_len = max([sup.shape[0] for sup in sum_up])
        for i in range(batch_size):
            zeros = _cuda(torch.zeros((max_len, 4, self.embedding_dim)))
            zeros[:sum_up[i].shape[0], :, :] = sum_up[i]
            sum_up[i] = zeros.unsqueeze(0)
        return torch.cat(sum_up, dim=0)

    def load_memory(self, story, kb_len, conv_len, dh_outputs,
                    graphs, node_nums: list, domain_span: list, domain_num: list):
        batch_size = len(story)
        story_size = story.size()
        embed_A = self.embedding(story.contiguous().view(story_size[0], -1))
        embed_A = embed_A.view(story_size + (embed_A.size(-1),))
        embed_A = torch.sum(embed_A, 2).squeeze(2)
        embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len, dh_outputs, domain_num)
        features = self.dropout_layer(embed_A)
        linear_out_features = []
        self.graph_out_features = features.clone()
        for i in range(batch_size):
            try:
                linear_out_features.append(self.node_linear(features[i][:node_nums[i]], domain_span[i]))
            except TypeError:
                print(domain_span[i])
                print(node_nums[i])
        graph_out = self.HAN(graphs, torch.cat(linear_out_features))
        begin = 0
        for i in range(batch_size):
            self.graph_out_features[i][:node_nums[i]] = graph_out[begin:begin + node_nums[i]]
            begin += node_nums[i]
        return

    def forward(self, query_vector):
        u = [query_vector]
        m_A = self.graph_out_features
        if len(list(u[-1].size())) == 1:
            u[-1] = u[-1].unsqueeze(0)
        u_temp = u[-1].unsqueeze(1).expand_as(m_A)
        prob_logits = torch.sum(m_A * u_temp, 2)
        prob_soft = self.softmax(prob_logits)
        return prob_soft, prob_logits


class Decoder(nn.Module):
    def __init__(self, shared_emb, lang, embedding_dim, dropout):
        super(Decoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.C = shared_emb
        self.softmax = nn.Softmax(dim=1)
        self.sketch_rnn = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, EKGraph, story_size, story_lengths, copy_list, encode_hidden, target_batches, max_target_length,
                batch_size, use_teacher_forcing, get_decoded_words):
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length, batch_size, story_size))
        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        memory_mask_for_step = _cuda(torch.ones(batch_size, story_size))
        decoded_fine, decoded_coarse = [], []
        hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)

        for t in range(max_target_length):
            embed_q = self.dropout_layer(self.C(decoder_input))  # b * e
            if len(embed_q.size()) == 1:
                embed_q = embed_q.unsqueeze(0)
            _, hidden = self.sketch_rnn(embed_q.unsqueeze(0), hidden)
            query_vector = hidden[0]
            p_vocab = self.attend_vocab(self.C.weight, hidden.squeeze(0))
            all_decoder_outputs_vocab[t] = p_vocab
            _, topvi = p_vocab.data.topk(1)

            prob_soft, prob_logits = EKGraph(query_vector)
            all_decoder_outputs_ptr[t] = prob_logits
            if use_teacher_forcing:
                decoder_input = target_batches[:, t]
            else:
                decoder_input = topvi.squeeze()
            if get_decoded_words:
                search_len = min(5, min(story_lengths))
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = prob_soft.data.topk(search_len)
                temp_f, temp_c = [], []

                for bi in range(batch_size):
                    token = topvi[bi].item()  # topvi[:,0][bi].item()
                    temp_c.append(self.lang.index2word[token])
                    if '@' in self.lang.index2word[token]:
                        cw = 'UNK'
                        for i in range(search_len):
                            if toppi[:, i][bi] < story_lengths[bi] - 1:
                                try:
                                    cw = copy_list[bi][toppi[:, i][bi].item()]
                                except IndexError:
                                    print(str(len(copy_list)) + " " + str(bi))
                                    print(str(len(copy_list[bi])))
                                    print(str(toppi[:, i][bi].item()))
                                break
                        temp_f.append(cw)
                        if args['record']:
                            memory_mask_for_step[bi, toppi[:, i][bi].item()] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])

                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)
        return all_decoder_outputs_vocab, all_decoder_outputs_ptr, decoded_fine, decoded_coarse

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        # scores = F.softmax(scores_, dim=1)
        return scores_


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_edges : number of homogeneous graphs generated from the different types of edges.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    """

    def __init__(self, num_edges, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        self.gat_layers = nn.ModuleList()
        for i in range(num_edges):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads, dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_edges = num_edges

    def forward(self, homo_graphs, h):
        """
        Inputs
        ------
        homo_graphs : list[DGLGraph]
            List of batched graphs
        h : tensor
            Input features(batched)

        Outputs
        -------
        tensor
        The output feature
        """
        semantic_embeddings = []
        for i, graph in enumerate(homo_graphs):
            semantic_embeddings.append(self.gat_layers[i](graph, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

        return self.semantic_attention(semantic_embeddings)


class HAN(nn.Module):
    def __init__(self, num_edges, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_edges, in_size, hidden_size, num_heads[0], dropout))
        for layer in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_edges, hidden_size * num_heads[layer - 1],
                                        hidden_size, num_heads[layer], dropout))
        self.Linear = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, homo_graphs, h):
        for gnn in self.layers:
            h = gnn(homo_graphs, h)

        return self.Linear(h)


class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """ The forward propagation of attention.
        Here we require the first dimension of input key
        and value are equal.
        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        # something wrong in QKVAttention score
        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ) / math.sqrt(self.__hidden_dim), dim=-1)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )

        return attention_x


class UnFlatSelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context


class AttrProxy(object):

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
