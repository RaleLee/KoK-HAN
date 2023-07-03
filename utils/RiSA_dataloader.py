import json
from copy import deepcopy

import torch
import ast
import numpy as np
import platform

from utils.config import *
import torch.utils.data as data

data_loc = "data/RiSAWOZ/"


def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info, src_word2id, trg_word2id):
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['context_arr'])
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        context_arr = self.data_info['context_arr'][index]
        context_arr = self.preprocess(context_arr, self.src_word2id, trg=False)
        response = self.data_info['response'][index]
        response = self.preprocess(response, self.trg_word2id)
        ptr_index = torch.Tensor(self.data_info['ptr_index'][index])
        conv_arr = self.data_info['conv_arr'][index]
        conv_arr = self.preprocess(conv_arr, self.src_word2id, trg=False)
        kb_arr = self.data_info['kb_arr'][index]
        kb_arr = self.preprocess(kb_arr, self.src_word2id, trg=False)
        sketch_response = self.data_info['sketch_response'][index]
        sketch_response = self.preprocess(sketch_response, self.trg_word2id)

        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][index]

        # additional plain information
        data_info['context_arr_plain'] = self.data_info['context_arr'][index]
        data_info['response_plain'] = self.data_info['response'][index]
        data_info['kb_arr_plain'] = self.data_info['kb_arr'][index]
        data_info['sketch_response_plain'] = self.data_info['sketch_response'][index]

        return data_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        story = torch.Tensor(story)
        return story

    def collate_fn(self, data):
        def merge(sequences, story_dim):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            if story_dim:
                padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    if len(seq) != 0:
                        padded_seqs[i, :end, :] = seq[:end]
            else:
                padded_seqs = torch.ones(len(sequences), max_len).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        def merge_index(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        data.sort(key=lambda x: len(x['conv_arr']), reverse=True)
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences
        context_arr, context_arr_lengths = merge(item_info['context_arr'], True)
        response, response_lengths = merge(item_info['response'], False)
        ptr_index, _ = merge(item_info['ptr_index'], False)
        conv_arr, conv_arr_lengths = merge(item_info['conv_arr'], True)
        sketch_response, _ = merge(item_info['sketch_response'], False)
        kb_arr, kb_arr_lengths = merge(item_info['kb_arr'], True)

        context_arr = _cuda(context_arr.contiguous())
        response = _cuda(response.contiguous())
        ptr_index = _cuda(ptr_index.contiguous())
        conv_arr = _cuda(conv_arr.transpose(0, 1).contiguous())
        sketch_response = _cuda(sketch_response.contiguous())
        if len(list(kb_arr.size())) > 1:
            kb_arr = _cuda(kb_arr.transpose(0, 1).contiguous())

        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        for i, length in enumerate(context_arr_lengths):
            assert length - (conv_arr_lengths[i] + kb_arr_lengths[i]) <= 4
            assert length == data_info['node_num'][i] + 1

        data_info['context_arr_lengths'] = context_arr_lengths
        data_info['response_lengths'] = response_lengths
        data_info['conv_arr_lengths'] = conv_arr_lengths
        data_info['kb_arr_lengths'] = kb_arr_lengths

        return data_info


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def get_domain_span(node_list, domains, domain2idx):
    span = {}
    start, end, dialogue_start = 0, 0, 0
    sub_span_list = []
    domain_num = 0
    domain = -1
    for node in node_list:
        if node[0][1].startswith('$'):
            end = node[1] - 1
            dialogue_start = node[1]
            sub_span_list.append((start, end))
            break
        if len(node) == 3 and isinstance(node[0], str):
            domain_num += 1
            span[domain2idx[domains[node[1]]]] = [(node[2])]
            continue
        domain_idx = domain2idx[domains[node[1]]]
        if domain == -1:
            domain = node[1]
            start = node[2]
            continue
        if domain_idx == domain:
            continue
        else:
            end = node[2] - 1
            sub_span_list.append((start, end))
            domain = domain2idx[domains[node[1]]]
            start = node[2]
    for s in sub_span_list:
        domain_idx = domain2idx[domains[node_list[s[0]][1]]]
        span[domain_idx].append(s)
    span[0] = [(dialogue_start, len(node_list)-1)]
    return span


def read_langs(file_name, graph_list, domain2idx, idx2domain):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, domain_dict = [], [], [], [], {}
    node_list = []
    max_resp_len = 0
    with open(file_name, encoding="utf-8") as fin:
        cnt_lin, sample_counter, node_idx = 1, 0, 0
        for line in fin:
            line = line.strip()
            if line:
                if line.startswith("#"):
                    flag = 0
                    line = line.split()
                    for a in line:
                        if a == "#":
                            continue
                        if a.startswith("0"):
                            domain_idx = a
                            flag = 1
                            continue
                        if flag == 1:
                            domain_dict[int(domain_idx)] = a
                            flag = 0
                            node_list.append([a, int(domain_idx), node_idx])
                            node_idx += 1
                            context_arr.append([a] + ["PAD"]*(MEM_TOKEN_SIZE-1))
                            continue
                        dialog_id = a
                    continue
                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u, r, u_seged, r_seged, gold_ent = line.split('\t')
                    gen_u = generate_triplet(u_seged, "$u", str(nid))
                    context_arr += gen_u
                    conv_arr += gen_u
                    for tri in gen_u:
                        node_list.append([tri, node_idx])
                        node_idx += 1

                    gold_ent = ast.literal_eval(gold_ent)

                    ptr_index = []
                    for key in r_seged.split():
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in gold_ent)]
                        if index:
                            index = max(index)
                        else:
                            index = len(context_arr)
                        ptr_index.append(index)

                    sketch_response = generate_template(r_seged, gold_ent, kb_arr, domain_dict,node_list)
                    span = get_domain_span(node_list, domain_dict, domain2idx)
                    for i, node in enumerate(node_list):
                        if isinstance(node_list[i][0], str):
                            assert node_list[i][0] == context_arr[i][0]
                        else:
                            assert node_list[i][0][0] == context_arr[i][0]

                    data_detail = {
                        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # $$$$ is NULL token
                        'response': r_seged,
                        'sketch_response': sketch_response,
                        'ptr_index': ptr_index + [len(context_arr)],
                        'ent_index': gold_ent,
                        'conv_arr': list(conv_arr),
                        'kb_arr': list(kb_arr),
                        'ID': dialog_id,
                        'domains': domain_dict,
                        'domain_num': len(domain_dict),
                        'node_list': deepcopy(node_list + [['$$$$'] * MEM_TOKEN_SIZE]),
                        'node_num': len(node_list),
                        'graph': graph_list[sample_counter],
                        'domain_span': span
                    }
                    data.append(data_detail)
                    gen_r = generate_triplet(r_seged, "$s", str(nid))
                    context_arr += gen_r
                    conv_arr += gen_r
                    for tri in gen_r:
                        node_list.append([tri, node_idx])
                        node_idx += 1
                    if max_resp_len < len(r_seged.split()):
                        max_resp_len = len(r_seged.split())
                    sample_counter += 1
                else:
                    r = line
                    kb_info = generate_triplet(r, "", str(nid))
                    context_arr = context_arr + kb_info
                    kb_arr += kb_info
                    kb_info.extend([int(nid), node_idx])
                    node_list.append(kb_info)
                    node_idx += 1
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr, node_list, domain_dict = [], [], [], [], {}
                node_idx = 0

    return data, max_resp_len


def generate_triplet(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn' + str(time), 'word' + str(idx)] + ["PAD"] * (MEM_TOKEN_SIZE - 4)
            sent_new.append(temp)
    else:
        token_list, object_list = [], []
        for i in range(len(sent_token)):
            if i == 0 or i == 1:
                token_list.append(sent_token[i])
            else:
                object_list.append(sent_token[i])
        token_list.append(" ".join(object_list))
        sent_token = token_list[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(token_list))
        sent_new.append(sent_token)
    return sent_new


def generate_template(sentence, gold_entity, kb_arr, domain_dict, node_list):
    dup_list = ["名称", "区域", "是否地铁直达", "电话号码", "地址", "评分", "价位", "制片国家/地区", "类型", "年代", "主演", "导演", "片名", "主演名单", "豆瓣评分",
                "出发地", "目的地", "日期", "到达时间", "票价", "天气"]
    domain = ""
    sketch_response = []
    if not gold_entity:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            if word not in gold_entity:
                sketch_response.append(word)
            else:
                ent_type = None
                for kb_item in kb_arr:
                    if word == kb_item[0]:
                        for k in node_list:
                            if k[0] == kb_item:
                                domain = domain_dict[k[1]]
                        ent_type = kb_item[1]
                        break
                if ent_type in dup_list:
                    ent_type = domain + '_' + ent_type
                sketch_response.append('@' + ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response


def read_graph(file_path):
    node_num = 0
    edge1_src, edge1_trg, edge2_src, edge2_trg, edge3_src, edge3_trg = [], [], [], [], [], []
    graph_list = []
    with open(file_path, encoding="utf-8") as fin:
        flag = 0
        for line in fin:
            line = line.strip()
            if flag == 0:
                node_num = str(line)
                flag += 1
            elif flag == 1:
                l = line.split()
                for edge in l:
                    node = edge.split('-')
                    edge1_src.append(int(node[0]))
                    edge1_trg.append(int(node[1]))
                flag += 1
            elif flag == 2:
                l = line.split()
                for edge in l:
                    node = edge.split('-')
                    edge2_src.append(int(node[0]))
                    edge2_trg.append(int(node[1]))
                flag += 1
            elif flag == 3:
                l = line.split()
                for edge in l:
                    node = edge.split('-')
                    edge3_src.append(int(node[0]))
                    edge3_trg.append(int(node[1]))
                flag += 1
            elif flag == 4:
                graph_list.append(
                    [node_num, np.array(edge1_src), np.array(edge1_trg), np.array(edge2_src), np.array(edge2_trg),
                     np.array(edge3_src), np.array(edge3_trg)])
                node_num, edge1_src, edge1_trg, edge2_src, edge2_trg, edge3_src, edge3_trg = 0, [], [], [], [], [], []
                flag = 0
    return graph_list


def get_seq(pairs, lang, batch_size, type):
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []

    for pair in pairs:
        for k in pair.keys():
            data_info[k].append(pair[k])
        if type:
            lang.index_words(pair['context_arr'])
            lang.index_words(pair['response'], trg=True)
            lang.index_words(pair['sketch_response'], trg=True)
    dataset = Dataset(data_info, lang.word2index, lang.word2index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              collate_fn=dataset.collate_fn)
    return data_loader


def prepare_data(fast_test=False, v_train=False):
    train_graph = read_graph(data_loc + "train_graph.txt")
    dev_graph = read_graph(data_loc + "dev_graph.txt")
    test_graph = read_graph(data_loc + "test_graph.txt")
    domain2idx = {"对话历史": 0, "旅游景点": 1, "餐厅": 2, "酒店": 3, "火车": 4, "飞机": 5, "天气": 6, "电影": 7, "电视剧": 8}
    idx2domain = {0: "对话历史", 1: "旅游景点", 2: "餐厅", 3: "酒店", 4: "火车", 5: "飞机", 6: "天气", 7: "电影", 8: "电视剧"}
    domain_list = ["对话历史", "旅游景点", "餐厅", "酒店", "火车", "飞机", "天气", "电影", "电视剧"]
    global_entity = get_global_entities(data_loc + "ontology_2.0.json", domain_list)

    if not fast_test:
        train_data, train_max_len = read_langs(data_loc + "train.txt", train_graph, domain2idx, idx2domain)
    else:
        train_data, train_max_len = read_langs(data_loc + "dev.txt", train_graph, domain2idx, idx2domain)
    if not v_train:
        dev_data, dev_max_len = read_langs(data_loc + "dev.txt", dev_graph, domain2idx, idx2domain)
        test_data, test_max_len = read_langs(data_loc + "test.txt", test_graph, domain2idx, idx2domain)
    else:
        dev_data, dev_max_len = read_langs(data_loc + "train.txt", train_graph, domain2idx, idx2domain)
        test_data, test_max_len = read_langs(data_loc + "train.txt", train_graph, domain2idx, idx2domain)

    max_rep_len = max(train_max_len, dev_max_len, test_max_len)

    assert len(train_data) == len(train_graph)
    assert len(dev_data) == len(dev_graph)
    assert len(test_data) == len(test_graph)

    lang = Lang()

    train_dataloader = get_seq(train_data, lang, args["batch"], True)
    dev_dataloader = get_seq(dev_data, lang, args["batch"], False)
    test_dataloader = get_seq(test_data, lang, args["batch"], False)

    return train_dataloader, dev_dataloader, test_dataloader, lang, max_rep_len, domain2idx, idx2domain, global_entity


def get_global_entities(path, domain_list: list):
    print("Reading global entity...")
    with open(path, encoding='utf-8') as f:
        global_entity = json.load(f)
        global_entity_list = []
        for key in global_entity.keys():
            if key not in domain_list:
                continue
            key_dict = global_entity[key]
            for attribute in key_dict.keys():
                global_entity_list += [it for it in key_dict[attribute]]
        global_entity_list = list(set(global_entity_list))
    return global_entity_list


if __name__ == '__main__':
    prepare_data()
