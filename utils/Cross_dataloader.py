import json
from copy import deepcopy, copy

import torch
import ast
import numpy as np

import torch.utils.data as data

from utils.config import *


data_loc = "data/CrossWOZ/"


def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x


class Dataset(data.Dataset):

    def __init__(self, data_info, src_word2id, trg_word2id):
        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['context_arr'])
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id

    def __getitem__(self, index):
        """
        Returns one data pair (source and target).
        """
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
        """
        Converts words to ids.
        """
        list_flag = 0
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    if isinstance(word, list):
                        temp = [word2id[w] if w in word2id else UNK_token for w in word]
                        story[i].append(temp)
                        list_flag = 1
                        continue
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        story = torch.Tensor(story) if list_flag == 0 else story
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

        # sort a list by sequence length (descending order) to use pack_padded_sequence
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
        kb_total_arr_lengths = [len(seq) for seq in item_info['kb_total_arr']]

        context_arr = _cuda(context_arr.contiguous())
        response = _cuda(response.contiguous())
        ptr_index = _cuda(ptr_index.contiguous())
        conv_arr = _cuda(conv_arr.transpose(0, 1).contiguous())
        sketch_response = _cuda(sketch_response.contiguous())

        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        for i, length in enumerate(data_info["node_num"]):
            assert length == (conv_arr_lengths[i] + kb_total_arr_lengths[i] + len(data_info["domains"][i]))
            assert length == context_arr_lengths[i] - 1

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
                    if isinstance(word, list):
                        for w in word:
                            self.index_word(w)
                    else:
                        self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def get_domain_span(node_list, domains, domain2idx):
    span = {}
    start, end, dialogue_start, dialogue_end = 0, 0, 0, 0
    sub_span_list = []
    domain_num = 0
    domain = -1
    for node in node_list:
        if (not isinstance(node[0][1], list)) and node[0][1].startswith('$'):
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
            domain = domain_idx
            start = node[2]
            continue
        if domain_idx == domain:
            continue
        else:
            end = node[2] - 1
            sub_span_list.append((start, end))
            domain = domain2idx[domains[node[1]]]
            start = node[2]
    for node in node_list[dialogue_start:]:
        if (not isinstance(node[0][1], list)) and node[0][1].startswith('$'):
            continue
        else:
            dialogue_end = node[2] - 1
            span[0] = [(dialogue_start, dialogue_end)]
            break
    domain = -1
    for node in node_list[dialogue_end+1:]:
        domain_idx = domain2idx[domains[node[1]]]
        if domain == -1:
            domain = domain_idx
            start = node[2]
            continue
        if domain_idx == domain:
            continue
        else:
            end = node[2] - 1
            sub_span_list.append((start, end))
            domain = domain2idx[domains[node[1]]]
            start = node[2]
    sub_span_list.append((start, node_list[-1][2]))
    for s in sub_span_list:
        domain_idx = domain2idx[domains[node_list[s[0]][1]]]
        span[domain_idx].append(s)
    return span


def read_langs(file_name, graph_list, domain2idx, idx2domain):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, domain_dict, kb_object_span = [], [], [], [], {}, []
    list_object_node = []
    segment_len, combine_or_not = [], []
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
                            context_arr.append([a] + ["PAD"] * (MEM_TOKEN_SIZE - 1))
                            continue
                        dialog_id = a
                    continue
                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u_seged, r_seged, gold_ent = line.split('\t')
                    gen_u = generate_triplet(u_seged, "$u", str(nid))
                    context_arr += gen_u
                    conv_arr += gen_u
                    for tri in gen_u:
                        node_list.append([tri, node_idx])
                        node_idx += 1
                    turn_context_arr = deepcopy(context_arr)
                    turn_kb_arr = deepcopy(kb_arr)
                    tem, idx = [], node_idx
                    for n in list_object_node:
                        t = copy(n)
                        t.append(idx)
                        tem.append(t)
                        idx += 1
                        turn_context_arr.append(n[0])
                        turn_kb_arr.append(n[0])
                    turn_node_list = copy(node_list) + tem

                    gold_ent = ast.literal_eval(gold_ent)

                    ptr_index = []
                    for key in r_seged.split():
                        index = [loc for loc, val in enumerate(turn_context_arr) if (val[0] == key and key in gold_ent)]
                        if index:
                            index = max(index)
                        else:
                            index = len(turn_context_arr)
                        ptr_index.append(index)

                    sketch_response = generate_template(r_seged, gold_ent, turn_kb_arr, domain_dict, turn_node_list)
                    span = get_domain_span(turn_node_list, domain_dict, domain2idx)
                    for i, node in enumerate(turn_node_list):
                        if isinstance(turn_node_list[i][0], str):
                            assert turn_node_list[i][0] == turn_context_arr[i][0]
                        else:
                            assert turn_node_list[i][0][0] == turn_context_arr[i][0]

                    data_detail = {
                        'context_arr': turn_context_arr + [['$$$$'] * MEM_TOKEN_SIZE],  # $$$$ is NULL token
                        'response': r_seged,
                        'sketch_response': sketch_response,
                        'ptr_index': ptr_index + [len(turn_context_arr)],
                        'ent_index': gold_ent,
                        'conv_arr': list(conv_arr),
                        'kb_arr': list(kb_arr),
                        'ID': dialog_id,
                        'domains': domain_dict,
                        'domain_num': len(domain_dict),
                        'node_list': turn_node_list + [['$$$$'] * MEM_TOKEN_SIZE],
                        'node_num': len(turn_node_list),
                        'graph': graph_list[sample_counter],
                        'domain_span': span,
                        'kb_total_arr': turn_kb_arr
                    }
                    assert data_detail["conv_arr"][0] == data_detail["context_arr"][len(data_detail["kb_arr"])+len(data_detail["domains"])]
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
                    kb_len = len(kb_info)
                    if kb_len > 1:
                        for i, no in enumerate(kb_info):
                            if i == 0:
                                temp = [no]
                                temp.extend([int(nid), node_idx])
                                node_list.append(temp)
                                context_arr += [no]
                                kb_arr += [no]
                            else:
                                temp = [no, int(nid)]
                                list_object_node.append(temp)
                    else:
                        context_arr = context_arr + kb_info
                        kb_arr += kb_info
                        kb_info.extend([int(nid), node_idx])
                        node_list.append(kb_info)
                    node_idx += 1
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr, node_list, domain_dict, kb_object_span = [], [], [], [], {}, []
                list_object_node = []
                segment_len, combine_or_not = [], []
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
        list_attributes = 0
        for i in range(len(sent_token)):
            if i == 0 or i == 1:
                token_list.append(sent_token[i])
            else:
                object_list.append(sent_token[i])
                if '[' in sent_token[i]:
                    list_attributes = 1
        object_ = "_".join(object_list) if list_attributes == 0 else " ".join(object_list)
        object_ = eval(object_) if '[' in object_ else object_
        if isinstance(object_, list):
            if len(object_) == 0:
                token_list.append("None")
                sent_token = token_list[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(token_list))
                sent_new.append(sent_token)
            else:
                sent_token = token_list[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(token_list))
                sent_new.append(sent_token)
            for o in object_:
                temp = copy(token_list)
                temp.append(o.strip().replace(" ", "_"))
                sent_token = temp[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(temp))
                sent_new.append(sent_token)
        else:
            token_list.append(object_)
            sent_token = token_list[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(token_list))
            sent_new.append(sent_token)
    return sent_new


def generate_template(sentence, gold_entity, kb_arr, domain_dict, node_list):
    duplicated_set = {"名称", "地铁", "电话", "地址", "评分", "周边景点", "周边餐馆", "周边酒店"}
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
                                break
                        ent_type = kb_item[1]
                        break
                if ent_type in duplicated_set:
                    ent_type = domain + '_' + ent_type
                sketch_response.append('@' + str(ent_type))
                assert ent_type is not None, word
    sketch_response = " ".join(sketch_response)
    return sketch_response


def read_graph(file_path):
    node_num = 0
    edge1_src, edge1_trg, edge2_src, edge2_trg, edge3_src, edge3_trg = [], [], [], [], [], []
    graph_list = []
    with open(file_path, encoding="utf-8") as fin:
        flag = 0
        cnt = 1
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

                cnt += 1

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


def get_domain_entities(path):
    with open(path, encoding='utf-8') as f:
        domain_entity = json.load(f)
        domain_entity_list = []
        for subject in domain_entity:
            key_dict = subject[1]
            for attribute in key_dict.keys():
                if attribute == "领域":
                    continue
                if attribute == "电话":
                    if ',' in key_dict[attribute]:
                        for j in key_dict[attribute].split(','):
                            domain_entity_list.append(j.strip().replace(' ', '_'))
                    elif ';' in key_dict[attribute]:
                        for j in key_dict[attribute].split(';'):
                            domain_entity_list.append(j.strip().replace(' ', '_'))
                    else:
                        domain_entity_list.append(str(key_dict[attribute]).strip().replace(' ', '_'))
                    continue
                if isinstance(key_dict[attribute], list):
                    domain_entity_list += key_dict[attribute]
                else:
                    domain_entity_list.append(str(key_dict[attribute]).strip().replace(" ", "_"))
        global_entity_list = list(set(domain_entity_list))
    return global_entity_list


def prepare_data(fast_test=False, v_train=False, test_only=False):
    # domain_dict
    domain2idx = {"对话历史": 0, "景点": 1, "餐馆": 2, "酒店": 3, "地铁": 4, "出租": 5}
    idx2domain = {0: "对话历史", 1: "景点", 2: "餐馆", 3: "酒店", 4: "地铁", 5: "出租"}
    # domain_list
    domain_list = ["对话历史", "景点", "餐馆", "酒店", "地铁", "出租"]
    attraction_entity = get_domain_entities(data_loc + "database-2.0/attraction_db.json")
    hotel_entity = get_domain_entities(data_loc + "database-2.0/hotel_db.json")
    metro_entity = get_domain_entities(data_loc + "database-2.0/metro_db.json")
    res_entity = get_domain_entities(data_loc + "database-2.0/restaurant_db.json")
    taxi_entity = get_domain_entities(data_loc + "database-2.0/taxi_db.json")
    global_entity = list(set(attraction_entity + hotel_entity + metro_entity + res_entity + taxi_entity))

    train_graph = read_graph(data_loc + "train_graph.txt") if not fast_test else read_graph(data_loc + "dev_graph.txt")
    dev_graph = read_graph(data_loc + "dev_graph.txt") if not v_train else read_graph(data_loc + "train_graph.txt")
    test_graph = read_graph(data_loc + "test_graph.txt") if not v_train else read_graph(data_loc + "train_graph.txt")

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


if __name__ == '__main__':
    prepare_data()
