from torch.optim import lr_scheduler
from torch import optim
import random
import dgl
import os
from tqdm import tqdm

from utils.measures import moses_multi_bleu
from utils.masked_cross_entropy import *
from utils.config import *
from models.multi_modules import *


class MultiTableGraph(nn.Module):
    def __init__(self, hidden_size, lang, max_resp_len, path, lr, n_layers, dropout,
                 num_edges, graph_hidden_size, num_heads, num_graph_layer, attention_out_size,
                 weight_decay, domain2idx, idx2domain, global_entity, dataset_name):
        super(MultiTableGraph, self).__init__()
        self.copy_list = []
        self.name = "MultiTableGraph"
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_resp_len = max_resp_len
        self.decoder_hop = n_layers
        self.softmax = nn.Softmax(dim=0)
        self.heads = [num_heads for _ in range(num_graph_layer)]
        self.domain2idx = domain2idx
        self.idx2domain = idx2domain
        self.attention_out_size = attention_out_size
        self.global_entity = global_entity
        self.dataset_name = dataset_name
        self.num_edges = num_edges
        self.is_cross = True if dataset_name == 'cross' else False

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th')
                self.extKnow = torch.load(str(path) + '/enc_kb.th')
                self.decoder = torch.load(str(path) + '/dec.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th', lambda storage, loc: storage)
                self.extKnow = torch.load(str(path) + '/enc_kb.th', lambda storage, loc: storage)
                self.decoder = torch.load(str(path) + '/dec.th', lambda storage, loc: storage)
        else:
            self.encoder = Encoder(lang.n_words, hidden_size, attention_out_size, dropout)
            self.extKnow = GraphEK(lang.n_words, hidden_size, dropout,
                                   num_edges, graph_hidden_size, hidden_size, self.heads, domain2idx)
            self.decoder = Decoder(self.encoder.embedding, lang, hidden_size, dropout)

        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.AdamW(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
        self.extKnow_optimizer = optim.AdamW(self.extKnow.parameters(), lr=lr, weight_decay=weight_decay)
        self.decoder_optimizer = optim.AdamW(self.decoder.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)

        self.reset()

        if USE_CUDA:
            self.encoder.cuda()
            self.extKnow.cuda()
            self.decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        # print_loss_g = self.loss_g / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_l = self.loss_l / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LG:{:.2f},LP:{:.2f}'.format(print_loss_avg, print_loss_v, print_loss_l)

    def save_model(self, dec_type, pred: list, real: list, sk_pred: list, sk_real: list):

        layer_info = str(self.n_layers)
        directory = 'save/MG-' + args["addName"] + self.dataset_name + 'HDD' + str(
            self.hidden_size) + 'BSZ' + str(args['batch']) + 'DR' + str(self.dropout) + 'L' + layer_info + 'lr' + str(
            self.lr) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.extKnow, directory + '/enc_kb.th')
        torch.save(self.decoder, directory + '/dec.th')
        with open(os.path.join(directory, 'sample.txt'), 'w', encoding='utf-8') as ref:
            for p, r, sp, sr in zip(pred, real, sk_pred, sk_real):
                ref.write(str(p) + "\t" + str(r) + "\t" + str(sp) + "\t" + str(sr) + '\n')
        self.dev_save_directory = directory

    def save_test_samples(self, pred: list, real: list, sk_pred: list, sk_real: list):
        with open(os.path.join(self.dev_save_directory, 'test_sample.txt'), 'w', encoding='utf-8') as ref:
            for p, r, sp, sr in zip(pred, real, sk_pred, sk_real):
                ref.write(str(p) + "\t" + str(r) + "\t" + str(sp) + "\t" + str(sr) + '\n')

    def reset(self):
        self.loss, self.print_every, self.loss_v, self.loss_l = 0, 1, 0, 0

    def _cuda(self, x, d_type=None):
        if USE_CUDA:
            if d_type is not None:
                return torch.as_tensor(x, dtype=d_type).cuda()
            return torch.Tensor(x).cuda()
        else:
            if d_type is not None:
                return torch.as_tensor(x, dtype=d_type)
            return torch.Tensor(x)

    def train_batch(self, data, clip, reset=0):
        if reset:
            self.reset()
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.extKnow_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Encode and Decode
        use_teacher_forcing = random.random() < args['teacher_forcing_ratio']
        max_target_length = max(data['response_lengths'])
        all_decoder_outputs_vocab, all_decoder_outputs_ptr, _, _ = \
            self.encode_and_decode(data, max_target_length, use_teacher_forcing, False)

        loss_v = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),
            data['sketch_response'].contiguous(),
            data['response_lengths'])
        loss_l = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(),
            data['ptr_index'].contiguous(),
            data['response_lengths'])
        loss = loss_v + loss_l
        loss.backward()

        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        ec = torch.nn.utils.clip_grad_norm_(self.extKnow.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.extKnow_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()
        self.loss_v += loss_v.item()
        self.loss_l += loss_l.item()

    def construct_graph(self, batch_graph_list: list, node_nums: list, edge_nums: int):
        """
        Construct graph for batch_graph_list
        return [list of [batched DGLGraph for an edge]]
        """
        edge_dgl_graph, batched_dgl_graph, len_dia = [[] for _ in range(edge_nums)], [], len(batch_graph_list[0])
        for i, dia in enumerate(batch_graph_list):
            node_num, cnt = node_nums[i], 0
            for k in range(1, len_dia, 2):
                g = dgl.DGLGraph()
                if USE_CUDA:
                    g = g.to('cuda:0')
                g.add_nodes(node_num)
                u = self._cuda(np.concatenate([dia[k], dia[k+1]]), d_type=torch.int64)
                v = self._cuda(np.concatenate([dia[k+1], dia[k]]), d_type=torch.int64)
                g.add_edges(u, v)
                g = dgl.add_self_loop(g)
                edge_dgl_graph[cnt].append(g)
                cnt += 1
        for list_g in edge_dgl_graph:
            batched_dgl_graph.append(dgl.batch(list_g))
        return batched_dgl_graph

    def encode_and_decode(self, data, max_target_length, use_teacher_forcing, get_decoded_words):
        # GLMP Paper: In addition, to increase model generalization and simulate OOV setting, we randomly mask a small
        #             number of input source tokens into an unknown token
        if args['unk_mask'] and self.decoder.training:
            story_size = data['context_arr'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
            rand_mask[:, :, 0] = rand_mask[:, :, 0] * bi_mask
            conv_rand_mask = np.ones(data['conv_arr'].size())
            for bi in range(story_size[0]):
                start, end = data['kb_arr_lengths'][bi], data['kb_arr_lengths'][bi] + data['conv_arr_lengths'][bi]
                conv_rand_mask[:end - start, bi, :] = rand_mask[bi, start:end, :]
            rand_mask = self._cuda(rand_mask)
            conv_rand_mask = self._cuda(conv_rand_mask)
            conv_story = data['conv_arr'] * conv_rand_mask.long()
            story = data['context_arr'] * rand_mask.long()
        else:
            story, conv_story = data['context_arr'], data['conv_arr']
        # each element is a batched dgl graph
        graphs = self.construct_graph(data['graph'], data['node_num'], self.num_edges)
        # Encode dialog history and KB to vectors
        cat_hidden, sent_rep = self.encoder(conv_story, data['conv_arr_lengths'])
        self.extKnow.load_memory(story, data['kb_arr_lengths'], data['conv_arr_lengths'], cat_hidden, graphs,
                                 data['node_num'], data['domain_span'], data['domain_num'])

        # Get the words that can be copy from the memory
        batch_size = len(data['context_arr_lengths'])
        self.copy_list = []
        for elm in data['context_arr_plain']:
            elm_temp = [word_arr[0] for word_arr in elm]
            self.copy_list.append(elm_temp)
        outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse = self.decoder.forward(
            self.extKnow,
            story.shape[1],
            data['context_arr_lengths'],
            self.copy_list,
            sent_rep,
            data['sketch_response'],
            max_target_length,
            batch_size,
            use_teacher_forcing,
            get_decoded_words)

        return outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse

    def evaluate(self, dev, matric_best, early_stop=None, is_test=False):
        if is_test:
            print("Testing...")
        else:
            print("STARTING EVALUATION")
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.extKnow.train(False)
        self.decoder.train(False)

        ref, hyp = [], []
        ref_sk, hyp_sk = [], []
        acc, total = 0, 0
        # dialog_acc_dict = {}
        F1_pred, F1_pred_coarse, F1_count, F1_count_coarse = 0, 0, 0, 0
        TP_all, FP_all, FN_all = 0, 0, 0

        pbar = tqdm(enumerate(dev), total=len(dev), ncols=50)
        # new_precision, new_recall, new_f1_score = 0, 0, 0

        for j, data_dev in pbar:
            # Encode and Decode
            _, _, decoded_fine, decoded_coarse = self.encode_and_decode(data_dev, self.max_resp_len, False, True)
            decoded_coarse = np.transpose(decoded_coarse)
            decoded_fine = np.transpose(decoded_fine)
            for bi, row in enumerate(decoded_fine):
                st = ''
                for e in row:
                    if e == 'EOS':
                        break
                    else:
                        st += e + ' '
                st_c = ''
                for e in decoded_coarse[bi]:
                    if e == 'EOS':
                        break
                    else:
                        st_c += e + ' '
                pred_sent = st.lstrip().rstrip()
                pred_sent_coarse = st_c.lstrip().rstrip()
                gold_sent = data_dev['response_plain'][bi].lstrip().rstrip()
                gold_sent_coarse = data_dev['sketch_response_plain'][bi].lstrip().rstrip()
                ref.append(gold_sent)
                hyp.append(pred_sent)
                hyp_sk.append(pred_sent_coarse)
                ref_sk.append(gold_sent_coarse)

                if args['dataset'] == 'risa':
                    # compute F1 SCORE
                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(),
                                                        self.global_entity, data_dev['kb_arr_plain'][bi])
                    sk_single_f1, sk_count = self.compute_sketch_prf(gold_sent_coarse.split(), pred_sent_coarse.split())
                    F1_pred += single_f1
                    F1_count += count
                    TP_all += single_tp
                    FP_all += single_fp
                    FN_all += single_fn

                    F1_pred_coarse += sk_single_f1
                    F1_count_coarse += sk_count
                elif args['dataset'] == 'cross':
                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(),
                                                        self.global_entity, data_dev['kb_arr_plain'][bi])
                    sk_single_f1, sk_count = self.compute_sketch_prf(gold_sent_coarse.split(), pred_sent_coarse.split())
                    F1_pred += single_f1
                    F1_count += count
                    TP_all += single_tp
                    FP_all += single_fp
                    FN_all += single_fn

                    F1_pred_coarse += sk_single_f1
                    F1_count_coarse += sk_count

                total += 1
                if gold_sent == pred_sent:
                    acc += 1

                if args['genSample']:
                    self.print_examples(bi, data_dev, pred_sent, pred_sent_coarse, gold_sent)

        # Set back to training mode
        self.encoder.train(True)
        self.extKnow.train(True)
        self.decoder.train(True)

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref))
        bleu_score_coarse = moses_multi_bleu(np.array(hyp_sk), np.array(ref_sk))
        acc_score = acc / float(total)
        print("ACC SCORE:\t" + str(acc_score))

        if args['dataset'] == 'risa':
            P_score = TP_all / float(TP_all + FP_all) if (TP_all + FP_all) != 0 else 0
            R_score = TP_all / float(TP_all + FN_all) if (TP_all + FN_all) != 0 else 0
            F1_score = self.compute_F1(P_score, R_score)

            print("F1 SCORE:\t{}".format(F1_score))
            print("BLEU SCORE:\t" + str(bleu_score))
            print("Sketch f1 score:\t{}".format(F1_pred_coarse / float(F1_count_coarse)))
            print('Sketch BLEU score:\t' + str(bleu_score_coarse))
        elif args['dataset'] == 'cross':
            P_score = TP_all / float(TP_all + FP_all) if (TP_all + FP_all) != 0 else 0
            R_score = TP_all / float(TP_all + FN_all) if (TP_all + FN_all) != 0 else 0
            F1_score = self.compute_F1(P_score, R_score)

            print("F1 SCORE:\t{}".format(F1_score))
            print("BLEU SCORE:\t" + str(bleu_score))
            print("Sketch f1 score:\t{}".format(F1_pred_coarse / float(F1_count_coarse)))
            print('Sketch BLEU score:\t' + str(bleu_score_coarse))

        # if is_test:
            # self.save_test_samples(hyp, ref, hyp_sk, ref_sk)
        if early_stop == 'BLEU':
            if bleu_score >= matric_best:
                self.save_model('BLEU-' + str(bleu_score), hyp, ref, hyp_sk, ref_sk)
                print("MODEL SAVED")
            return bleu_score
        elif early_stop == 'ENTF1':
            if F1_score >= matric_best:
                self.save_model('ENTF1-{:.4f}'.format(F1_score), hyp, ref, hyp_sk, ref_sk)
                print("MODEL SAVED")
            return F1_score
        else:
            if acc_score >= matric_best:
                self.save_model('ACC-{:.4f}'.format(acc_score), hyp, ref, hyp_sk, ref_sk)
                print("MODEL SAVED")
            return acc_score

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return TP, FP, FN, F1, count

    def compute_F1(self, precision, recall):
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        return F1

    def compute_sketch_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        gold_sk, pred_sk = [], []
        for word in gold:
            if '@' in word:
                gold_sk.append(word)
        if len(gold_sk) == 0:
            return 0, 0
        for word in pred:
            if '@' in word:
                pred_sk.append(word)
        for g in gold_sk:
            if g in pred_sk:
                TP += 1
            else:
                FN += 1
        for p in set(pred_sk):
            if p not in gold_sk:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0

        return F1, 1

    def print_examples(self, batch_idx, data, pred_sent, pred_sent_coarse, gold_sent):
        kb_len = len(data['context_arr_plain'][batch_idx]) - data['conv_arr_lengths'][batch_idx] - 1
        print("{}: ID{} id{} ".format(data['domain'][batch_idx], data['ID'][batch_idx], data['id'][batch_idx]))
        for i in range(kb_len):
            kb_temp = [w for w in data['context_arr_plain'][batch_idx][i] if w != 'PAD']
            kb_temp = kb_temp[::-1]
            if 'poi' not in kb_temp:
                print(kb_temp)
        flag_uttr, uttr = '$u', []
        for word_idx, word_arr in enumerate(data['context_arr_plain'][batch_idx][kb_len:]):
            if word_arr[1] == flag_uttr:
                uttr.append(word_arr[0])
            else:
                print(flag_uttr, ': ', " ".join(uttr))
                flag_uttr = word_arr[1]
                uttr = [word_arr[0]]
        print('Sketch System Response : ', pred_sent_coarse)
        print('Final System Response : ', pred_sent)
        print('Gold System Response : ', gold_sent)
        print('\n')
