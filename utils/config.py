import os
import argparse

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0

MAX_LENGTH = 10

parser = argparse.ArgumentParser(description='Multi_Table_Graph')
# control gpu in args
parser.add_argument('-g', '--gpu', action='store_true', help='Use cuda backend', default=False)
# epoch
parser.add_argument('-ep', '--epoch', help='epoch of train', default=200, type=int)
# use for fast_test
parser.add_argument('-fs', '--fast_test', action='store_true', help='use for load dev as train, to check bug',
                    default=False)
# use for verify train
parser.add_argument('-vt', '--verify_train', action='store_true', help='use for load train as train, to check bug',
                    default=False)
parser.add_argument('-nt', '--no_test', action='store_true', help='use for no test, to check bug',
                    default=False)

parser.add_argument('-ds', '--dataset', help='dataset, risa or cross', default='risa')
parser.add_argument('-t', '--task', help='Task Number', required=False, default="")
parser.add_argument('-dec', '--decoder', help='decoder model', required=False, default='MultiTableGraph')
parser.add_argument('-hdd', '--hidden', help='Hidden size', required=False, default=128)
parser.add_argument('-bsz', '--batch', help='Batch_size', required=False, type=int, default=32)
parser.add_argument('-lr', '--learn', help='Learning Rate', required=False, default=0.001)
parser.add_argument('-dr', '--drop', help='Drop Out', required=False, default=0.2)
parser.add_argument('-um', '--unk_mask', help='mask out input token to UNK', type=int, required=False, default=0)
parser.add_argument('-l', '--layer', help='Layer Number', required=False, default=1)
parser.add_argument('-lm', '--limit', help='Word Limit', required=False, default=-10000)
parser.add_argument('-path', '--path', help='path of the file to load', required=False)
parser.add_argument('-clip', '--clip', help='gradient clipping', required=False, default=10)
parser.add_argument('-tfr', '--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False,
                    default=0.9)

parser.add_argument('-sample', '--sample', help='Number of Samples', required=False, default=None)
parser.add_argument('-evalp', '--evalp', help='evaluation period', required=False, default=1)
parser.add_argument('-an', '--addName', help='An add name for the save folder', required=False, default='')
parser.add_argument('-gs', '--genSample', help='Generate Sample', required=False, default=0)
parser.add_argument('-es', '--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False, default='BLEU')
parser.add_argument('-abg', '--ablationG', help='ablation global memory pointer', type=int, required=False, default=0)
parser.add_argument('-abh', '--ablationH', help='ablation context embedding', type=int, required=False, default=0)
parser.add_argument('-rec', '--record', help='use record function during inference', type=int, required=False,
                    default=1)
# fix random seed
parser.add_argument('-fixed', '--fixed', help='fix seeds', required=False, default=True)
parser.add_argument('-rs', '--random_seed', help='choose random_seed', required=False, default=2021)

# num_edges, graph_hidden_size, graph_out_size, num_heads
parser.add_argument('-edge', '--num_edges', type=int, default=3)
parser.add_argument('-ghs', '--graph_hidden_size', type=int, default=128)
parser.add_argument('-heads', '--num_heads', type=int, default=8)
parser.add_argument('-ngl', '--num_graph_layer', type=int, default=2)

parser.add_argument('-cs', '--case_study', action='store_true', default=False)

# AdamW weight decay
parser.add_argument('-wd', '--weight_decay', help='weight decay for AdamW', type=float, default=1e-2)
# attention_output_size
parser.add_argument('-aos', '--attention_output_size', type=int, default=128)
args = vars(parser.parse_args())
print(str(args))
USE_CUDA = args['gpu']
print("USE_CUDA: " + str(USE_CUDA))

LIMIT = int(args["limit"])
MEM_TOKEN_SIZE = 4

if args["ablationG"]:
    args["addName"] += "ABG"
if args["ablationH"]:
    args["addName"] += "ABH"
