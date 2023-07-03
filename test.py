from models.multi_table import *

# fixed random seed
if args['fixed']:
    torch.manual_seed(args['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['random_seed'])
        torch.cuda.manual_seed_all(args['random_seed'])
        torch.backends.cudnn.deterministic = True
    np.random.seed(args['random_seed'])
    random.seed(args['random_seed'])

early_stop = args['earlyStop']
if args['dataset'] == 'risa':
    from utils.RiSA_dataloader import *
    early_stop = 'BLEU'
elif args['dataset'] == 'cross':
    from utils.Cross_dataloader import *
else:
    print("[ERROR] You need to provide the --dataset information")

# Configure models and load data
avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, lang, max_resp_len, domain2idx, idx2domain, global_entity = prepare_data()

# Testing
model = globals()['MultiTableGraph'](
    int(args['hidden']),
    lang,
    max_resp_len,
    args['path'],
    lr=float(args['learn']),
    n_layers=int(args['layer']),
    dropout=float(args['drop']),
    num_edges=args['num_edges'],
    graph_hidden_size=args['graph_hidden_size'],
    num_heads=args['num_heads'],
    num_graph_layer=args['num_graph_layer'],
    attention_out_size=args['attention_output_size'],
    weight_decay=args['weight_decay'],
    domain2idx=domain2idx,
    idx2domain=idx2domain,
    global_entity=global_entity,
    dataset_name=args['dataset']
)

final_test = model.evaluate(test, 1e7, is_test=True)
