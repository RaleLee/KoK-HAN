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
    # early_stop = 'BLEU'
elif args['dataset'] == 'cross':
    from utils.Cross_dataloader import *
else:
    print("[ERROR] You need to provide the --dataset information")

# Configure models and load data
avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, lang, max_resp_len, domain2idx, idx2domain, global_entity = prepare_data(fast_test=args['fast_test'],
                                                                                           v_train=args['verify_train'])

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

for epoch in range(args['epoch']):
    print("Epoch:{}".format(epoch))
    # Run the train function
    # train
    pbar = tqdm(enumerate(train), total=len(train), ncols=100)
    for i, data in pbar:
        model.train_batch(data, int(args['clip']), reset=(i == 0))
        pbar.set_description(model.print_loss())
        # break
    # eval
    if (epoch + 1) % int(args['evalp']) == 0:
        acc = model.evaluate(dev, avg_best, early_stop)
        model.scheduler.step(acc)
        if acc >= avg_best:
            avg_best = acc
            cnt = 0
            if not args['no_test']:
                acc = model.evaluate(test, 1e7, early_stop, is_test=True)
            print("dev metric is better than before, test metric:{}".format(acc))
        else:
            cnt += 1
        if cnt == 8 or (acc == 1.0 and early_stop == None):
            print("Ran out of patient, early stop...")
            break
