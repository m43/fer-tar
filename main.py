import argparse
import pickle
import pprint

from classifier import CompoundClassifier, FCClassifier, LSTMClassifier, HackyCompoundClassifier, MCCBaseline, \
    SVMClassifier
from dataset import load_features, TRAITS, wrap_datasets
from eval import eval
from extractor import *
from rnn_dataset import load_rnn_features, load_embeddings, pad_collate_fn, load_features_2, EMOTION_DROP_VERSIONS
from utils import setup_torch_reproducibility, setup_torch_device, project_path, get_str_formatted_time, ensure_dir


def welcome():
    print("Personality Trait Classification on Essays")
    print("Text Analysis and Retrieval, 2021")
    print("FER, University of Zagreb, Croatia\n")


MODELS = ["lstm", "fc", "svm", "chunk", "mcc"]


def main(args):
    welcome()
    time_str = get_str_formatted_time()  # string time identifier
    print(f"TIME: {time_str}")

    # ~~~~ Config ~~~~ #
    device = setup_torch_device()
    seed = 42  # Initial seed, used to setup reproducibility
    setup_torch_reproducibility(seed)
    n_runs = args.n_runs  # How many times will each model be run, used to calculate std. dev.
    test_ratio, val_ratio = 0.2, 0.2
    batch_size = args.bs
    train_args = {"epochs": args.epochs, "lr": args.lr, "wd": args.wd, "es_patience": args.espat, "es_epsilon": 1e-7,
                  "es_maxiter": 30, "device": device, "debug_print": True}
    model = args.model
    assert model in MODELS

    run_name_prefix = args.run_name_prefix
    run_name_suffix = f"_nruns={n_runs}" \
                      f"_e={train_args['epochs']}" \
                      f"_lr={train_args['lr']}" \
                      f"_wd={train_args['wd']}" \
                      f"_espat={train_args['es_patience']}" \
                      f".es2"

    # ~~~~ LSTM (& MCC) setup ~~~~ #
    rnn_init = {'rnn_layers': args.rnn_layers, 'bidirectional': args.bidirectional, 'activation_fn': torch.nn.ReLU,
                'device': device, 'clip': args.clip}
    if model == "lstm" or model == "mcc":
        extract_cfg = {
            'valid_ratio': val_ratio, 'test_ratio': test_ratio,
            'device': device,
            "min_freq": 2,
            "max_size": -1,
            'w2v_limit': None,
            's2v': False,
            'wiki': True,
            "max_essays": None
        }

        use_pickle = False
        pickle_path = os.path.join(project_path, "saved/kiseli.pk")
        if use_pickle and os.path.exists(pickle_path):
            print("Loading (train, valid, trainval, test, vocab) from pickled dump")
            with open(pickle_path, "rb") as f:
                train, valid, trainval, test, vocab = pickle.load(f)
        else:
            train, valid, trainval, test, vocab = load_rnn_features(**extract_cfg)
            with open(pickle_path, "wb") as f:
                pickle.dump((train, valid, trainval, test, vocab), f)

        train, valid, trainval, test = wrap_datasets(batch_size, pad_collate_fn, train, valid, trainval, test)
        # pickle.

        if extract_cfg['s2v']:
            in_dim = 600 if extract_cfg['wiki'] else 700
        else:
            in_dim = 300
            print("Load embeddings")
            embeddings = load_embeddings(vocab, **extract_cfg)
            rnn_init['embeddings'] = embeddings
            print("Embeddings loaded")

        rnn_init['rnn_dims'] = [
            in_dim,
            args.lstm_hiddendim_lstm
        ]
        # rnn_init['fc_hidden'] = [512, 128, 1]
        rnn_init["fc_hid_dim"] = args.lstm_hiddendim_fc

        clf_hook = lambda: CompoundClassifier([(LSTMClassifier, rnn_init)] * 5)
        run_name_middle = f"LSTM" \
                          f"_rnn_layers={rnn_init['rnn_layers']}" \
                          f"_hLSTM={rnn_init['rnn_dims'][1]}" \
                          f"_hFC={rnn_init['fc_hid_dim']}" \
                          f"_biderctional={1 if rnn_init['bidirectional'] else 0}" \
                          f"_clip={rnn_init['clip']}"

    # ~~~~ LSTM+CHUNK setup ~~~~ #
    if model == "chunk":
        extract_cfg = {
            'valid_ratio': val_ratio, 'test_ratio': test_ratio,
            'device': device,
            "min_freq": 2,
            "max_size": -1,
            'w2v_limit': None,
            'emotion_drop': args.emotion_drop,
            'min_chunk_length': args.min_chunk_length,

        }
        (train, train_auth), (val, val_auth), (tv, tv_auth), (test, test_auth), vocab = load_features_2(**extract_cfg)
        print("Wrapping datasets into loaders...", end=' ')
        train, val, tv, test = wrap_datasets(batch_size, pad_collate_fn, train, val, tv, test)
        print("DONE")
        train, valid, trainval, test = (train, train_auth), (val, val_auth), (tv, tv_auth), (test, test_auth)

        in_dim = 300
        print("Loading W2V embeddings...", end=' ')
        embeddings = load_embeddings(vocab, **extract_cfg)
        print("DONE")
        rnn_init['embeddings'] = embeddings

        rnn_init['rnn_dims'] = [in_dim, args.lstm_hiddendim_lstm]
        # rnn_init['fc_hidden'] = [512, 128, 1]
        rnn_init["fc_hid_dim"] = args.lstm_hiddendim_fc

        clf_hook = lambda: HackyCompoundClassifier([(LSTMClassifier, rnn_init)] * 5)
        run_name_middle = f"LSTM+CHUNK" \
                          f"_emo={extract_cfg['emotion_drop']}" \
                          f"_rnn_layers={rnn_init['rnn_layers']}" \
                          f"_hLSTM={rnn_init['rnn_dims'][1]}" \
                          f"_hFC={rnn_init['fc_hid_dim']}" \
                          f"_biderctional={1 if rnn_init['bidirectional'] else 0}" \
                          f"_clip={rnn_init['clip']}"

    # ~~~~ FC & SVM setup ~~~~ #
    if model == "fc" or model == "svm":
        extract_cfg = {
            'valid_ratio': val_ratio, 'test_ratio': test_ratio,
            'device': device,
            'wiki': True,  # S2V -- wiki or torronto
            'w2v_limit': None
        }
        ext_hooks = []
        hook_names = []
        if args.fcsvm_hook_bow:
            ext_hooks += [BOWExtractor]
            hook_names += ["BOW"]
        if args.fcsvm_hook_w2v:
            ext_hooks += [W2VExtractor]
            hook_names += ["W2V"]
        if args.fcsvm_hook_custom:
            ext_hooks += [InterpunctionExtractor, CapitalizationExtractor, RepeatingLettersExtractor,
                          WordCountExtractor]
            hook_names += ["CUSTOM"]
        run_name_middle = "+".join(hook_names)
        print("EXT HOOKS", ext_hooks)

        train, valid, trainval, test = load_features(ext_hooks, **extract_cfg)
        in_dim = len(train[0][0])
        train, valid, trainval, test = wrap_datasets(batch_size, None, train, valid, trainval, test)

        if model == "fc":
            fc_init = {"neurons_per_layer": [in_dim, *args.fc_npl, 1], "activation_module": torch.nn.ReLU,
                       "device": device}
            clf_hook = lambda: CompoundClassifier([(FCClassifier, fc_init)] * 5)
            run_name_middle = f"FC" \
                              f"_npl=in-{'-'.join([str(h) for h in fc_init['neurons_per_layer']])}" \
                              f"__{run_name_middle}"

        if model == "svm":
            svm_init = {"c": args.svm_c, "gamma": args.svm_gamma, "decision_function_shape": "ovo",
                        "kernel": args.svm_kernel, "in_dim": in_dim}
            clf_hook = lambda: CompoundClassifier([(SVMClassifier, svm_init)] * 5)
            run_name_middle = f"SVM" \
                              f"_c={svm_init['c']}" \
                              f"_gamma={svm_init['gamma']}" \
                              f"_kernel={svm_init['kernel']}" \
                              f"_{run_name_middle}"


    # ~~~~ MCC setup ~~~~ #
    elif model == "mcc":
        run_name_middle = "MCC"
        clf_hook = lambda: CompoundClassifier(
            [(MCCBaseline, {}), (MCCBaseline, {}), (MCCBaseline, {}), (MCCBaseline, {}), (MCCBaseline, {}), ])

    run_name = run_name_prefix + run_name_middle + run_name_suffix
    data_loaders = (train, valid, trainval, test)
    save_dir = os.path.join(project_path, f"imgs")  # Where should logs be saved
    ensure_dir(save_dir)
    results_csv = os.path.join(save_dir, f"{run_name}.csv")
    if os.path.exists(results_csv):
        print(f"Configuration already run, file exists: {results_csv}")
    print(f"Running configuration: {results_csv}")

    # ~~~~ Collect results ~~~~ #
    print("\nCollecting results")
    results = {}
    for i in range(n_runs):
        clf = clf_hook()
        print(f"i::[{i + 1}/{n_runs}] started")
        print("Using classifier " + str(clf))

        clf.train(data_loaders, **train_args)

        for subset_name, subset in [("train", data_loaders[0]), ("test", data_loaders[-1])]:
            if subset_name not in results:
                results[subset_name] = {}
            eval_results = eval(clf, subset)
            for trait_name in TRAITS:
                if trait_name not in results[subset_name]:
                    results[subset_name][trait_name] = {}
                for metric_name in eval_results[trait_name]:
                    r = results[subset_name][trait_name].get(metric_name, [])
                    e = eval_results[trait_name][metric_name]
                    results[subset_name][trait_name][metric_name] = r + [e]
        print(f"i::[{i + 1}/{n_runs}] current results: {results}\n\n")

    metric_names = results[subset_name][TRAITS[0]]
    for subset_name in results:
        for trait_name in TRAITS:
            for metric_name in metric_names:
                r = results[subset_name][trait_name][metric_name]
                mean, std = np.mean(np.array(r, np.float64)), np.std(np.array(r, np.float64))
                results[subset_name][trait_name][metric_name] = (mean, std)

    print("Results collected")
    print()

    # ~~~~ Save results ~~~~ #
    # TXT
    with open(results_csv, 'w') as csv:
        csv.write(f"line_id,subset_name,trait_name,metric_name,mean,std\n")
        for subset_name in results:
            for trait_name in TRAITS:
                for metric_name in metric_names:
                    line_id = f"{subset_name}-{trait_name}-{metric_name}"
                    mean, std = results[subset_name][trait_name][metric_name]
                    csv.write(f"{line_id},{subset_name},{trait_name},{metric_name},{mean},{std}\n")

    # ~~~~ Print results ~~~~ #
    pp = pprint.PrettyPrinter(width=100, compact=True)
    pp.pprint(results)

    note = f"""
    You can go ahead and open the csv file in your favorite
    spreadsheet editor. My favorite is Google Sheets, as all
    of us can collaborate, paste local results and visualize
    them. I open the csv in LibreOffice Calc, copy the content
    and paste it into Google Sheets. Here is a google sheets
    document that I've shared with you: https://cutt.ly/wbUjfdt

    Local csv path: {results_csv}"""
    print(note)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Configuration parameters')

    # For ALL models (mostly, SVM does not have many if these)
    p.add_argument('--epochs', default=100, type=int, help='Maximum number of epochs')
    p.add_argument('--lr', default=1e-3, type=float, help='The value of the learning rate')
    p.add_argument('--wd', default=1e-6, type=float, help='L2 regularization (weight decay)')
    p.add_argument('--bs', default=32, type=int, help='Batch size')
    p.add_argument('--n_runs', default=10, type=int, help='Number of times to run (to determine avg and stddev)')
    p.add_argument('--espat', default=15, type=int, help='Early stopping patiance')
    p.add_argument('--model', default="fc", help=f'Which model out of {",".join(MODELS)}', choices=MODELS)
    p.add_argument('--run_name_prefix', default="", help=f'Prefix that will be added to the run name')
    # --epochs 100 --lr 1e-3 --wd 1e-6 --bs 32 --n_runs 10 --espat 15 --model fc

    # LSTM and CHUNK
    p.add_argument('--lstm_hiddendim_lstm', default=200, type=int, help='LSTM hiddendim')
    p.add_argument('--lstm_hiddendim_fc', default=200, type=int, help='hiddendim of FC after LSTM')
    p.add_argument('--rnn_layers', default=1, type=int, help='Number of stacked LSTMs')
    p.add_argument('--clip', default=1, type=float, help='Graddient clipping value (clipping gradient norm)')
    p.add_argument('--bidirectional', action='store_true')
    # --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 200 --rnn_layers 1 --clip 1
    # --bidirectional

    # CHUNK
    p.add_argument('--emotion_drop', default="none", choices=EMOTION_DROP_VERSIONS,
                   help=f'Which emotion_drop version out of {",".join(EMOTION_DROP_VERSIONS)}')
    p.add_argument('--min_chunk_length', default=20, type=int)
    # --min_chunk_length --emotion_drop none
    # --min_chunk_length --emotion_drop v1sent
    # --min_chunk_length --emotion_drop v2chunk

    # FC and SVM
    p.add_argument('--fcsvm_hook_custom', action='store_true')
    p.add_argument('--fcsvm_hook_bow', action='store_true')
    p.add_argument('--fcsvm_hook_w2v', action='store_true')
    # --fcsvm_hook_custom
    # --fcsvm_hook_bow
    # --fcsvm_hook_w2v

    # FC
    p.add_argument('--fc_npl', default=[100], nargs="+", type=int, help="FC Neurons per layer")
    # --fc_npl 100
    # ... --fc_npl 100 200 300 ...

    # SVM
    p.add_argument('--svm_c', default=1, type=float)
    p.add_argument('--svm_gamma', default="auto")
    p.add_argument('--svm_kernel', default="rbf")
    # --svm_c 1 --svm_gamma auto --svm_kernel rbf

    args = p.parse_args()
    if args.svm_gamma != "auto":
        args.svm_gamma = float(args.svm_gamma)
    print(args)
    main(args)

"""
One night is all you need:
night.sh

#MCC
python -m main --model mcc --n_runs 1

#LSTM pure
!python -m main --model lstm --epochs 100 --lr 1e-3 --wd 1e-6 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 1 --clip 1      2>&1 | tee -a imgs/logs/lstm_1.txt
!python -m main --model lstm --epochs 100 --lr 1e-3 --wd 1e-5 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 120 --lstm_hiddendim_fc 120 --rnn_layers 1 --clip 0.25 --bidirectional      2>&1 | tee -a imgs/logs/lstm_2.txt
!python -m main --model lstm --epochs 100 --lr 1e-4 --wd 1e-6 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 1 --clip 1      2>&1 | tee -a imgs/logs/lstm_3.txt
!python -m main --model lstm --epochs 100 --lr 1e-3 --wd 1e-6 --bs 256 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 1 --clip 1      2>&1 | tee -a imgs/logs/lstm_4.txt
!python -m main --model lstm --epochs 100 --lr 1e-3 --wd 1e-6 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 1 --clip 0.25      2>&1 | tee -a imgs/logs/lstm_5.txt
!python -m main --model lstm --epochs 100 --lr 1e-3 --wd 1e-6 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 2 --clip 0.25      2>&1 | tee -a imgs/logs/lstm_6.txt
!python -m main --model lstm --epochs 100 --lr 1e-3 --wd 1e-5 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 1 --clip 0.25      2>&1 | tee -a imgs/logs/lstm_7.txt
!python -m main --model lstm --epochs 100 --lr 1e-3 --wd 1e-4 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 1 --clip 0.25      2>&1 | tee -a imgs/logs/lstm_8.txt
!python -m main --model lstm --epochs 100 --lr 1e-3 --wd 1e-5 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 120 --lstm_hiddendim_fc 120 --rnn_layers 1 --clip 0.25      2>&1 | tee -a imgs/logs/lstm_9.txt
!python -m main --model lstm --epochs 100 --lr 1e-3 --wd 1e-5 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 120 --lstm_hiddendim_fc 120 --rnn_layers 2 --clip 0.25 --bidirectional      2>&1 | tee -a imgs/logs/lstm_10.txt

#LSTM chunk
!python -m main --model chunk --epochs 100 --lr 1e-3 --wd 1e-6 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 1 --clip 1 --min_chunk_length 20       2>&1 | tee -a imgs/logs/lstm_chunk_1.txt
!python -m main --model chunk --epochs 100 --lr 1e-3 --wd 1e-6 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 1 --clip 1 --min_chunk_length 20 --emotion_drop v1sent       2>&1 | tee -a imgs/logs/lstm_chunk_2.txt
!python -m main --model chunk --epochs 100 --lr 1e-4 --wd 1e-6 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 1 --clip 1 --min_chunk_length 20 --emotion_drop v1sent       2>&1 | tee -a imgs/logs/lstm_chunk_3.txt
!python -m main --model chunk --epochs 100 --lr 1e-3 --wd 1e-6 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 1 --clip 1 --min_chunk_length 20 --emotion_drop v2chunk       2>&1 | tee -a imgs/logs/lstm_chunk_4.txt
!python -m main --model chunk --epochs 100 --lr 1e-4 --wd 1e-6 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 1 --clip 1 --min_chunk_length 20 --emotion_drop v2chunk       2>&1 | tee -a imgs/logs/lstm_chunk_5.txt
!python -m main --model chunk --epochs 100 --lr 1e-3 --wd 1e-5 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 120 --lstm_hiddendim_fc 120 --rnn_layers 1 --clip 0.25 --min_chunk_length 20 --bidirectional --emotion_drop v2chunk       2>&1 | tee -a imgs/logs/lstm_chunk_6.txt
!python -m main --model chunk --epochs 100 --lr 1e-4 --wd 1e-6 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 1 --clip 1 --min_chunk_length 20 --emotion_drop v2chunk       2>&1 | tee -a imgs/logs/lstm_chunk_7.txt
!python -m main --model chunk --epochs 100 --lr 1e-3 --wd 1e-6 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 1 --clip 1 --min_chunk_length 20 --emotion_drop v2chunk       2>&1 | tee -a imgs/logs/lstm_chunk_8.txt
!python -m main --model chunk --epochs 100 --lr 1e-3 --wd 1e-5 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 120 --lstm_hiddendim_fc 120 --rnn_layers 1 --clip 0.25 --min_chunk_length 20 --bidirectional --emotion_drop v2chunk       2>&1 | tee -a imgs/logs/lstm_chunk_9.txt
!python -m main --model chunk --epochs 100 --lr 1e-4 --wd 1e-6 --bs 32 --n_runs 10 --espat 15 --lstm_hiddendim_lstm 200 --lstm_hiddendim_fc 150 --rnn_layers 1 --clip 1 --min_chunk_length 20 --emotion_drop v2chunk       2>&1 | tee -a imgs/logs/lstm_chunk_10.txt

#FC
!python -m main --model fc --epochs 100 --lr 1e-5 --wd 5e-5 --bs 32 --n_runs 10 --espat 40 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --fc_npl 100       2>&1 | tee -a imgs/logs/fc_1.txt
!python -m main --model fc --epochs 100 --lr 1e-4 --wd 5e-5 --bs 32 --n_runs 10 --espat 15 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --fc_npl 100       2>&1 | tee -a imgs/logs/fc_2.txt
!python -m main --model fc --epochs 100 --lr 5e-4 --wd 5e-5 --bs 32 --n_runs 10 --espat 15 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --fc_npl 100       2>&1 | tee -a imgs/logs/fc_3.txt
!python -m main --model fc --epochs 100 --lr 1e-3 --wd 5e-5 --bs 32 --n_runs 10 --espat 15 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --fc_npl 100       2>&1 | tee -a imgs/logs/fc_4.txt
!python -m main --model fc --epochs 100 --lr 5e-3 --wd 5e-5 --bs 32 --n_runs 10 --espat 15 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --fc_npl 100       2>&1 | tee -a imgs/logs/fc_5.txt
!python -m main --model fc --epochs 100 --lr 1e-4 --wd 5e-5 --bs 32 --n_runs 10 --espat 15 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --fc_npl 100 100       2>&1 | tee -a imgs/logs/fc_6.txt
!python -m main --model fc --epochs 100 --lr 1e-4 --wd 5e-5 --bs 32 --n_runs 10 --espat 15 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --fc_npl 50       2>&1 | tee -a imgs/logs/fc_7.txt
!python -m main --model fc --epochs 100 --lr 1e-4 --wd 5e-5 --bs 32 --n_runs 10 --espat 15 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --fc_npl 50       2>&1 | tee -a imgs/logs/fc_8.txt
!python -m main --model fc --epochs 100 --lr 1e-4 --wd 5e-5 --bs 256 --n_runs 10 --espat 15 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --fc_npl 100       2>&1 | tee -a imgs/logs/fc_9.txt
!python -m main --model fc --epochs 100 --lr 1e-4 --wd 5e-5 --bs 32 --n_runs 10 --espat 15 --fcsvm_hook_custom --fcsvm_hook_bow --fc_npl 100       2>&1 | tee -a imgs/logs/fc_10.txt
!python -m main --model fc --epochs 100 --lr 1e-4 --wd 5e-5 --bs 32 --n_runs 10 --espat 15 --fcsvm_hook_bow --fc_npl 100       2>&1 | tee -a imgs/logs/fc_11.txt
!python -m main --model fc --epochs 100 --lr 1e-4 --wd 5e-5 --bs 32 --n_runs 10 --espat 15 --fcsvm_hook_custom --fc_npl 100       2>&1 | tee -a imgs/logs/fc_12.txt
!python -m main --model fc --epochs 100 --lr 1e-4 --wd 1e-3 --bs 32 --n_runs 10 --espat 15 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --fc_npl 100       2>&1 | tee -a imgs/logs/fc_13.txt
!python -m main --model fc --epochs 100 --lr 1e-4 --wd 5e-5 --bs 32 --n_runs 10 --espat 15 --fcsvm_hook_w2v --fc_npl 100       2>&1 | tee -a imgs/logs/fc_12.txt

#SVM
python -m main --model svm --n_runs 1 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --svm_c 1 --svm_gamma auto --svm_kernel rbf           2>&1 | tee -a imgs/logs/svm_1.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --svm_c 100 --svm_gamma auto --svm_kernel rbf           2>&1 | tee -a imgs/logs/svm_2.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --svm_c 1000 --svm_gamma auto --svm_kernel rbf           2>&1 | tee -a imgs/logs/svm_3.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_bow --fcsvm_hook_w2v --svm_c 1 --svm_gamma auto --svm_kernel rbf           2>&1 | tee -a imgs/logs/svm_4.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_custom --fcsvm_hook_w2v --svm_c 1 --svm_gamma auto --svm_kernel rbf           2>&1 | tee -a imgs/logs/svm_5.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_custom --fcsvm_hook_bow --svm_c 1 --svm_gamma auto --svm_kernel rbf           2>&1 | tee -a imgs/logs/svm_6.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_custom --svm_c 1 --svm_gamma 1 --svm_kernel rbf           2>&1 | tee -a imgs/logs/svm_7.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_w2v --svm_c 1 --svm_gamma auto --svm_kernel rbf           2>&1 | tee -a imgs/logs/svm_8.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_bow --svm_c 1 --svm_gamma auto --svm_kernel rbf           2>&1 | tee -a imgs/logs/svm_9.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --svm_c 100 --svm_gamma 1 --svm_kernel rbf           2>&1 | tee -a imgs/logs/svm_10.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --svm_c 1000 --svm_gamma 1 --svm_kernel rbf           2>&1 | tee -a imgs/logs/svm_11.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --svm_c 1 --svm_gamma 0.001 --svm_kernel rbf           2>&1 | tee -a imgs/logs/svm_12.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --svm_c 100 --svm_gamma 0.001 --svm_kernel rbf           2>&1 | tee -a imgs/logs/svm_13.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --svm_c 1000 --svm_gamma 0.001 --svm_kernel rbf           2>&1 | tee -a imgs/logs/svm_14.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --svm_c 1 --svm_gamma auto --svm_kernel poly           2>&1 | tee -a imgs/logs/svm_15.txt
python -m main --model svm --n_runs 1 --fcsvm_hook_custom --fcsvm_hook_bow --fcsvm_hook_w2v --svm_c 1 --svm_gamma auto --svm_kernel linear           2>&1 | tee -a imgs/logs/svm_16.txt


"""
