import pickle
import pprint

from classifier import CompoundClassifier, FCClassifier, LSTMClassifier, HackyCompoundClassifier
from dataset import load_features, TRAITS, wrap_datasets
from eval import eval
from extractor import *
from rnn_dataset import load_rnn_features, load_embeddings, pad_collate_fn, load_features_2
from utils import setup_torch_reproducibility, setup_torch_device, project_path, get_str_formatted_time, ensure_dir


def welcome():
    print("Personality Trait Classification on Essays")
    print("Text Analysis and Retrieval, 2021")
    print("FER, University of Zagreb, Croatia\n")


if __name__ == '__main__':
    welcome()
    # ~~~~ Config ~~~~ #
    n_runs = 1  # How many times will each model be run, used to calculate std. dev.
    seed = 42  # Initial seed, used to setup reproducibility
    time_str = get_str_formatted_time()  # string time identifier
    test_ratio, val_ratio = 0.2, 0.2
    run_name = f"baselines_s={seed}_n={n_runs}_testratio={test_ratio}"  # The run name is used for logging
    save_dir = os.path.join(project_path, f"imgs/{run_name}")  # Where should logs be saved
    ensure_dir(save_dir)

    # ~~~~ Setup ~~~~ #
    device = setup_torch_device()
    setup_torch_reproducibility(seed)
    batch_size = 32

    use_lstm = True
    augmented = False

    if use_lstm:
        # ~~~~ LSTM setup ~~~~ #
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

        rnn_init = {'rnn_layers': 1, 'bidirectional': False, 'activation_fn': torch.nn.ReLU,
                    'device': device, 'clip': 20}

        if extract_cfg['s2v']:
            in_dim = 600 if extract_cfg['wiki'] else 700
        else:
            in_dim = 300
            print("Load embeddings")
            embeddings = load_embeddings(vocab, **extract_cfg)
            rnn_init['embeddings'] = embeddings
            print("Embeddings loaded")

        rnn_init['rnn_dims'] = [in_dim, 200]
        # rnn_init['fc_hidden'] = [512, 128, 1]

        clf_hook = lambda: CompoundClassifier([
            (LSTMClassifier, rnn_init), (LSTMClassifier, rnn_init),
            (LSTMClassifier, rnn_init), (LSTMClassifier, rnn_init),
            (LSTMClassifier, rnn_init)
        ])
        clf_short_name = "LSTM-LSTM-LSTM-LSTM-LSTM(TEST)"
    # elif augmented:
    #     # ~~~~ LSTM setup ~~~~ #
    #
    #     extract_cfg = {
    #         'valid_ratio': val_ratio, 'test_ratio': test_ratio,
    #         'device': device,
    #         "min_freq": 2,
    #         "max_size": -1,
    #         'w2v_limit': None,
    #         'emotion_drop': True
    #     }
    #     (train, train_auth), (val, val_auth), (tv, tv_auth), (test, test_auth), vocab = load_features_2(**extract_cfg)
    #     print("Wrapping datasets into loaders...", end=' ')
    #     train, val, tv, test = wrap_datasets(batch_size, pad_collate_fn, train, val, tv, test)
    #     print("DONE")
    #     train, valid, trainval, test = (train, train_auth), (val, val_auth), (tv, tv_auth), (test, test_auth)
    #
    #     rnn_init = {'rnn_layers': 1, 'bidirectional': False, 'activation_fn': torch.nn.ReLU,
    #                 'device': device, 'clip': 2.0}
    #
    #     in_dim = 300
    #     print("Loading W2V embeddings...", end=' ')
    #     embeddings = load_embeddings(vocab, **extract_cfg)
    #     print("DONE")
    #     rnn_init['embeddings'] = embeddings
    #
    #     rnn_init['rnn_dims'] = [in_dim, 512]
    #     rnn_init['fc_hidden'] = [512, 128, 1]
    #
    #     clf_hook = lambda: HackyCompoundClassifier([
    #         (LSTMClassifier, rnn_init), (LSTMClassifier, rnn_init),
    #         (LSTMClassifier, rnn_init), (LSTMClassifier, rnn_init),
    #         (LSTMClassifier, rnn_init)
    #     ])
    #     clf_short_name = "LSTM-LSTM-LSTM-LSTM-LSTM"
    # else:
    #     # ~~~~ FC & SVM setup ~~~~ #
    #
    #     extract_cfg = {
    #         'valid_ratio': val_ratio, 'test_ratio': test_ratio,
    #         'device': device,
    #         'wiki': True,
    #         'w2v_limit': 2500000
    #     }
    #     ext_hooks = (BOWExtractor,)
    #     train, valid, trainval, test = load_features(ext_hooks, **extract_cfg)
    #     in_dim = len(train[0][0])
    #     train, valid, trainval, test = wrap_datasets(batch_size, None, train, valid, trainval, test)
    #
    #     fc_init = {"neurons_per_layer": [in_dim, 100, 1], "activation_module": torch.nn.ReLU, "device": device}
    #     svm_init = {"c": 1, "gamma": "auto", "decision_function_shape": "ovo", "kernel": "rbf", "in_dim": in_dim}
    #
    #     clf_hook = lambda: CompoundClassifier([
    #         (FCClassifier, fc_init), (FCClassifier, fc_init),
    #         (FCClassifier, fc_init), (FCClassifier, fc_init),
    #         (FCClassifier, fc_init)
    #     ])
    #     clf_short_name = "FC-FC-FC-FC-FC(BOW)"

    train_args = {"epochs": 100, "lr": 1e-3, "wd": 5e-7, "es_patience": 30, "es_epsilon": 1e-7, "es_maxiter": 30,
                  "device": device, "debug_print": True}

    data_loaders = (train, valid, trainval, test)

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
    results_csv = os.path.join(save_dir, f"results_{clf_short_name}_{time_str}.csv")
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
