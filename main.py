import pprint

from classifier import CompoundClassifier, FCClassifier, SVMClassifier
from extractor import *
from dataset import load_features, TRAITS, wrap_datasets
from eval import eval
from utils import setup_torch_reproducibility, setup_torch_device, project_path, get_str_formatted_time, ensure_dir


def welcome():
    print("Personality Trait Classification on Essays")
    print("Text Analysis and Retrieval, 2021")
    print("FER, University of Zagreb, Croatia\n")


if __name__ == '__main__':
    welcome()
    # ~~~~ Config ~~~~ #
    n_runs = 5  # How many times will each model be run, used to calculate std. dev.
    seed = 72  # Initial seed, used to setup reproducibility
    time_str = get_str_formatted_time()  # string time identifier
    test_ratio, val_ratio = 0.2, 0.2
    run_name = f"baselines_s={seed}_n={n_runs}_testratio={test_ratio}"  # The run name is used for logging
    save_dir = os.path.join(project_path, f"imgs/{run_name}")  # Where should logs be saved
    ensure_dir(save_dir)

    # ~~~~ Setup ~~~~ #
    device = setup_torch_device()
    setup_torch_reproducibility(seed)
    batch_size = 16
    extract_cfg = {
        'valid_ratio': val_ratio, 'test_ratio': test_ratio,
        'device': device,
        'bow_fit_raw': True,
        's2v_fit_raw': True, 'wiki': False,
        'w2v_limit': 1000000
    }
    ext_hooks = (InterpunctionExtractor, RepeatingLettersExtractor, CapitalizationExtractor, WordCountExtractor)
    train, valid, test = load_features(ext_hooks, **extract_cfg)
    in_dim = len(train[0][0])
    train, valid, test = wrap_datasets(batch_size, train, valid, test)

    fc_init = {"neurons_per_layer": [in_dim, 300, 100, 1], "activation_module": torch.nn.ReLU, "device": device}
    svm_init = {"c": 1, "gamma": "auto", "decision_function_shape": "ovo", "kernel": "rbf", "in_dim":in_dim}

    clf_hook = lambda: CompoundClassifier([
        (FCClassifier, fc_init), (SVMClassifier, svm_init),
        (FCClassifier, fc_init), (FCClassifier, fc_init),
        (FCClassifier, fc_init)
    ])
    clf_short_name = "FC-SVM-FC-FC-FC"

    train_args = {"epochs": 20, "lr": 1e-4, "wd": 0, "early_stopping_iters": 5, "early_stopping_epsilon": 1e-7,
                  "device": device, "debug_print": True}

    # ~~~~ Collect results ~~~~ #
    print("\nCollecting results")
    results = {}
    for i in range(n_runs):
        clf = clf_hook()
        print(f"i::[{i + 1}/{n_runs}] started")
        print("Using classifier " + str(clf))

        clf.train((train, valid, test), **train_args)

        for subset_name, subset in [("train", train), ("test", test)]:
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
