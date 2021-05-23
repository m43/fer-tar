import os
import pprint

import numpy as np

import classifier
from extractor import *
from dataset import load_features, TRAITS, wrap_datasets
from eval import eval
from utils import setup_torch_reproducibility, setup_torch_device, project_path, get_str_formatted_time, ensure_dir

if __name__ == '__main__':
    print("Personality Trait Classification on Essays")
    print("Text Analysis and Retrieval, 2021")
    print("FER, University of Zagreb, Croatia")

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
    classifiers = []
    extract_cfg = {
        'valid_ratio': val_ratio, 'test_ratio': test_ratio,
        'device': device,
        'bow_fit_raw': True,
        's2v_fit_raw': True, 'wiki': False,
        'w2v_limit': 1000000
    }
    ext_hooks = (InterpunctionExtractor, RepeatingLettersExtractor, CapitalizationExtractor, WordCountExtractor)
    fc_args = {"epochs": 20, "lr": 1e-4, "batch_size": 16, "wd": 0}

    train, valid, test = load_features(ext_hooks, **extract_cfg)
    train, valid, test = wrap_datasets(fc_args['batch_size'], train, valid, test)

    classifiers += [("FCClassifier", lambda: classifier.FCClassifier(train[0][0].size(0), dev=device), fc_args)]

    # ~~~~ Collect results ~~~~ #
    print("Collecting results")
    results = {clf_name: {} for clf_name, _, _ in classifiers}
    for i in range(n_runs):
        print(f"i::[{i + 1}/{n_runs}] started")
        for clf_name, clf_hook, extra_args in classifiers:
            clf = clf_hook()
            print(f"Classifier: {clf_name}")
            clf.train(train_x,
                      train_y,
                      **extra_args)
            for subset_name, subset in ("train", (train_x, train_y)), ("test", (test_x, test_y)):  # ("val", val)
                if subset_name not in results[clf_name]:
                    results[clf_name][subset_name] = {}
                eval_results = eval(clf, *subset)
                for trait_name in TRAITS:
                    if trait_name not in results[clf_name][subset_name]:
                        results[clf_name][subset_name][trait_name] = {}
                    for metric_name in eval_results[trait_name]:
                        r = results[clf_name][subset_name][trait_name].get(metric_name, [])
                        e = eval_results[trait_name][metric_name]
                        results[clf_name][subset_name][trait_name][metric_name] = r + [e]
        print(f"i::[{i + 1}/{n_runs}] current results: {results}\n\n")

    metric_names = results[clf_name][subset_name][TRAITS[0]]
    for clf_name, clf, _ in classifiers:
        for subset_name in results[clf_name]:
            for trait_name in TRAITS:
                for metric_name in metric_names:
                    r = results[clf_name][subset_name][trait_name][metric_name]
                    mean, std = np.mean(np.array(r, np.float64)), np.std(np.array(r, np.float64))
                    results[clf_name][subset_name][trait_name][metric_name] = (mean, std)

    print("Results collected")
    print()

    # ~~~~ Save results ~~~~ #
    # TXT
    results_txt = os.path.join(save_dir, f"results_{time_str}.txt")
    print(f"Results path: {results_txt}")
    with open(results_txt, "w") as f:
        f.write(f"'{results_txt}'\n{results}")
    # CSV
    results_csv = os.path.join(save_dir, f"results_{time_str}.csv")
    with open(results_csv, 'w') as csv:
        csv.write(f"line_id,subset_name,trait_name,metric_name")
        for clf_name, clf, _ in classifiers:
            csv.write(f",{clf_name}-mean,{clf_name}-std")
        csv.write("\n")
        for subset_name in results[clf_name]:
            for trait_name in TRAITS:
                for metric_name in metric_names:
                    line_id = f"{subset_name}-{trait_name}-{metric_name}"
                    csv.write(f"{line_id},{subset_name},{trait_name},{metric_name}")
                    for clf_name, clf, _ in classifiers:
                        mean, std = results[clf_name][subset_name][trait_name][metric_name]
                        csv.write(f",{mean},{std}")
                    csv.write("\n")
                # csv.write("\n")
            # csv.write("\n")
        # csv.write("\n")

    # ~~~~ Print results ~~~~ #
    pp = pprint.PrettyPrinter(width=100, compact=True)
    pp.pprint(results)

    with open(results_csv, "r") as csv:
        print(csv.read())

    note = f"""
You can go ahead and open the csv file in your favorite
spreadsheet editor. My favorite is Google Sheets, as all
of us can collaborate, paste local results and visualize
them. I open the csv in LibreOffice Calc, copy the content
and paste it into Google Sheets. Here is a google sheets
document that I've shared with you: https://cutt.ly/wbUjfdt

Local csv path:
{results_csv}
Local txt path:
{results_txt}"""
    print(note)
