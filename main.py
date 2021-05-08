import os
import pprint

import numpy as np

import dataset
from baselines import RandomBaseline, MCCBaseline
from dataset import split_dataset, TRAITS
from eval import eval
from utils import setup_torch_reproducibility, setup_torch_device, project_path, get_str_formatted_time, ensure_dir

if __name__ == '__main__':
    print("Personality Trait Classification on Essays")
    print("Text Analysis and Retrieval, 2021")
    print("FER, University of Zagreb, Croatia")

    # ~~~~ Config ~~~~ #
    n_runs = 10  # How many times will each model be run, used to calculate std. dev.
    reload_dataset = False  # Should the dataset be shuffled after each run
    seed = 72  # Initial seed, used to setup reproducibility
    time_str = get_str_formatted_time()  # string time identifier
    test_split_ratio, val_split_ratio = 0.2, 0
    run_name = f"baselines_s={seed}_n={n_runs}_testratio={test_split_ratio}"  # The run name is used for logging
    save_dir = os.path.join(project_path, f"imgs/{run_name}")  # Where should logs be saved
    ensure_dir(save_dir)
    classifiers = []
    classifiers += [("MCCBaseline", MCCBaseline())]
    classifiers += [("RandomBaseline", RandomBaseline())]

    # ~~~~ Setup ~~~~ #
    device = setup_torch_device()
    setup_torch_reproducibility(seed)
    dataset = dataset.load_dataset()
    train, _, test = split_dataset(dataset, test_split_ratio, val_split_ratio)

    # ~~~~ Collect results ~~~~ #
    print("Collecting results")
    results = {clf_name: {} for clf_name, clf in classifiers}
    for i in range(n_runs):
        print(f"i::{i}")
        if reload_dataset:
            train, _, test = split_dataset(dataset, test_split_ratio, val_split_ratio)
        for clf_name, clf in classifiers:
            print(f"Classifier: {clf_name}")
            clf.train(train)  # TODO should the be explicitly reinitialized (weights to be randomly picked again)?
            for subset_name, subset in ("train", train), ("test", test):  # ("val", val)
                if subset_name not in results[clf_name]:
                    results[clf_name][subset_name] = {}
                eval_results = eval(clf, subset)
                for trait_name in TRAITS:
                    if trait_name not in results[clf_name][subset_name]:
                        results[clf_name][subset_name][trait_name] = {}
                    for metric_name in eval_results[trait_name]:
                        r = results[clf_name][subset_name][trait_name].get(metric_name, [])
                        e = eval_results[trait_name][metric_name]
                        results[clf_name][subset_name][trait_name][metric_name] = r + [e]
        print(f"i::{i} current results: {results}")

    metric_names = results[clf_name][subset_name][TRAITS[0]]
    for clf_name, clf in classifiers:
        for subset_name in results[clf_name]:
            for trait_name in TRAITS:
                for metric_name in metric_names:
                    r = results[clf_name][subset_name][trait_name][metric_name]
                    mean, std = np.mean(np.array(r, np.float)), np.std(np.array(r, np.float))
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
        for clf_name, clf in classifiers:
            csv.write(f",{clf_name}-mean,{clf_name}-std")
        csv.write("\n")
        for subset_name in results[clf_name]:
            for trait_name in TRAITS:
                for metric_name in metric_names:
                    line_id = f"{subset_name}-{trait_name}-{metric_name}"
                    csv.write(f"{line_id},{subset_name},{trait_name},{metric_name}")
                    for clf_name, clf in classifiers:
                        mean, std = results[clf_name][subset_name][trait_name][metric_name]
                        csv.write(f",{mean},{std}")
                    csv.write("\n")
                # csv.write("\n")
            csv.write("\n")
        csv.write("\n\n")

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
document that I shared with you: https://cutt.ly/wbUjfdt
Local csv path:
{results_csv}
Local txt path:
{results_txt}"""
    print(note)
