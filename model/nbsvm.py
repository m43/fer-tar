import os
import pprint
import re
from collections import Counter

import numpy as np
import torch

from dataset import split_dataset, TRAITS, DS_PATH
from utils import project_path, ensure_dir, get_str_formatted_time, setup_torch_reproducibility


def tokenize(sentence, grams):
    words = sentence.split()
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i:i + gram])]
    return tokens


def build_dict(x, grams):
    dic = Counter()
    for sentence in x:
        dic.update(tokenize(sentence, grams))
    return dic


def process_files(pos, neg, dic, r, outfn, grams):
    output = []
    for beg_line, x in zip(["1", "-1"], [pos, neg]):
        for l in x:
            tokens = tokenize(l, grams)
            indexes = []
            for t in tokens:
                try:
                    indexes += [dic[t]]
                except KeyError:
                    pass
            indexes = list(set(indexes))
            indexes.sort()
            line = [beg_line]
            for i in indexes:
                line += ["%i:%f" % (i + 1, r[i])]
            output += [" ".join(line)]
    output = "\n".join(output)
    f = open(outfn, "w")
    f.writelines(output)
    f.close()


def compute_ratio(poscounts, negcounts, alpha=1):
    alltokens = list(set(list(poscounts.keys()) + list(negcounts.keys())))
    dic = dict((t, i) for i, t in enumerate(alltokens))
    d = len(dic)
    print("computing r...")
    p, q = np.ones(d) * alpha, np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]
    p /= abs(p).sum()
    q /= abs(q).sum()
    r = np.log(p / q)
    return dic, r


def load_dataset(text_preprocessing_fn=None):
    R_ID = r'\d{4}_\d+\.txt'
    R_TEXT = r'"?.+"?'
    R_BIG5 = r'[n|y]'
    REGEX_ROW = f"[^#]?({R_ID}),({R_TEXT}),({R_BIG5}),({R_BIG5}),({R_BIG5}),({R_BIG5}),({R_BIG5})"
    PATTERN_ROW = re.compile(REGEX_ROW)

    dataset = []
    # loading does not work with 'utf8' for some reason
    with open(DS_PATH, 'r', encoding='cp1252') as essays:
        while True:
            # for large files, line by line is better than reading all lines
            line = essays.readline()
            if not line:
                break

            match = PATTERN_ROW.match(line)
            if match is None:
                continue

            groups = match.groups()
            author, text = groups[0], groups[1]
            c_ext, c_neu, c_agr, c_con, c_opn = [g == 'y' for g in groups[2:]]
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            if text_preprocessing_fn is not None:
                text = text_preprocessing_fn(text)
            dataset.append([author, text, c_ext, c_neu, c_agr, c_con, c_opn])

    x = [line[1] for line in dataset]
    y = torch.tensor([line[2:] for line in dataset], dtype=torch.float32)
    return x, y


def main(ngram):
    seed = 42
    setup_torch_reproducibility(seed=seed)

    # Preprocessing -- turn to lower case and add spaces around all non-alphanumeric characters (ex. .,":!?)
    preprocess = lambda t: "".join([c if c.isalnum() or c == " " else " " + c + " " for c in t.lower()])
    x, y = load_dataset(preprocess)
    # (train_x, train_y_all), _, (test_x, test_y_all) = split_dataset(x, y, valid_ratio=0, test_ratio=0.2)
    (train_x, train_y_all), (valx, valy), (test_x, test_y_all) = split_dataset(x, y, test_ratio=0.2, valid_ratio=0.2)
    train_x, train_y_all = train_x + valx, torch.vstack([train_y_all, valy])

    # os.system("./playground/install_liblinear.sh")
    liblinear = os.path.join(project_path, "build/liblinear-1.96")
    # ngram = "1234"  # N-grams considered e.g. 123 is uni+bi+tri-grams
    logs_dir = os.path.join(project_path, f"logs/nbsvm/ngram={ngram}_{get_str_formatted_time()}")
    ensure_dir(logs_dir)

    out_all = os.path.join(logs_dir, f"out.txt")
    results = {}
    # results = {trait: {} for trait in TRAITS}
    for i, trait in zip(range(5), TRAITS):
        print(f"[{trait.upper()}]")
        train_y, test_y = train_y_all[:, i], test_y_all[:, i]
        out = os.path.join(logs_dir, f"out_{i}_{trait}.txt")

        def x_to_pos_neg(x, y):
            x_pos, x_neg = [], []
            for i in range(len(x)):
                if y[i] == 1:
                    x_pos.append(x[i])
                else:
                    x_neg.append(x[i])
            return x_pos, x_neg

        train_x_pos, train_x_neg = x_to_pos_neg(train_x, train_y)
        test_x_pos, test_x_neg = x_to_pos_neg(test_x, test_y)
        train_y = torch.tensor([1.] * len(train_x_pos) + [0.] * len(train_x_neg))
        test_y = torch.tensor([1.] * len(test_x_pos) + [0.] * len(test_x_neg))
        print(f"len(train_x_pos)={len(train_x_pos)}   len(train_x_neg)={len(train_x_neg)}")
        print(f"len(test_x_pos)={len(test_x_pos)}   len(test_x_neg)={len(test_x_neg)}")

        ngram = [int(i) for i in ngram]
        print("counting...")
        poscounts = build_dict(train_x_pos, ngram)
        negcounts = build_dict(train_x_neg, ngram)
        print(f"len(poscounts)-->{len(poscounts)}   len(negcounts)-->{len(negcounts)}")

        dic, r = compute_ratio(poscounts, negcounts)
        print("processing files...")

        train_nbsvm_txt = os.path.join(logs_dir, f"train-nbsvm-{i}-{trait}.txt")
        test_nbsvm_txt = os.path.join(logs_dir, f"test-nbsvm-{i}-{trait}.txt")
        model_logreg_path = os.path.join(logs_dir, f"model.logreg")

        process_files(train_x_pos, train_x_neg, dic, r, train_nbsvm_txt, ngram)
        process_files(test_x_pos, test_x_neg, dic, r, test_nbsvm_txt, ngram)

        trainsvm = os.path.join(liblinear, "train")
        predictsvm = os.path.join(liblinear, "predict")
        os.system(f"{trainsvm} -s 0 {train_nbsvm_txt} {model_logreg_path}")
        print("training done")

        def ma_eval(subset_nbsvm_txt, y):
            os.system(f"{predictsvm} -b 1 {subset_nbsvm_txt} {model_logreg_path} {out}")

            with open(out, "r") as f:
                predictions = torch.tensor([1. if float(line.split()[0]) == 1 else 0. for line in f.readlines()[1:]])

            tp = int(sum(np.logical_and(predictions == y, y == 1)))
            fn = int(sum(np.logical_and(predictions != y, y == 1)))
            tn = int(sum(np.logical_and(predictions == y, y == 0)))
            fp = int(sum(np.logical_and(predictions != y, y == 0)))

            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            accuracy = (tp + tn) / (tp + fn + tn + fp)
            f1 = (2 * precision * recall) / (precision + recall)
            r = {"acc": accuracy, "pre": precision, "rec": recall, "f1": f1}
            return r

        for subset_name, subset_txt, y in [("train", train_nbsvm_txt, train_y), ("test", test_nbsvm_txt, test_y)]:
            if subset_name not in results:
                results[subset_name] = {}
            eval_results = ma_eval(subset_txt, y)
            if trait not in results[subset_name]:
                results[subset_name][trait] = {}
            for metric_name in eval_results:
                r = results[subset_name][trait].get(metric_name, [])
                e = eval_results[metric_name]
                results[subset_name][trait][metric_name] = r + [e]

        # results["ACC"] = results.get("ACC", []) + [accuracy]
        # results["PRC"] = results.get("PRC", []) + [precision]
        # results["REC"] = results.get("REC", []) + [recall]
        # results["F1"] = results.get("F1", []) + [f1]

        # metrics = f"[{trait.upper()}]\n"
        # for measure, values in results.items():
        #     metrics += f"{measure:<3s}:{values[-1]}\n"
        # print(metrics)
        # with open(out_all, "a") as f:
        #     f.write(metrics)

    # metrics = f"\n[AVERAGE]\n"
    # for measure, values in results.items():
    #     metrics += f"AVG_{measure}:{sum(values) / len(values):.5f}\n"
    # print(metrics)
    # with open(out_all, "a") as f:
    #     f.write(metrics)

    time_str = get_str_formatted_time()  # string time identifier
    run_name = f"nbsvm_seed={seed}_ngram={ngram}"
    save_dir = os.path.join(project_path, f"imgs/{run_name}")  # Where should logs be saved
    ensure_dir(save_dir)

    metric_names = results["test"][TRAITS[0]]
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
    results_csv = os.path.join(save_dir, f"results_{ngram}_{time_str}.csv")
    # CSV
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
    ngrams_to_try = ["1", "12", "123", "1234"]
    # ngrams_to_try = ["1", "12", "123"]
    # ngrams_to_try = ["1234"]
    for ngram in ngrams_to_try:
        main(ngram)
