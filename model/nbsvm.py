import os
from collections import Counter

import numpy as np
import torch

from dataset import load_dataset, split_dataset, TRAITS
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


if __name__ == "__main__":
    setup_torch_reproducibility(seed=72)

    # Preprocessing -- turn to lower case and add spaces around all non-alphanumeric characters (ex. .,":!?)
    preprocess = lambda t: "".join([c if c.isalnum() or c == " " else " " + c + " " for c in t.lower()])
    x, y = load_dataset(preprocess)
    (train_x, train_y_all), _, (test_x, test_y_all) = split_dataset(x, y, valid_ratio=0, test_ratio=0.2)

    # os.system("./playground/install_liblinear.sh")
    liblinear = os.path.join(project_path, "build/liblinear-1.96")
    ngram = "123"  # N-grams considered e.g. 123 is uni+bi+tri-grams
    logs_dir = os.path.join(project_path, f"logs/nbsvm/ngram={ngram}_{get_str_formatted_time()}")
    ensure_dir(logs_dir)

    out_all = os.path.join(logs_dir, f"out.txt")
    results = {}
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
        os.system(f"{predictsvm} -b 1 {test_nbsvm_txt} {model_logreg_path} {out}")

        with open(out, "r") as f:
            predictions = torch.tensor([float(line.split()[0]) for line in f.readlines()[1:]])
            predictions = (predictions + 1) / 2
        tp = int(sum(np.logical_and(predictions == test_y, test_y == 1)))
        fn = int(sum(np.logical_and(predictions != test_y, test_y == 1)))
        tn = int(sum(np.logical_and(predictions == test_y, test_y == 0)))
        fp = int(sum(np.logical_and(predictions != test_y, test_y == 0)))

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + fn + tn + fp)
        f1 = (2 * precision * recall) / (precision + recall)

        results["ACC"] = results.get("ACC", []) + [accuracy]
        results["PRC"] = results.get("PRC", []) + [precision]
        results["REC"] = results.get("REC", []) + [recall]
        results["F1"] = results.get("F1", []) + [f1]

        metrics = f"[{trait.upper()}]\n"
        for measure, values in results.items():
            metrics += f"{measure:<3s}:{values[-1]}\n"
        print(metrics)
        with open(out_all, "a") as f:
            f.write(metrics)

    metrics = f"\n[AVERAGE]\n"
    for measure, values in results.items():
        metrics += f"AVG_{measure}:{sum(values) / len(values):.5f}\n"
    print(metrics)
    with open(out_all, "a") as f:
        f.write(metrics)
