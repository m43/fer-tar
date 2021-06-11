import classifier
import dataset
import re
import csv
import utils
import torch
from torch.utils.data import TensorDataset, DataLoader
from nltk import word_tokenize, sent_tokenize, WordNetLemmatizer
from extractor import *

import matplotlib.pyplot as plt

# def plot(y, label):
#
#     x = [i for i in range(len(y))]
#     figure, axis = plt.subplots(1, 1, figsize=(10, 5), dpi=80)
#     plt.xlim(xmin=x[0]-10, xmax=x[-1]+10)
#
#     axis.scatter(x, y, marker='o', color='k')
#
#     axis.set_ylabel(f"Length WRT {label}")
#     axis.set_xlabel(f"Essays (sorted by length WRT {label})")
#     axis.xaxis.labelpad = 10
#     axis.yaxis.labelpad = 10
#     plt.tight_layout()
#     plt.show()
#
# x, y = dataset.load_dataset()
# x = x[1:]
#
# RE_WSPACE = f'\\s+'
#
# def load_emotional_words():
#     emotional_words = set()
#     with open("emotions/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", 'r', encoding='cp1252') as em_path:
#         while True:
#             line = em_path.readline().strip()
#             if not line:
#                 break
#             parts = re.split(RE_WSPACE, line)
#             word, flag = parts[0], int(parts[2])
#             if flag == 1:
#                 emotional_words.add(word)
#     return emotional_words
#
#
# def emotionally_neutral_drop(x):
#     e_words = load_emotional_words()
#     lemmatizer = WordNetLemmatizer()
#     x_dropped = []
#     for tokens in x:
#         tokens_dropped = []
#         for t in tokens:
#             lemma = lemmatizer.lemmatize(t)
#             if lemma in e_words:
#                 tokens_dropped.append(t)
#         x_dropped.append(tokens_dropped)
#     return x_dropped
#
#
# x_tok = sorted([[t for t in word_tokenize(xi.lower()) if t not in ".,?!"] for xi in x], key=lambda l: len(l))
# x_drop = emotionally_neutral_drop(x_tok)
# tok_lens = [len(xi) for xi in x_tok]
# drop_lens = [len(xi) for xi in x_drop]
# ratio = [drop_lens[i] / tok_lens[i] for i in range(len(tok_lens))]
# avg_ratio = sum(ratio) / len(ratio)
#
# min_ratio_index = 0
# min_ratio_value = ratio[0]
# for i in range(len(ratio)):
#     if ratio[i] < min_ratio_value:
#         min_ratio_index = i
#         min_ratio_value = ratio[i]
#
# print(x_tok[min_ratio_index])

pass

# char_counts = sorted([len(xi) for xi in x])
# word_counts = sorted([len(xi.split()) for xi in x])
# token_counts = sorted([len([t for t in word_tokenize(xi) if t not in ".,?!"]) for xi in x])
# sent_counts = sorted([len(sent_tokenize(xi)) for xi in x])
#
# plot(char_counts, 'characters')
# plot(word_counts, 'words')
# plot(token_counts, 'tokens')
# plot(sent_counts, 'sentences')
#
# print("OUT")

def load_dataset(path):
    """
    Function for loading dataset from .csv file. It reads the .csv file and parses it, thus creating a list of
    all file entries. A entry consists of 7 attributes which are: author, text and bool flags for extroversion,
    neuroticism, agreeableness, conscientiousness, openness.

    :return: (x, y); (list[string], torch.tensor)
    """
    dataset = []
    with open(path, 'r', encoding='cp1252') as essays:
        dsreader = csv.reader(essays, delimiter=',', quotechar='"')
        for row in dsreader:
            if row[0].startswith("#"):
                continue
            dataset_row = [(row[i] if i < 2 else (1. if row[i] == 'y' else 0.)) for i in range(len(row))]
            dataset.append(dataset_row)

    x = [line[1] for line in dataset]
    y = torch.tensor([line[2:] for line in dataset], dtype=torch.float32)
    return x, y


def split_dataset(x, y, test_ratio=0.2, valid_ratio=0.2):
    """
    Function for randomly splitting the dataset into train, valid and test sets with given ratios.
    The dataset is shuffled randomly. If valid_ratio == 0.0, an empty valid subset will be returned.

    :param x: list of essays; list[string]
    :param y: torch tensor with targets; torch.tensor(n,5)
    :param test_ratio: the ratio of datapoints in the test subset after the split; float
    :param valid_ratio: the ratio of datapoints in the valid subset after the split; float
    :return: the subsets like (train_x, train_y), (val_x, val_y), (test_x, test_y);
             tuple(tuple(list[string],torch.tensor))
    """
    shuffle_indices = torch.randperm(y.shape[0])
    x = [x[i] for i in shuffle_indices]
    y = y[shuffle_indices]

    n = len(x)
    n_val, n_test = int(valid_ratio * n), int(test_ratio * n)
    n_train = n - n_val - n_test

    train_x, val_x, test_x = x[0:n_train], x[n_train:n_train + n_val], x[n_train + n_val:]
    train_y, val_y, test_y = y[0:n_train], y[n_train:n_train + n_val], y[n_train + n_val:]
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def load_features(extractor_hooks, x=None, y=None, **kwargs):
    print("Creating train/valid/test splits...", end=' ')
    (trnx, trny), (_, _), (_, _) = split_dataset(x, y, test_ratio=kwargs['test_ratio'],
                                                       valid_ratio=kwargs['valid_ratio'])
    print("DONE")

    print("Tokenizing essays into sentence-level tokens...", end=' ')
    sen_trnx = [sent_tokenize(xi.lower()) for xi in trnx]
    print("DONE")
    print("Tokenizing essays into word-level tokens...", end=' ')
    tok_trnx = [word_tokenize(xi.lower()) for xi in trnx]
    print("DONE")

    kwargs['train_raw'] = trnx
    kwargs['train_tok'] = tok_trnx
    kwargs['train_sen'] = sen_trnx

    print("Initializing extractors...")
    extractors = [hook(**kwargs) for hook in extractor_hooks]
    print("DONE")

    x_our, y_our = load_dataset('dataset/fml.csv')
    tok_our = [word_tokenize(xi.lower()) for xi in x_our]
    sen_our = [sent_tokenize(xi.lower()) for xi in x_our]

    print("Extracting features...")
    our_feats = torch.cat([e.extract(x_our, tok_our, sen_our) for e in extractors], dim=1)
    print("DONE")

    our_ds = TensorDataset(our_feats, y_our)
    return DataLoader(dataset=our_ds, batch_size=1, shuffle=False)


def demo():
    utils.setup_torch_reproducibility(42)
    ext_hooks = (BOWExtractor, W2VExtractor, CapitalizationExtractor, InterpunctionExtractor,
                 RepeatingLettersExtractor, WordCountExtractor)
    trnx, trny = load_dataset("dataset/essays.csv")
    our_loader = load_features(ext_hooks, trnx, trny, test_ratio=0.2, valid_ratio=0.2, w2v_limit=500000)

    kwargs = {
        'c': 1, 'gamma': 'auto', 'decision_function_shape': 'ovo', 'kernel': 'rbf',
        'activation_module': torch.nn.ReLU, 'device': utils.setup_torch_device()
    }
    clf = classifier.CompoundClassifier([])
    clf.load("SVM-FC-SVM-FC-FC(BOW,W2V,CUSTOM)", **kwargs)

    classification, true = clf.classify(our_loader)
    print(classification)


if __name__ == '__main__':
    demo()


