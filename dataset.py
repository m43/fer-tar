import csv
import os

import torch
from nltk.tokenize import word_tokenize, sent_tokenize
from torch.utils.data import TensorDataset, DataLoader

from utils import project_path

DS_PATH = os.path.join(project_path, 'dataset/essays.csv')
TRAITS = ['ext', 'neu', 'agr', 'con', 'opn']


def load_dataset(text_preprocessing_fn=None, max_essays=None):
    """
    Function for loading dataset from .csv file. It reads the .csv file and parses it, thus creating a list of
    all file entries. A entry consists of 7 attributes which are: author, text and bool flags for extroversion,
    neuroticism, agreeableness, conscientiousness, openness.

    :return: (x, y); (list[string], torch.tensor)
    """
    dataset = []
    with open(DS_PATH, 'r', encoding='cp1252') as essays:
        dsreader = csv.reader(essays, delimiter=',', quotechar='"')
        for row in dsreader:
            if row[0].startswith("#"):
                continue
            dataset_row = [(row[i] if i < 2 else (1. if row[i] == 'y' else 0.)) for i in range(len(row))]
            dataset.append(dataset_row)
            if max_essays is not None and len(dataset) == max_essays:
                break

    print(f"LOADED DATASET OF {len(dataset)} ESSAYS")

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
    _, x = zip(*sorted(zip(shuffle_indices, x)))
    y = y[shuffle_indices]

    n = len(x)
    n_val, n_test = int(valid_ratio * n), int(test_ratio * n)
    n_train = n - n_val - n_test

    train_x, val_x, test_x = x[0:n_train], x[n_train:n_train + n_val], x[n_train + n_val:]
    train_y, val_y, test_y = y[0:n_train], y[n_train:n_train + n_val], y[n_train + n_val:]
    # assert train_x[0].startswith("I am hoping that I will be able to keep up with my thoughts for twenty minutes") or \
    #        train_x[0].startswith("i am hoping that i will be able to keep up with my thoughts for twenty minutes .  it")
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def load_features(extractor_hooks, x=None, y=None, **kwargs):
    if x is None and y is None:
        print("Loading dataset from CSV file...", end=' ')
        x, y = load_dataset()
        print("DONE")

    print("Creating train/valid/test splits...", end=' ')
    (trnx, trny), (valx, valy), (tesx, tesy) = split_dataset(x, y, test_ratio=kwargs['test_ratio'],
                                                             valid_ratio=kwargs['valid_ratio'])
    print("DONE")

    print("Tokenizing essays into sentence-level tokens...", end=' ')
    sen_trnx = [sent_tokenize(xi.lower()) for xi in trnx]
    sen_valx = [sent_tokenize(xi.lower()) for xi in valx]
    sen_tesx = [sent_tokenize(xi.lower()) for xi in tesx]
    print("DONE")
    print("Tokenizing essays into word-level tokens...", end=' ')
    tok_trnx = [word_tokenize(xi.lower()) for xi in trnx]
    tok_valx = [word_tokenize(xi.lower()) for xi in valx]
    tok_tesx = [word_tokenize(xi.lower()) for xi in tesx]
    print("DONE")

    kwargs['train_raw'] = trnx
    kwargs['train_tok'] = tok_trnx
    kwargs['train_sen'] = sen_trnx

    print("Initializing extractors...")
    extractors = [hook(**kwargs) for hook in extractor_hooks]
    print("DONE")

    print("Extracting features...")
    trn_feats = torch.cat([e.extract(trnx, tok_trnx, sen_trnx) for e in extractors], dim=1)
    val_feats = torch.cat([e.extract(valx, tok_valx, sen_valx) for e in extractors], dim=1)
    tes_feats = torch.cat([e.extract(tesx, tok_tesx, sen_tesx) for e in extractors], dim=1)
    print("DONE")

    train_ds = TensorDataset(trn_feats, trny)
    valid_ds = TensorDataset(val_feats, valy)
    trainval_ds = TensorDataset(torch.cat((trn_feats, val_feats), dim=0), torch.cat((trny, valy), dim=0))
    test_ds = TensorDataset(tes_feats, tesy)

    return train_ds, valid_ds, trainval_ds, test_ds


def wrap_datasets(batch_size, collate_fn=None, *datasets):
    return [DataLoader(dataset=d, batch_size=batch_size, collate_fn=collate_fn, shuffle=True) for d in datasets]


if __name__ == '__main__':
    x, y = load_dataset()

    assert len(x) == len(y) == 2467
    assert len(y[0]) == 5
    assert (y[0] == torch.tensor([0., 1., 1., 0., 1.])).all()

    n = len(x)
    first_datum = x[0], y[0]
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_dataset(x, y, 0.2, 0)
    assert first_datum[0] == x[0] and (first_datum[1] == y[0]).all()  # assert that original dataset was not shuffled
    assert len(test_x) == int(0.2 * len(x))
    assert len(val_x) == 0
    assert n == len(train_x) + len(val_x) + len(test_x) == len(train_y) + len(val_y) + len(test_y)

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_dataset(x, y, 0.3, 0.3)
    assert len(test_x) == int(0.3 * len(x))
    assert len(val_x) == int(0.3 * len(x))
    assert n == len(train_x) + len(val_x) + len(test_x) == len(train_y) + len(val_y) + len(test_y)
