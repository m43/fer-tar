import os
import re

import torch

from utils import project_path

DS_PATH = os.path.join(project_path, 'dataset/essays.csv')
TRAITS = ['ext', 'neu', 'agr', 'con', 'opn']

R_ID = r'\d{4}_\d+\.txt'
R_TEXT = r'"?.+"?'
R_BIG5 = r'[n|y]'
REGEX_ROW = f"[^#]?({R_ID}),({R_TEXT}),({R_BIG5}),({R_BIG5}),({R_BIG5}),({R_BIG5}),({R_BIG5})"
PATTERN_ROW = re.compile(REGEX_ROW)


def load_dataset(text_preprocessing_fn = None):
    """
    Function for loading dataset from .csv file. It reads the .csv file and parses it, thus creating a list of
    all file entries. A entry consists of 7 attributes which are: author, text and bool flags for extroversion,
    neuroticism, agreeableness, conscientiousness, openness.

    :return: (x, y); (list[string], torch.tensor)
    """
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
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


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
