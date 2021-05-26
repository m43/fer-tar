import re
from dataclasses import astuple, dataclass

import gensim
import numpy as np
import torch
from nltk import word_tokenize
from torch.nn import Embedding
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

from dataset import load_dataset
from extractor import W2V_GOOGLE_NEWS_PATH

PAD = "<PAD>"
UNK = "<UNK>"

RE_WSPACE = f'\\s+'
RE_DELIM = f', '


def load_embeddings(vocab, **kwargs):
    embeddings = [torch.zeros(300), torch.randn(300)]  # padding embedding and unknown embedding

    model = gensim.models.KeyedVectors.load_word2vec_format(W2V_GOOGLE_NEWS_PATH,
                                                            binary=True,
                                                            limit=kwargs['w2v_limit'])

    for i in range(2, len(vocab.itos)):
        word = vocab.itos[i]
        if word in model.vocab:
            embeddings.append(torch.tensor(model[word]))

    return Embedding.from_pretrained(torch.stack(embeddings), padding_idx=0)


@dataclass()
class Instance:
    x: list
    y: list

    def __init__(self, x, y):
        self.x = word_tokenize(x.lower())
        self.y = y


class NLPDataset(Dataset):
    def __init__(self, x, y, vocabulary):
        self.txt_vocab = vocabulary
        self.instances = []
        for x_i, y_i in zip(x, y):
            self.instances.append(astuple(Instance(x_i, y_i)))
        self.len = len(self.instances)

    def __getitem__(self, index) -> T_co:
        if index < 0 or index >= self.len:
            raise IndexError(f"Invalid index {index} for array of len {self.len}")
        x, y = self.instances[index]
        return self.txt_vocab.encode(x), y

    def __len__(self):
        return self.len


def extract_frequencies(x):
    x_frequencies = {}

    for text in x:

        words = word_tokenize(text.lower())

        for w in words:
            x_count = 1 if x_frequencies.get(w) is None else x_frequencies.get(w)
            x_frequencies[w] = x_count + 1

    return sorted(x_frequencies.items(), key=lambda a: a[1], reverse=True)


class Vocab:
    def __init__(self, frequencies, max_size=-1, min_freq=0):
        self.itos = [PAD, UNK]
        self.stoi = {PAD: 0, UNK: 1}

        i = 2
        for key, value in frequencies:
            if max_size != -1 and (i + 1) >= max_size or value < min_freq:
                break
            self.stoi[key] = i
            self.itos.append(key)
            i += 1

        self.itos = np.array(self.itos)

    def encode(self, words):
        res = []
        if type(words) is list:
            for w in words:
                ind = self.stoi.get(w)
                if ind is None:
                    res.append(self.stoi[UNK])
                else:
                    res.append(ind)
        else:
            ind = self.stoi.get(words)
            if ind is not None:
                res.append(self.stoi[UNK])
            else:
                res.append(ind)
        return torch.tensor(res)


def pad_collate_fn(batch, pad_index=0):

    texts, labels = zip(*batch)  # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in texts])  # Needed for later

    max_len = 0
    for t in texts:
        if t.shape[0] > max_len:
            max_len = t.shape[0]
    texts_tensor = torch.empty((len(texts), max_len), dtype=torch.int)
    labels_tensor = torch.hstack(labels).reshape(-1, 1).float()

    for i in range(len(texts)):
        for j in range(len(texts[i])):
            texts_tensor[i, j] = texts[i][j]
        for k in range(len(texts[i]), max_len):
            texts_tensor[i, j] = pad_index

    return texts_tensor, labels_tensor, lengths


if __name__ == "__main__":
    ds_x, ds_y = load_dataset()
    print(len(ds_x))
    vocab = Vocab(extract_frequencies(ds_x), min_freq=2)
    print(len(vocab.itos))
    dataset = NLPDataset(ds_x, ds_y, vocab)
    print(len(dataset))

    embs = load_embeddings(vocab, **{"w2v_limit": None})

    dl = DataLoader(batch_size=16, dataset=dataset, collate_fn=pad_collate_fn)
    text, _, _ = next(iter(dl))
    print(text, text.shape)
