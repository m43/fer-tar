import re
from dataclasses import astuple, dataclass
import os

from utils import project_path
import gensim
import sent2vec
import numpy as np
import torch
from nltk import word_tokenize, sent_tokenize
from torch.nn import Embedding
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

from dataset import load_dataset, split_dataset
from extractor import W2V_GOOGLE_NEWS_PATH

PAD = "<PAD>"
UNK = "<UNK>"

RE_WSPACE = f'\\s+'
RE_DELIM = f', '

# Sent2vec Pre-Trained Models from https://github.com/epfml/sent2vec/)
S2V_WIKI_UNGIRAMS_PATH = os.path.join(project_path, "saved/s2v/wiki_unigrams.bin")
S2V_TORONTO_UNIGRAMS_PATH = os.path.join(project_path, "saved/s2v/torontobooks_unigrams.bin")


def load_embeddings(vocab, **kwargs):
    """
    Function used for loading pretrained word2vec embeddings.

    :param vocab: Vocabulary containing known words whose embeddings are to be loaded.
    :param kwargs: Additional arguments.
    :return: embedding layer: Embedding
    """
    embeddings = [torch.zeros(300), torch.randn(300)]  # padding embedding and unknown embedding

    model = gensim.models.KeyedVectors.load_word2vec_format(W2V_GOOGLE_NEWS_PATH,
                                                            binary=True,
                                                            limit=kwargs['w2v_limit'])

    for i in range(2, len(vocab.itos)):
        word = vocab.itos[i]
        if word in model.vocab:
            embeddings.append(torch.tensor(model[word]))
        else:
            embeddings.append(torch.randn(300))

    return Embedding.from_pretrained(torch.stack(embeddings), padding_idx=0)


@dataclass()
class Instance:
    """
    Class used for modeling single data instance. It consists of list of word tokens from raw essay and corresponding
    labels.
    """
    x: list
    y: list

    def __init__(self, x, y):
        self.x = word_tokenize(x.lower())
        self.y = y


class NLPDataset(Dataset):
    """
    Class used for modeling essay dataset.
    """
    def __init__(self, x, y, vocabulary):
        """
        Initialization method.

        :param x: List of raw essays.
        :param y: List of accompanying labels.
        :param vocabulary: Vocabulary containing known words.
        """
        self.txt_vocab = vocabulary
        self.instances = []
        for x_i, y_i in zip(x, y):
            self.instances.append(astuple(Instance(x_i, y_i)))
        self.len = len(self.instances)

    def __getitem__(self, index) -> T_co:
        """
        Returns a single instance at given index.
        :param index: index of the desired instance.
        :return: embedding indices corresponding to the desired essay word tokens and labels: tuple(list[int], list)
        """
        if index < 0 or index >= self.len:
            raise IndexError(f"Invalid index {index} for array of len {self.len}")
        x, y = self.instances[index]
        return self.txt_vocab.encode(x), y

    def __len__(self):
        """
        Method used for retrieving number of instances.
        :return: Number of dataset instances: int
        """
        return self.len


def load_s2v(wiki=True):
    s2v = sent2vec.Sent2vecModel()
    print(f"Loading pretrained ({'wiki' if wiki else 'toronto'}) S2V vectors...", end=' ')
    s2v.load_model(S2V_WIKI_UNGIRAMS_PATH if wiki else S2V_TORONTO_UNIGRAMS_PATH)
    print("DONE")
    return s2v, 600 if wiki else 700


class S2VDataset(Dataset):
    def __init__(self, x, y, s2v, shape):
        self.s2v = s2v
        self.shape = shape
        self.len = len(x)
        self.x, self.y = self.build(x, y)

    def build(self, x, y):
        x_emb = []
        for i, sentences in enumerate(x):
            emb_sents = torch.tensor(self.s2v.embed_sentences(sentences))   # N(sent) x 700 or 600
            x_emb.append(emb_sents)
        return x_emb, y

    def __getitem__(self, item):
        if item < 0 or item >= self.len:
            raise IndexError(f"Invalid index {item} for array of len {self.len}")
        x, y = self.x[item], self.y[item]
        return x, y

    def __len__(self):
        return self.len


def extract_frequencies(x):
    """
    Method used for extracting word occurrences in the list of raw essays.
    :param x: list of raw essays:
    :return: dictionary which maps number of occurrences to each word token, sorted in descending order according to
    number of occurrences
    """
    x_frequencies = {}

    for text in x:

        words = word_tokenize(text.lower())

        for w in words:
            x_count = 1 if x_frequencies.get(w) is None else x_frequencies.get(w)
            x_frequencies[w] = x_count + 1

    return sorted(x_frequencies.items(), key=lambda a: a[1], reverse=True)


class Vocab:
    """
    Class which models a vocabulary of known words and their indices.
    """
    def __init__(self, frequencies, max_size=-1, min_freq=0):
        """
        Initialization method.

        :param frequencies: dictionary containing known words and their occurrences: dict
        :param max_size: maximum number of words to be kept, if -1 is passed then there is no limit on the number of
        words to be kept: int
        :param min_freq: minimum number of word occurrences needed for the word to be considered relevant: int
        """
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
        """
        Method used for retrieving embedding indices for each input word.
        :param words: list of input word tokens or a single word token
        :return: list containing embedding indices: list[int]
        """
        res = []
        if type(words) is list:
            for w in words:
                ind = self.stoi.get(w, self.stoi[UNK])
                res.append(ind)
        else:
            ind = self.stoi.get(words, self.stoi[UNK])
            res.append(ind)
        return torch.tensor(res)


def load_rnn_features(x=None, y=None, **kwargs):
    """
    Method used for loading rnn dataset and splitting that dataset into train, validation, train-validation and test
    dataset.
    :param x: loaded list of essays
    :param y: loaded list of labels
    :param kwargs: additional parameters.
    :return: tuple containing loaded and split datasets and vocabulary constructed over train dataset
    """
    if x is None and y is None:
        print("Loading dataset from CSV file...", end=' ')
        x, y = load_dataset()
        print("DONE")

    print("Creating train/valid/test splits...", end=' ')
    (trnx, trny), (valx, valy), (tesx, tesy) = split_dataset(x, y, test_ratio=kwargs['test_ratio'],
                                                             valid_ratio=kwargs['valid_ratio'])
    print("DONE")

    if kwargs['s2v']:
        vocab = None
        trn_sent, val_sent, tes_sent = [[sent_tokenize(ex.lower()) for ex in ds] for ds in [trnx, valx, tesx]]
        trnval_sent = trn_sent + val_sent
        s2v, dims = load_s2v(kwargs['wiki'])

        train_ds = S2VDataset(trn_sent, trny, s2v, dims)
        valid_ds = S2VDataset(val_sent, valy, s2v, dims)
        trainval_ds = S2VDataset(trnval_sent, torch.cat((trny, valy), dim=0), s2v, dims)
        test_ds = S2VDataset(tes_sent, tesy, s2v, dims)

    else:
        print("Building vocabulary...", end=' ')
        vocab = Vocab(extract_frequencies(trnx), max_size=kwargs["max_size"], min_freq=kwargs["min_freq"])
        print("DONE")

        train_ds = NLPDataset(trnx, trny, vocab)
        valid_ds = NLPDataset(valx, valy, vocab)
        trainval_ds = NLPDataset(trnx + valx, torch.cat((trny, valy), dim=0), vocab)
        test_ds = NLPDataset(tesx, tesy, vocab)

    return train_ds, valid_ds, trainval_ds, test_ds, vocab


def pad_collate_fn(batch, pad_index=0):
    """
    Collate function used for padding input list of embedding indices to tensor of same shape.
    :param batch: essays and labels
    :param pad_index: embedding index to be used for padding
    :return: padded essay tensor, labels tensor and original essay lengths
    """

    texts, labels = zip(*batch)  # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in texts])  # Needed for later

    texts_tensor = pad_sequence(list(texts), padding_value=pad_index, batch_first=True)
    labels_tensor = torch.vstack(labels)

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
