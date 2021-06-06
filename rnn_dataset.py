import os
import re
from dataclasses import dataclass

import gensim
import numpy as np
import sent2vec
import torch
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from torch.nn import Embedding
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

from dataset import load_dataset, split_dataset
from extractor import W2V_GOOGLE_NEWS_PATH
from utils import project_path

PAD = "<PAD>"
UNK = "<UNK>"

RE_WSPACE = f'\\s+'
RE_DELIM = f', '

# Sent2vec Pre-Trained Models from https://github.com/epfml/sent2vec/)
S2V_WIKI_UNGIRAMS_PATH = os.path.join(project_path, "saved/s2v/wiki_unigrams.bin")
S2V_TORONTO_UNIGRAMS_PATH = os.path.join(project_path, "saved/s2v/torontobooks_unigrams.bin")

EMOTIONAL_WORDS_PATH = os.path.join(project_path, "emotions/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")


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
        self.x = x
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
            self.instances.append(Instance(x_i, y_i))
        self.len = len(self.instances)

    def __getitem__(self, index) -> T_co:
        """
        Returns a single instance at given index.
        :param index: index of the desired instance.
        :return: embedding indices corresponding to the desired essay word tokens and labels: tuple(list[int], list)
        """
        if index < 0 or index >= self.len:
            raise IndexError(f"Invalid index {index} for array of len {self.len}")
        instance = self.instances[index]
        return self.txt_vocab.encode(instance.x), instance.y

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
            emb_sents = torch.tensor(self.s2v.embed_sentences(sentences))  # N(sent) x 700 or 600
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
    Method used for extracting word occurrences in the list of tokenized essays.
    :param x: list of tokenized essays
    :return: dictionary which maps number of occurrences to each word token, sorted in descending order according to
    number of occurrences
    """
    x_frequencies = {}

    for words in x:

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
        x, y = load_dataset(max_essays=kwargs["max_essays"])
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
        trnx, valx, tesx = [[word_tokenize(ex.lower()) for ex in ds] for ds in [trnx, valx, tesx]]

        print("Building vocabulary...", end=' ')
        vocab = Vocab(extract_frequencies(trnx), max_size=kwargs["max_size"], min_freq=kwargs["min_freq"])
        print("DONE")

        train_ds = NLPDataset(trnx, trny, vocab)
        valid_ds = NLPDataset(valx, valy, vocab)
        trainval_ds = NLPDataset(trnx + valx, torch.cat((trny, valy), dim=0), vocab)
        test_ds = NLPDataset(tesx, tesy, vocab)

    return train_ds, valid_ds, trainval_ds, test_ds, vocab


def load_emotional_words():
    emotional_words = set()
    with open(EMOTIONAL_WORDS_PATH, 'r', encoding='cp1252') as em_path:
        while True:
            line = em_path.readline().strip()
            if not line:
                break
            parts = re.split(RE_WSPACE, line)
            word, flag = parts[0], int(parts[2])
            if flag == 1:
                emotional_words.add(word)
    return emotional_words


def emotionally_neutral_drop(subset, emotional_words):
    lemmatizer = WordNetLemmatizer()
    subset_new = []
    for txt in subset:
        sentences = sent_tokenize(txt)
        relevant_sentences = []
        for s in sentences:
            words = word_tokenize(s.lower())
            for w in words:
                lemma = lemmatizer.lemmatize(w)
                if lemma in emotional_words:
                    relevant_sentences.append(s)  # at least one emotionally charged word is needed for relevance
                    break
        if len(relevant_sentences) == 0:
            subset_new.append(txt)
        else:
            subset_new.append(' '.join(relevant_sentences))  # possibly discard empty essays
    return tuple(subset_new)


def emotionally_neutral_drop_chunks(data, emotional_words):
    # data: list[ list[ list[ str ] ] ]
    #       N     C     T     word
    # N = number of examples
    # C = number of chunks in example
    # T = number of tokens in chunk
    lemmatizer = WordNetLemmatizer()
    new_data = []
    dropped_chunks_per_author = []
    for example in data:
        new_example = []
        for chunk in example:
            for token in chunk:
                lemma = lemmatizer.lemmatize(token)
                if lemma in emotional_words:
                    new_example.append(chunk)
                    break
        if len(new_example) == 0:
            new_example = example
        dropped_chunks_per_author.append(len(example) - len(new_example))
        new_data.append(new_example)
    # print(f"Dropped chunks per author: {dropped_chunks_per_author}")
    # print(f"Total dropped chunks: {sum(dropped_chunks_per_author)}")
    # print(f"Author count: {len(dropped_chunks_per_author)}")
    return new_data


INTERPUNCTION = '.!?,'


def to_sentences(tokens, min_chunk_length):
    chunks = []
    start = 0
    end = min_chunk_length
    while start < len(tokens):
        while end < len(tokens) and tokens[end] not in INTERPUNCTION:
            end += 1
        if 10 < end - start:
            chunks.append(tokens[start:end])
        start = end
        end = min(start + min_chunk_length, len(tokens))
    return chunks


EMOTION_DROP_VERSIONS = ["none", "v1sent", "v2chunk"]


def load_features_2(**kwargs):
    def extract_authors(x, y):
        x_new = []
        x_auth = []
        y_new = []

        for i, chunks in enumerate(x):
            x_new += chunks
            x_auth += [i for _ in chunks]
            y_new += [y[i] for _ in chunks]

        return x_new, x_auth, torch.stack(y_new)

    x, y = load_dataset()

    x_toks = [word_tokenize(xi.lower()) for xi in x]
    x_chunked_toks = [to_sentences(tokens, kwargs['min_chunk_length']) for tokens in x_toks]

    emotion_drop = kwargs.get("emotion_drop")
    assert emotion_drop in EMOTION_DROP_VERSIONS

    if emotion_drop == "v1sent":
        print("Loading emotionally charged words...", end=' ')
        emotional_words = load_emotional_words()
        print("DONE")
        print("Dropping emotionally neutral sentences...", end=' ')
        x = emotionally_neutral_drop(x, emotional_words)
        print("DONE")
    elif emotion_drop == "v2chunk":
        print("Loading emotionally charged words...", end=' ')
        emotional_words = load_emotional_words()
        print("DONE")
        print("Dropping emotionally neutral chunks...", end=' ')
        x_chunked_toks = emotionally_neutral_drop_chunks(x_chunked_toks, emotional_words)
        print("DONE")

    (trnx, trny), (valx, valy), (tesx, tesy) = split_dataset(x_chunked_toks, y, test_ratio=kwargs['test_ratio'],
                                                             valid_ratio=kwargs['valid_ratio'])
    trnx, trna, trny = extract_authors(trnx, trny)
    valx, vala, valy = extract_authors(valx, valy)
    tesx, tesa, tesy = extract_authors(tesx, tesy)

    print("Building vocabulary...", end=' ')
    vocab = Vocab(extract_frequencies(trnx), max_size=kwargs["max_size"], min_freq=kwargs["min_freq"])
    print("DONE")

    print("Building datasets...", end=' ')
    train_ds = NLPDataset(trnx, trny, vocab)
    valid_ds = NLPDataset(valx, valy, vocab)
    trainval_ds = NLPDataset(trnx + valx, torch.cat((trny, valy), dim=0), vocab)
    test_ds = NLPDataset(tesx, tesy, vocab)
    print("DONE")
    return (train_ds, trna), (valid_ds, vala), (trainval_ds, trna + vala), (test_ds, tesa), vocab


# load_features_2(test_ratio=0.2, valid_ratio=0.2, max_size=-1, min_freq=1, emotion_drop=True)


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
    # packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)
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
