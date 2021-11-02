from abc import ABC, abstractmethod

import gensim
import torch

from dataset import PAD
from extractor import W2V_GOOGLE_NEWS_PATH


class SequenceGenerator(ABC):
    """
    Abstract class to model sequence generators
    """

    @abstractmethod
    def generate(self, x_tok, **kwargs):
        """

        :param x_tok:
        :param kwargs:
        :return:
        """
        pass


class W2VGenerator(SequenceGenerator):

    def __init__(self, vocab, **kwargs):
        self.vocab = vocab
        self.model = gensim.models.KeyedVectors.load_word2vec_format(W2V_GOOGLE_NEWS_PATH,
                                                                     binary=True,
                                                                     limit=kwargs['w2v_limit'])

    def generate(self, x_tok, **kwargs):
        pad_vector = kwargs[PAD]
        vecs = []
        lens = []
        max_len = 0
        for tokens in x_tok:  # batch of tokenized essays
            in_vocab = [t for t in tokens if t in self.vocab.stoi and t in self.model.vocab]
            lens.append(torch.tensor(len(in_vocab)))  # original essay lens

            if len(in_vocab) > max_len:
                max_len = len(in_vocab)
            vec = []

            for word in in_vocab:
                vec.append(torch.from_numpy(self.model[word]))
            vecs.append(vec)

        for i in range(len(vecs)):
            for ind in range(lens[i], max_len):
                vecs[i].append(pad_vector)
            vecs[i] = torch.stack(vecs[i])

        return torch.stack(vecs), lens


