import os
import re
from abc import abstractmethod, ABC

import gensim
import numpy as np
import torch

from utils import project_path

punct = ['.', '!', '?']
RE_PUNCT = r'[?.!]'
RE_WSPACE = r'\s+'

# Word2vec Pre-Trained Models from https://code.google.com/archive/p/word2vec/
W2V_GOOGLE_NEWS_PATH = os.path.join(project_path, "saved/w2v/GoogleNews-vectors-negative300.bin")

# Sent2vec Pre-Trained Models from https://github.com/epfml/sent2vec/)
S2V_TWITTER_UNIGRAMS_PATH = os.path.join(project_path, "saved/s2v/twitter_unigrams.bin")
S2V_TWITTER_BIGRAMS_PATH = os.path.join(project_path, "saved/s2v/twitter_bigrams.bin")


class FeatureExtractor(ABC):
    """
    Abstract class to model features extractors
    """

    @abstractmethod
    def extract(self, x):
        """
        Extracts features from given essays.

        :param x: a list of essays; list[str]
        :return: a tensor containing the extracted features; torch.tensor
        """
        pass


class DummyExtractor(FeatureExtractor):
    """
    Dummy feature extractor that returns random vectors of given dimensionality
    """

    def __init__(self, dim=100):
        self.dim = dim

    def extract(self, x):
        return torch.randn((len(x), self.dim))


class BOWExtractor(FeatureExtractor):
    def __init__(self, x):
        from sklearn.feature_extraction.text import CountVectorizer
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(x)

    def extract(self, x):
        coo = self.vectorizer.transform(x).tocoo()

        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


class W2VExtractor(FeatureExtractor):
    def __init__(self, pretrained_path=W2V_GOOGLE_NEWS_PATH):
        assert os.path.exists(pretrained_path)
        self.model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_path, binary=True)

    def extract(self, x):
        vecs = []
        for text in x:
            words = re.split(RE_WSPACE, text)
            vec = torch.empty((len(words), 300))
            for i, word in enumerate(words):
                try:
                    vec[i] = torch.tensor(self.model[word])
                except KeyError:  # out of vocabulary word
                    continue
            vecs.append(torch.mean(vec, dim=0))
        return torch.stack(vecs)


class S2VExtractor(FeatureExtractor):
    def __init__(self, pretrained_model_path=S2V_TWITTER_UNIGRAMS_PATH):
        import sent2vec  # https://github.com/epfml/sent2vec/)
        self.model = sent2vec.Sent2vecModel()
        assert os.path.exists(pretrained_model_path)
        self.model.load_model(pretrained_model_path)

    def extract(self, x):
        vecs = []
        for i, text in enumerate(x):
            sentences = [s for s in re.split(RE_PUNCT, text[1]) if s]
            embeddings = torch.tensor(self.model.embed_sentences(sentences))
            vecs.append(torch.mean(embeddings, dim=0))
        return torch.stack(vecs)


class D2VExtractor(FeatureExtractor):
    def extract(self, x):
        return torch.zeros((len(x), 1))


class InterpunctionExtractor(FeatureExtractor):

    def extract(self, x):
        """
        Extracts interpunction counts for each input entry.

        :param x: a list of essays; list[str]
        :return: a tensor of shape (len(dataset),len(punct)) containing extracted interpunction features; tensor
        """
        counts_per_person = []
        for text in x:
            punct_count = torch.zeros(len(punct))
            for c in text:
                for i in range(len(punct)):
                    if c == punct[i]:
                        punct_count[i] += 1
                        break
            counts_per_person.append(punct_count)  # TODO how to normalize? Total number od punctuation marks?
        # Total number of sentences?
        return torch.stack(counts_per_person)


class InterpunctionNormalizedBySentences(FeatureExtractor):
    def extract(self, x):
        int_ext = InterpunctionExtractor()
        counts_per_person = int_ext.extract(x)

        for i, xi in enumerate(x):
            sentences = re.split(RE_PUNCT, xi)
            sentences = [s for s in sentences if s]
            counts_per_person[i] /= len(sentences)

        return counts_per_person


class InterpunctionNormalizedByOccurence(FeatureExtractor):
    def extract(self, x):
        int_ext = InterpunctionExtractor()
        counts_per_person = int_ext.extract(x)

        for i in range(len(x)):
            sum = torch.sum(counts_per_person[i], 0)
            if sum > 0:
                for j in range(len(counts_per_person[i])):
                    counts_per_person[i, j] /= sum

        return counts_per_person


class CapitalizationExtractor(FeatureExtractor):
    def extract(self, dataset):
        """
        Extracts capitalization counts relative to number of sentences for each input entry.

        :param x: a list of essays; list[str]
        :return: a tensor of shape (len(dataset),1) containing extracted capitalization features; torch.tensor[float]
        """
        cap_per_person = torch.zeros((len(dataset), 1), dtype=torch.float32)

        for i, entry in enumerate(dataset):
            for c in entry:
                if c.isupper():
                    cap_per_person[i] += 1
            sentences = re.split(RE_PUNCT, entry)
            sentences = [s for s in sentences if s]  # consider only non empty sentences
            cap_per_person[i] /= len(sentences)  # normalize capitalization with the number of sentences
        return cap_per_person


class RepeatingLettersExtractor(FeatureExtractor):
    def extract(self, x):
        values = []
        for example in x:
            text = example.lower()
            values.append([0])
            count_reps = 1
            for i in range(1, len(text)):
                if (not text[i].isalpha()) or (text[i] != text[i - 1]):
                    if count_reps > 2:
                        values[-1][0] += count_reps
                    count_reps = 1
                else:
                    count_reps += 1
        return torch.tensor(values, dtype=torch.float32)


class WordCountExtractor(FeatureExtractor):

    def extract(self, x):
        return torch.tensor([[1. * len(example.split())] for example in x])


class CompositeExtractor(FeatureExtractor):

    def __init__(self, extractors):
        self.extractors = extractors

    def extract(self, x):
        features = self.extractors[0].extract(x)
        for i in range(1, len(self.extractors)):
            features = torch.cat((features, self.extractors[i].extract(x)), dim=1)
        return features


if __name__ == '__main__':
    train_x = ["Četrnaest palmi na otoku sreće Žalo po kojem se valja val", "no interpunction", "CAPITAL", "lower",
               "12345", "This is a noooooormal sentence."]  # "/.,/.,/.,", "?!."
    test_x = ["Četrnaest palmi"]

    print("Initializing extractors...")

    extractors = [
        DummyExtractor(),
        DummyExtractor(10),
        BOWExtractor(train_x),
        W2VExtractor(),
        S2VExtractor(),
        D2VExtractor(),
        InterpunctionExtractor(),
        InterpunctionNormalizedBySentences(),
        InterpunctionNormalizedByOccurence(),
        CapitalizationExtractor(),
        RepeatingLettersExtractor(),
        WordCountExtractor()
    ]
    extractors += [CompositeExtractor(tuple(extractors))]

    print("Extractors initialized.")
    for e in extractors:
        print(f"Extractor {e.__class__.__name__} -- TRAIN:")
        print(e.extract(train_x))
        print(f"Extractor {e.__class__.__name__} -- TEST:")
        print(e.extract(test_x))

    # TODO there are some nan's and inf's in the output, but these are probably edge cases.
