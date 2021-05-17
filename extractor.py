import re
from abc import abstractmethod, ABC

import gensim
import sent2vec
import torch

punct = ['.', '!', '?']
RE_PUNCT = r'[?.!]'
RE_WSPACE = r'\s+'
WORD_VEC_PATH = "./Word2Vec/GoogleNews-vectors-negative300.bin"
SENT_VEC_PATH = "sentences/wiki_unigrams.bin"


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
        return self.vectorizer.transform(x)


class W2VExtractor(FeatureExtractor):
    def __init__(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(WORD_VEC_PATH, binary=True)

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
    def __init__(self):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(SENT_VEC_PATH)

    def extract(self, x):
        vecs = []
        for i, text in enumerate(x):
            sentences = re.split(RE_PUNCT, text[1])
            embeddings = [torch.tensor(self.model.embed_sentence(s)) for s in sentences if s]
            embeddings = torch.stack(embeddings)
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
               "12345",
               "/.,/.,/.,", "?!.", "This is a noooooormal sentence."]
    test_x = ["Četrnaest palmi"]

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

    for e in extractors:
        print(f"Extractor {e.__class__.__name__} -- TRAIN:")
        print(e.extract(train_x))
        print(f"Extractor {e.__class__.__name__} -- TEST:")
        print(e.extract(test_x))

    # TODO there are some nan's and inf's in the output, but these are probably edge cases.
