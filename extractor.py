import re
import gensim

from abc import abstractmethod, ABC

import torch

punct = ['.', '!', '?']
RE_PUNCT = r'[?.!]'
RE_WSPACE = r'\s+'
WORD_VEC_PATH = "./Word2Vec/GoogleNews-vectors-negative300.bin"


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, dataset):
        pass


class DummyExtractor(FeatureExtractor):
    def extract(self, dataset):
        return torch.randn((len(dataset), 30))


class BOWExtractor(FeatureExtractor):
    def __init__(self, dataset):
        pass

    def extract(self, dataset):
        # list[list[str,str,bool...]]
        # list[np.array]
        pass


class W2VExtractor(FeatureExtractor):
    def __init__(self):
        self.model = gensim.models.Word2Vec.load_word2vec_format(WORD_VEC_PATH, binary=True)

    def extract(self, x):
        vecs = []
        for i, text in enumerate(x):
            vec = torch.zeros((1, 300))
            for word in re.split(RE_WSPACE):
                vec += self.model[word]
            vecs.append(torch.mean(vec))
        return torch.stack(vecs)

class S2VExtractor(FeatureExtractor):
    def extract(self, dataset):
        pass


class D2VExtractor(FeatureExtractor):
    def extract(self, dataset):
        pass


class InterpunctionExtractor(FeatureExtractor):

    def extract(self, dataset):
        """
        Extracts interpunction counts for each input entry.

        :param dataset: the dataset from which the features are to be extracted
        :return: a tensor of shape len(dataset) x 3 containing extracted interpunction features: tensor
        """
        counts_per_person = []
        for _, text, _, _, _, _, _ in dataset:
            punct_count = torch.tensor([0.0, 0.0, 0.0])
            for c in text:
                for i in range(len(punct)):
                    if c == punct[i]:
                        punct_count[i] += 1
                        break
            counts_per_person.append(punct_count)  # TODO how to normalize? Total number od punctuation marks?
        # Total number of sentences?
        return torch.stack(counts_per_person)


class InterpunctionNormalizedBySentences(FeatureExtractor):
    def extract(self, dataset):
        int_ext = InterpunctionExtractor()
        counts_per_person = int_ext.extract(dataset)

        for i, entry in enumerate(dataset):
            sentences = re.split(RE_PUNCT, entry[1])
            sentences = [s for s in sentences if s]
            counts_per_person[i] /= len(sentences)

        return counts_per_person


class InterpunctionNormalizedByOccurence(FeatureExtractor):
    def extract(self, dataset):
        int_ext = InterpunctionExtractor()
        counts_per_person = int_ext.extract(dataset)

        for i in range(len(dataset)):
            sum = torch.sum(counts_per_person[i], 0)
            if sum > 0:
                for j in range(len(counts_per_person[i])):
                    counts_per_person[i, j] /= sum

        return counts_per_person


class CapitalizationExtractor(FeatureExtractor):
    def extract(self, dataset):
        """
        Extracts capitalization counts relative to number of sentences for each input entry.

        :param dataset: the dataset from which the features are to be extracted
        :return: a tensor of shape len(dataset) x 1 containing extracted capitalization features: tensor[int...]
        """
        cap_per_person = torch.zeros((len(dataset), 1))

        for i, entry in enumerate(dataset):
            for c in entry[1]:
                if c.isupper():
                    cap_per_person[i] += 1
            sentences = re.split(RE_PUNCT, entry[1])
            sentences = [s for s in sentences if s]  # consider only non empty sentences
            cap_per_person[i] /= len(sentences)  # normalize capitalization with the number of sentences
        return cap_per_person


class RepeatingLettersExtractor(FeatureExtractor):
    def extract(self, dataset):
        values = []
        for example in dataset:
            text = example[1].lower()
            values.append([0])
            count_reps = 1
            for i in range(1, len(text)):
                if (not text[i].isalpha()) or (text[i] != text[i - 1]):
                    if count_reps > 2:
                        values[-1][0] += count_reps
                    count_reps = 1
                else:
                    count_reps += 1
            values[-1][0] *= 1.0
        return torch.tensor(values)


class WordCountExtractor(FeatureExtractor):
    def __init__(self, dataset):
        lens = torch.tensor([[1. * len(example[1].split())] for example in dataset])
        self.mean = torch.mean(lens)
        self.stddev = torch.sqrt(torch.var(lens))

    def extract(self, dataset):
        lens = torch.tensor([[1. * len(example[1].split())] for example in dataset])
        return (lens - self.mean) / self.stddev


class CompositeExtractor(FeatureExtractor):

    def __init__(self, extractors):
        self.extractors = extractors

    def extract(self, dataset):
        features = self.extractors[0].extract(dataset)
        for i in range(1, len(self.extractors)):
            features = torch.cat((features, self.extractors[i].extract(dataset)), dim=1)
        return features
