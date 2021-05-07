from abc import abstractmethod, ABC
import torch

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, dataset):
        pass

class BOWExtractor(FeatureExtractor):
    def __init__(self, dataset):
        pass

    def extract(self, dataset):
        #list[list[str,str,bool...]]
        #list[np.array]
        pass

class W2VExtractor(FeatureExtractor):
    def extract(self, dataset):
        pass

class S2VExtractor(FeatureExtractor):
    def extract(self, dataset):
        pass

class D2VExtractor(FeatureExtractor):
    def extract(self, dataset):
        pass

class InterpunctionExtractor(FeatureExtractor):
    def extract(self, dataset):
        pass

class CapitalizationExtractor(FeatureExtractor):
    def extract(self, dataset):
        pass


class RepeatingLettersExtractor(FeatureExtractor):
    def extract(self, dataset):
        values = []
        for example in dataset:
            text = example[1].lower()
            values.append([0])
            count_reps = 1
            for i in range(1, len(text)):
                if (not text[i].isalpha()) or (text[i] != text[i-1]):
                    if count_reps > 2:
                        values[-1][0] += count_reps
                    count_reps = 1
                else:
                    count_reps += 1
            values[-1][0] *= 1.0
        return torch.tensor(values)


class WordCountExtractor(FeatureExtractor):
    def extract(self, dataset):
        lens = torch.tensor([[1. * len(example[1].split())] for example in dataset])
        shifted = lens - torch.mean(lens)
        scaled = shifted / torch.sqrt(torch.var(lens))
        return scaled

class CompositeExtractor(FeatureExtractor):

    def __init__(self, extractors):
        self.extractors = extractors

    def extract(self, dataset):
        pass