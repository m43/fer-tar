from abc import abstractmethod, ABC

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

class ProlongingVowelsExtractor(FeatureExtractor):
    def extract(self, dataset):
        pass

class WordCountExtractor(FeatureExtractor):
    def extract(self, dataset):
        pass

class CompositeExtractor(FeatureExtractor):

    def __init__(self, extractors):
        self.extractors = extractors

    def extract(self, dataset):
        pass