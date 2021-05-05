from baselines import Classifier

class FCClassifier(Classifier):
    def __init__(self, extractor):
        self.extractor = extractor

    def train(self, dataset):
        classifier = self.init_classifier(dataset)

    def classify(self, example):
        pass

    def init_classifier(self, dataset):
        pass

class LSTMClassifier(Classifier):
    pass