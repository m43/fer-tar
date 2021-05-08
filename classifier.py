import torch
from sklearn.svm import SVC

from baselines import Classifier
from dataset import split_dataset, load_dataset
from eval import eval
from extractor import DummyExtractor


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


class SVMClassifier(Classifier):
    """
    Classifier that creates 5 SVMs to predict each of the traits.
    """

    def __init__(self, extractor, c=1, gamma="auto", decision_function_shape="ovo", kernel='rbf'):
        self.extractor = extractor
        self.clf = None
        self.c = c
        self.gamma = gamma
        self.kernel = kernel
        self.decision_function_shape = decision_function_shape

    def train(self, dataset):
        features = self.extractor.extract(dataset)
        self.clfs = [SVC(kernel=self.kernel, decision_function_shape=self.decision_function_shape, C=self.c,
                         gamma=self.gamma) for _ in range(5)]
        targets = torch.Tensor([x[2:] for x in dataset])
        for i in range(5):
            self.clfs[i].fit(features, targets[:, i])

    def classify(self, example):
        features = extractor.extract([example])
        return [self.clfs[i].predict(features) for i in range(5)]


if __name__ == '__main__':
    ds = load_dataset()
    train, _, test = split_dataset(ds, 0.3, 0)

    # ~~~~ SVMClassifier ~~~~ #
    extractor = DummyExtractor()
    clf = SVMClassifier(extractor)
    clf.train(ds)
    print(eval(clf, train))
