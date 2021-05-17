import torch

import fc
from sklearn.svm import SVC

from baselines import Classifier
from dataset import split_dataset, load_dataset
from eval import eval
from extractor import DummyExtractor


class FCClassifier(Classifier):
    def __init__(self, extractor, dev):
        self.extractor = extractor
        self.dev = dev
        self.clfs = []
        self.mean = None
        self.stddev = None

    def train(self, x, y, epochs=5, lr=1e-4, batch_size=20, wd=1e-4):
        features = self.extractor.extract(x).to(device=self.dev)

        self.mean = torch.mean(features, dim=0)
        self.stddev = torch.sqrt(torch.var(features, dim=0))
        features = (features - self.mean) / self.stddev
        y = y.to(device=self.dev)

        self.clfs = [fc.DeepFC(in_dim=len(features[0]), device=self.dev)
                     for _ in range(5)]
        for i in range(len(self.clfs)):
            fc.train(self.clfs[i], features, y[:, i:i+1], self.dev, epochs, lr, batch_size, wd)

    def classify(self, x, y):
        features = self.extractor.extract(x).to(device=self.dev)
        features = (features - self.mean) / self.stddev
        preds = self.clfs[0].forward(features)
        for i in range(1, len(self.clfs)):
            preds = torch.cat((preds, self.clfs[i].forward(features)), dim=1)
        return preds


class LSTMClassifier(Classifier):
    pass


class SVMClassifier(Classifier):
    """
    Classifier that creates 5 SVMs to predict each of the traits.
    """

    def __init__(self, extractor, c=1, gamma="auto", decision_function_shape="ovo", kernel='rbf'):
        self.extractor = extractor
        self.clfs = None
        self.c = c
        self.gamma = gamma
        self.kernel = kernel
        self.decision_function_shape = decision_function_shape

    def train(self, x, y):
        features = self.extractor.extract(x)
        self.clfs = [SVC(kernel=self.kernel, decision_function_shape=self.decision_function_shape, C=self.c,
                         gamma=self.gamma) for _ in range(5)]
        for i in range(5):
            self.clfs[i].fit(features, y[:, i])

    def classify(self, example_x, example_y):
        features = self.extractor.extract([example_x])
        return [self.clfs[i].predict(features) for i in range(5)]


if __name__ == '__main__':
    x, y = load_dataset()
    (train_x, train_y), _, (test_x, test_y) = split_dataset(x, y, 0.3, 0)

    # ~~~~ SVMClassifier ~~~~ #
    extractor = DummyExtractor(10000)
    clf = SVMClassifier(extractor)
    clf.train(x, y)
    print("Train:", eval(clf, train_x, train_y))
    print("Test:", eval(clf, test_x, test_y))
