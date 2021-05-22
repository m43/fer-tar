from abc import abstractmethod, ABC

import torch
from typing import Tuple

from sklearn.svm import SVC

import fc
from baselines import Classifier
from dataset import split_dataset, load_dataset
from eval import eval
from extractor import DummyExtractor
from trainer.trainer import DeepFCTrainer

from torch.utils.data import DataLoader


class TraitClassifier(ABC):
    """
    Abstract class modeling a single binary classifier,
    to be used as a classifier for a single Big5 trait.
    """

    def __init__(self, index):
        """
        Constructor for a TraitClassifier, taking the index of the trait
        for which it is specialized.

        :param index: index of the trait to classify; int
        """
        self.index = index

    @abstractmethod
    def train(self, data: Tuple[DataLoader, DataLoader, DataLoader], **kwargs):
        """
        Trains the classifier on the given data which includes a train,
        validation and test splits.

        :param data: tuple of three datasets (train, valid, test); tuple(torch.DataLoader)
        :param kwargs: possible additional arguments; dict{str:Any}
        :return: None
        """
        pass

    @abstractmethod
    def forward(self, data: DataLoader):
        """
        Performs a forward pass with the given data as input,
        returning the scores. Since the DataLoader can shuffle
        the data, the true labels are also returned.

        For classifiers with an undefined "forward pass", such as
        SVM, this method returns the classification labels.

        :param data: dataset; torch.DataLoader
        :return: tensor of prediction scores and tensor of true labels; (torch.tensor, torch.tensor)
        """
        pass

    @abstractmethod
    def classify(self, data: DataLoader):
        """
        Predicts the true labels of the given data. Since the
        DataLoader shuffle the data, the true labels are also
        returned.

        :param data: dataset; torch.DataLoader
        :return: tensor of predicted labels and tensor of true labels; (torch.tensor, torch.tensor)
        """
        pass


class CompoundClassifier:
    def __init__(self, hooks):
        self.clfs = [hook(i, **kwargs) for i, (hook, kwargs) in enumerate(hooks)]

    def train(self, data, **kwargs):
        for clf in self.clfs:
            clf.train(data, **kwargs)

    def forward(self, data):
        scores = torch.zeros((len(data.dataset), 5))
        true = torch.zeros((len(data.dataset), 5))
        for i, clf in enumerate(self.clfs):
            clf_scores, clf_true = clf.forward(data)
            scores[:, i] = clf_scores.flatten()
            true[:, i] = clf_true.flatten()
        return scores, true

    def classify(self, data):
        preds = torch.zeros((len(data.dataset), 5))
        true = torch.zeros((len(data.dataset), 5))
        for i, clf in enumerate(self.clfs):
            clf_preds, clf_true = clf.classify(data)
            preds[:, i] = clf_preds.flatten()
            true[:, i] = clf_true.flatten()
        return preds, true


class FCClassifier(Classifier):
    def train(self, train_dataloader, valid_dataloader, test_dataloader, params):
        self.clfs = [fc.DeepFC(params["neurons_per_layer"]) for _ in range(5)]
        for i in range(len(self.clfs)):
            DeepFCTrainer.train(self.clfs[i], train_dataloader, valid_dataloader, test_dataloader, self.params)

    def classify(self, x, y):
        preds = self.clfs[0].forward(x)
        for i in range(1, len(self.clfs)):
            preds = torch.cat((preds, self.clfs[i].forward(x)), dim=1)
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
