from abc import abstractmethod, ABC

import numpy as np

import dataset
import eval
from utils import setup_torch_reproducibility


class Classifier(ABC):
    """
    Abstract class that models a big5 traits classifier.
    The classifier can be trained on a given document set
    and can predict the trait of a given text.
    """

    @abstractmethod
    def train(self, dataset):
        """
        :param dataset: the dataset; list[list[str, str, bool, bool, bool, bool, bool]]
        :return: returns nothing
        """
        pass

    @abstractmethod
    def classify(self, example):
        """
        :param example: text to be classified; string
        :return: predicted labels; list[bool]
        """
        pass


class MCCBaseline(Classifier):
    """
    Majority class classifier
    """

    def __init__(self):
        self.labels = None

    def train(self, dataset):
        count_true = [0, 0, 0, 0, 0]
        for i in range(len(dataset)):
            for j in range(5):
                count_true[j] += 1 if dataset[i][j + 2] else 0
        self.labels = [count > len(dataset) / 2 for count in count_true]

    def classify(self, example):
        return self.labels


class RandomBaseline(Classifier):
    """
    Random class classifier
    """

    def train(self, dataset):
        pass

    def classify(self, example):
        return [np.random.random() > 0.5 for _ in range(5)]


if __name__ == '__main__':
    setup_torch_reproducibility(72)
    ds = dataset.load_dataset()

    # ~~~~ MCCBaseline ~~~~ #
    model = MCCBaseline()
    model.train(ds)
    for ex in ds:
        label = model.classify(ex[2])

    print("MCCBaseline:")
    scores = eval.eval(model, ds)
    for k in scores.keys():
        print(scores[k])

    # ~~~~ RandomBaseline ~~~~ #
    model = RandomBaseline()
    model.train(ds)
    for ex in ds:
        label = model.classify(ex[2])
        print(label)

    print("RandomBaseline:")
    scores = eval.eval(model, ds)
    for k in scores.keys():
        print(scores[k])
