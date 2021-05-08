from abc import abstractmethod, ABC

import numpy as np
import torch

import eval
from dataset import load_dataset
from utils import setup_torch_reproducibility


class Classifier(ABC):
    """
    Abstract class that models a big5 traits classifier.
    The classifier can be trained on a given document set
    and can predict the trait of a given text.
    """

    @abstractmethod
    def train(self, x, y):
        """
        :param x: list of essays; list[string]
        :param y: torch tensor with targets; torch.tensor(n,5)
        :return: returns nothing
        """
        pass

    @abstractmethod
    def classify(self, example_x, example_y):
        """
        :param example_x: text to be classified; string
        :param example_y: torch tensor with targets for given text; torch.tensor
        :return: predicted labels; torch.tensor(5,)
        """
        pass


class MCCBaseline(Classifier):
    """
    Majority class classifier
    """

    def __init__(self):
        self.labels = None

    def train(self, x, y):
        count_true = [0, 0, 0, 0, 0]
        for i in range(len(x)):
            for j in range(5):
                count_true[j] += 1 if x[i][j] else 0
        self.labels = torch.tensor([count > len(x) / 2 for count in count_true])

    def classify(self, example_x, example_y):
        return self.labels


class RandomBaseline(Classifier):
    """
    Random class classifier
    """

    def train(self, x, y):
        pass

    def classify(self, example_x, example_y):
        return [np.random.random() > 0.5 for _ in range(5)]


if __name__ == '__main__':
    setup_torch_reproducibility(72)
    x, y = load_dataset()

    # ~~~~ MCCBaseline ~~~~ #
    model = MCCBaseline()
    model.train(x, y)
    for ex_x, ex_y in zip(x, y):
        label = model.classify(ex_x, ex_y)

    print("MCCBaseline:")
    scores = eval.eval(model, x, y)
    for k in scores.keys():
        print(scores[k])

    # ~~~~ RandomBaseline ~~~~ #
    model = RandomBaseline()
    model.train(x, y)
    for ex_x, ex_y in zip(x, y):
        label = model.classify(ex_x, ex_y)

    print("RandomBaseline:")
    scores = eval.eval(model, x, y)
    for k in scores.keys():
        print(scores[k])
