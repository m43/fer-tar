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
    def classify(self, x, y):
        """
        :param x: list of essays; list[string]
        :param y: torch tensor with targets; torch.tensor(n,5)
        :return: predicted labels; torch.tensor(n,5)
        """
        pass

class RandomBaseline(Classifier):
    """
    Random class classifier
    """

    def train(self, x, y):
        pass

    def classify(self, x, y):
        vals = torch.randn((x.size(0), 5))
        vals[vals >= 0] = 1.
        vals[vals < 0] = 0.
        return vals


if __name__ == '__main__':
    setup_torch_reproducibility(72)
    x, y = load_dataset()

    # # ~~~~ MCCBaseline ~~~~ #
    # model = MCCBaseline()
    # model.train(x, y)
    # for ex_x, ex_y in zip(x, y):
    #     label = model.classify(ex_x, ex_y)
    #
    # print("MCCBaseline:")
    # scores = eval.eval(model, x, y)
    # for k in scores.keys():
    #     print(scores[k])

    # ~~~~ RandomBaseline ~~~~ #
    model = RandomBaseline()
    model.train(x, y)
    for ex_x, ex_y in zip(x, y):
        label = model.classify(ex_x, ex_y)

    print("RandomBaseline:")
    scores = eval.eval(model, x, y)
    for k in scores.keys():
        print(scores[k])
