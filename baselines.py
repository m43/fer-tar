import ds_load
import eval
import random


class MFCBaseline:
    """
    Majority class classifier
    """
    def __init__(self):
        self.labels = None

    def train(self, dataset):
        """
        :param dataset: the dataset; list[list[str, str, bool, bool, bool, bool, bool]]
        :return: returns nothing
        """
        count_true = [0, 0, 0, 0, 0]
        for i in range(len(dataset)):
            for j in range(5):
                count_true[j] += 1 if dataset[i][j + 2] else 0
        self.labels = [count > len(dataset) / 2 for count in count_true]

    def classify(self, example):
        """
        :param example: text to be classified; string
        :return: predicted labels; list[bool]
        """
        return self.labels


class RandomBaseline:
    def train(self, dataset):
        pass

    def classify(self, example):
        return [random.random() > 0.5 for i in range(5)]


if __name__ == '__main__':
    ds = ds_load.load_dataset('./dataset/essays.csv')

    # ~~~~ MFCBaseline ~~~~ #
    model = MFCBaseline()
    model.train(ds)
    for ex in ds:
        label = model.classify(ex[2])

    print("MFCBaseline:")
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
