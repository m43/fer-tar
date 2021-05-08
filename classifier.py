from sklearn.svm import SVC

from baselines import Classifier
from dataset import split_dataset, load_dataset
from eval import eval
from extractor import DummyExtractor


class FCClassifier(Classifier):
    def __init__(self, extractor):
        self.extractor = extractor

    def train(self, x, y):
        classifier = self.init_classifier(x, y)

    def classify(self, example_x, example_y):
        pass

    def init_classifier(self, x, y):
        pass


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
