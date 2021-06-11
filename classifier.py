from abc import abstractmethod, ABC
from typing import Tuple

import torch
from sklearn.svm import SVC
from torch.utils.data import DataLoader

import fc
import rnn
from dataset import TRAITS
from rnn import LSTM

import os
import pickle


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
    def train(self, data: Tuple[DataLoader, DataLoader, DataLoader, DataLoader], **kwargs):
        """
        Trains the classifier on one of the dataset splits provided in the data tuple.
        The indices of elements in the tuple correspond to the following splits:
            0   -   train split
            1   -   validation split
            2   -   train+validation split
            3   -   test split

        :param data: tuple of four datasets (train, valid, trainval, test); tuple(torch.DataLoader)
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
        for i, clf in enumerate(self.clfs):
            if kwargs['debug_print']:
                print(f"Training '{TRAITS[i]}' classifier: {type(clf)} ...")
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

    def save(self, name):
        os.mkdir(f"./persist/{name}/")
        for i, clf in enumerate(self.clfs):
            if isinstance(clf, SVMClassifier):
                f = open(f"./persist/{name}/{i}-{TRAITS[i]}-{clf.in_dim}.svm", 'wb')
                s = pickle.dumps(clf.svm)
                f.write(s)
                f.close()
            elif isinstance(clf, FCClassifier):
                clf.model.to(device=torch.device('cpu'))
                torch.save(clf.model.state_dict(), f"./persist/{name}/{i}-{TRAITS[i]}-{','.join([str(d) for d in clf.model.dims])}.fc")
            else:
                raise ValueError(f"Save not implemented for classifier {clf}")

    def load(self, name, **kwargs):
        self.clfs = []
        for i, file in enumerate(sorted(os.listdir(f"./persist/{name}/"))):
            if file.endswith(".fc"):
                dims = [int(d) for d in file[:-3].split('-')[2].split(',')]
                kwargs['neurons_per_layer'] = dims
                clf = FCClassifier(i, **kwargs)

                clf.model.load_state_dict(torch.load(f"./persist/{name}/{file}"))
                clf.model.to(device=kwargs['device'])
                self.clfs.append(clf)
            elif file.endswith(".svm"):
                kwargs['in_dim'] = int(file[:-4].split('-')[2])
                clf = SVMClassifier(i, **kwargs)

                f = open(f"./persist/{name}/{file}", 'rb')
                svm = pickle.loads(f.read())
                f.close()

                clf.svm = svm
                self.clfs.append(clf)

    def __str__(self):
        return 'Compound Classifier:\n' + '\n'.join(['\t' + str(c) for c in self.clfs])


class HackyCompoundClassifier:
    def __init__(self, hooks):
        self.clfs = [hook(i, **kwargs) for i, (hook, kwargs) in enumerate(hooks)]

    def train(self, data_auth, **kwargs):
        data = [entry[0] for entry in data_auth]
        for i, clf in enumerate(self.clfs):
            if kwargs['debug_print']:
                print(f"Training '{TRAITS[i]}' classifier: {type(clf)} ...")
            clf.train(data, **kwargs)

    def forward(self, data):
        raise NotImplementedError("Ništa to...")

    def classify(self, data_auth):
        data, authors = data_auth
        data = DataLoader(dataset=data.dataset, batch_size=data.batch_size, collate_fn=data.collate_fn, shuffle=False)

        preds = torch.zeros((len(data.dataset), 5))
        trues = torch.zeros((len(data.dataset), 5))
        for i, clf in enumerate(self.clfs):
            clf_preds, clf_true = clf.classify(data)
            preds[:, i] = clf_preds.flatten()
            trues[:, i] = clf_true.flatten()

        results = {author_id: [torch.zeros((5,)), torch.zeros((5,)), 0] for author_id in authors}

        for auth_id, pred, true in zip(authors, preds, trues):
            results[auth_id][0] += pred.flatten()
            results[auth_id][1] = true.flatten()
            results[auth_id][2] += 1

        preds = torch.zeros((len(results.keys()), 5))
        trues = torch.zeros((len(results.keys()), 5))
        for i, (auth_id, (pred, true, count)) in enumerate(results.items()):
            pred /= count
            preds[i] = pred
            trues[i] = true

        preds[preds >= 0.5] = 1.
        preds[preds < 0.5] = 0.
        return preds, trues

    def __str__(self):
        return 'Compound Classifier:\n' + '\n'.join(['\t' + str(c) for c in self.clfs])

class SVMClassifier(TraitClassifier):
    def __init__(self, index, **kwargs):
        super().__init__(index)
        c, gamma, dfs, kernel = [kwargs[k] for k in ['c', 'gamma', 'decision_function_shape', 'kernel']]
        self.svm = SVC(kernel=kernel, decision_function_shape=dfs, C=c, gamma=gamma)
        self.in_dim = kwargs['in_dim']

    def _loader_to_tensor(self, loader):
        N = len(loader.dataset)
        x, y = torch.empty((N, self.in_dim)), torch.empty((N, 1))
        start, end = 0, loader.batch_size
        for i, batch in enumerate(loader):
            xb, yb = batch
            x[start:end] = xb
            y[start:end] = yb[:, self.index:self.index + 1]
            start, end = end, min(end + len(xb), N)
        return x, y

    def train(self, data, **kwargs):
        _, _, trainval, _ = data
        x, y = self._loader_to_tensor(trainval)
        np_trx, np_try = x.cpu().numpy(), y.cpu().numpy()
        self.svm.fit(np_trx, np_try.flatten())

    def forward(self, data):
        x, y = self._loader_to_tensor(data)
        np_x = x.cpu().numpy()
        return torch.from_numpy(self.svm.predict(np_x)), y

    def classify(self, data):
        return self.forward(data)

    def __str__(self):
        return f"{self.index + 1}. SVMClassifier[{TRAITS[self.index]}]"


class FCClassifier(TraitClassifier):
    def __init__(self, index, **kwargs):
        super().__init__(index)
        self.model = fc.DeepFC(**kwargs)
        self.device = kwargs['device']

    def train(self, data, **kwargs):
        train, valid, trainval, test = data
        fc.train(self.model, train, valid, trainval, test, index=self.index, **kwargs)

    def forward(self, data: DataLoader):
        N = len(data.dataset)
        scores = torch.zeros((N, 1))
        true = torch.zeros((N, 1))
        start, end = 0, data.batch_size
        with torch.no_grad():
            for i, batch in enumerate(data):
                x, y = batch
                true[start:end] = y[:, self.index:self.index + 1]
                scores[start:end] = self.model.forward(x.to(self.device))
                start = end
                end = min(end + data.batch_size, N)
        return scores, true

    def classify(self, data):
        scores, true = self.forward(data)
        scores[scores >= 0] = 1.
        scores[scores < 0] = 0.
        return scores, true

    def __str__(self):
        return f"{self.index + 1}. FCClassifier[{TRAITS[self.index]}]"


class LSTMClassifier(TraitClassifier):
    def __init__(self, index, **kwargs):
        super().__init__(index)
        self.model = LSTM(**kwargs)
        self.device = kwargs['device']

    def train(self, data, **kwargs):
        self.model.to(self.device)
        train, valid, trainval, test = data
        rnn.train(self.model, train, valid, trainval, test, index=self.index, **kwargs)
        self.model.to(torch.device("cpu"))

    def forward(self, data: DataLoader):
        self.model.to(self.device)
        N = len(data.dataset)
        scores = torch.zeros((N, 1))
        true = torch.zeros((N, 1))
        start, end = 0, data.batch_size
        with torch.no_grad():
            for i, batch in enumerate(data):
                x, y, _ = batch
                true[start:end] = y[:, self.index:self.index+1]
                scores[start:end] = self.model.forward(x.to(self.device))
                start = end
                end = min(end + data.batch_size, N)
        self.model.to(torch.device("cpu"))
        return scores, true

    def classify(self, data):
        self.model.to(self.device)
        scores, true = self.forward(data)
        scores[scores >= 0] = 1.
        scores[scores < 0] = 0.
        self.model.to(torch.device("cpu"))
        return scores, true

    def __str__(self):
        return f"{self.index + 1}. LSTMClassifier[{TRAITS[self.index]}]"



if __name__ == '__main__':
    # TODO: Update test
    # x, y = load_dataset()
    # (train_x, train_y), _, (test_x, test_y) = split_dataset(x, y, 0.3, 0)
    #
    # # ~~~~ SVMClassifier ~~~~ #
    # extractor = DummyExtractor(10000)
    # clf = SVMClassifier(extractor)
    # clf.train(x, y)
    # print("Train:", eval(clf, train_x, train_y))
    # print("Test:", eval(clf, test_x, test_y))
    pass
