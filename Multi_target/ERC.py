# Author: Seyedsaman Emami
# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import pickle
import os
import random
import numpy as np
from Base import _base
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class erc(_base.BaseEstimator):
    def __init__(self,
                 model,
                 cv=3,
                 chain=1,
                 seed=1,
                 path=None,
                 ):

        self.model = model
        self.cv = cv
        self.chain = chain
        self.seed = seed
        self.path = path

    def _fit_chain(self, X, y, dir):

        n = random.randint(0, 100)
        np.random.seed(n)

        self._check_params()

        kfold = KFold(n_splits=self.cv,
                      shuffle=True,
                      random_state=self.seed)

        self.permutation = np.random.permutation(y.shape[1])
        self.n = y.shape[1]
        pred = np.zeros_like(y)

        # self.models = np.empty((self.n, 1), dtype=object)

        self.models = {}
        keys = ['permutation_'+str(i) for i in self.permutation]
        for i, perm in enumerate(self.permutation):
            model = self.model
            model.fit(X, y[:, perm])
            self.models[perm] = model
            model_name = os.path.join(dir, 'h'+str(perm))
            pickle.dump(model, open(model_name, "wb"))
            splits = list(kfold.split(X, y))

            i += 1
            if i == len(self.permutation):
                break

            for (train_index, test_index) in splits:
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = self.model
                model.fit(x_train, y_train[:, perm])

                # meta-variable generation
                pred[test_index, perm] = model.predict(x_test)

            X = np.append(X, pred[:, perm][:, np.newaxis], axis=1)

    def fit(self, X, y):
        for chain in range(self.chain):
            path = os.path.join(self.path, str(chain))
            if not os.path.isdir(path):
                os.makedirs(path)
            self._fit_chain(X, y, path)
        return self

    def predict(self, X):
        pred = np.zeros((X.shape[0], self.n))
        for perm in self.permutation:
            model = self.models[perm]
            pred[:, perm] = model.predict(X)

    # def predict(self, X):
    #     subdir = [i for i in os.listdir(self.path)]
    #     chains = [os.path.join(self.path, dir_) for dir_ in subdir]
    #     for i, j in enumerate(chains):
    #         for perm in self.permutation:
    #             model = pickle.load(
    #                 open(os.path.join(chains[i], 'h'+str(perm)), "rb"))

    def _get_permutation(self):
        return self.permutation
