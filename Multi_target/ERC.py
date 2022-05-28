# Author: Seyedsaman Emami
# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import copy
import random
import numpy as np
from Base import _base
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor


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

    def _fit_chain(self, X, y):

        n = random.randint(0, 100)
        np.random.seed(n)

        self._check_params()

        kfold = KFold(n_splits=self.cv,
                      shuffle=True,
                      random_state=self.seed)

        self.permutation = np.random.permutation(y.shape[1])
        self.n = y.shape[1]
        pred = np.zeros_like(y)

        self.models = np.empty((self.n, 1), dtype=object)

        for i, perm in enumerate(self.permutation):
            exec(f'model_{perm} = copy.deepcopy(self.model)')
            exec(f'model_{perm}.fit(X, y[:, perm])')
            exec(f'self.models[perm, 0] = model_{perm}')
            splits = list(kfold.split(X, y))
            i += 1
            if i == len(self.permutation):
                break

            for (train_index, test_index) in splits:
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model_ = self.model
                model_.fit(x_train, y_train[:, perm])

                # meta-variable generation
                pred[test_index, perm] = model_.predict(x_test)

            X = np.append(X, pred[:, perm][:, np.newaxis], axis=1)

    def fit(self, X, y):
        self._fit_chain(X, y)

    def predict(self, X):
        pred = np.zeros((X.shape[0], self.n))
        i += 1
        for i, perm in enumerate(self.permutation):
            model = self.models[perm][0]
            pred[:, perm] = model.predict(X)

            X = np.append(X, pred[:, perm][:, np.newaxis], axis=1)
            if i == len(self.permutation):
                break

    def _get_permutation(self):
        return self.permutation
