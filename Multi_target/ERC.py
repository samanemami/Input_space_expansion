# Author: Seyedsaman Emami
# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

from itertools import permutations
import os
import pickle
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

    def _augmenting(self, X, y, i):

        x = np.append(X, y[:, i][:, np.newaxis], axis=1)

        return x

    def _init_estimator(self, X, y, perm):

        model = self.model
        model.fit(X, y[:, perm])
        pred = model.predict(X)

        return pred, model

    def _fit_chain(self, X, y):

        np.random.seed(self.seed)

        self._check_params()
        # Add a loop for the number of chains (ERC)
        # Number of chains: self.chain=1
        kfold = KFold(n_splits=self.cv,
                      shuffle=True,
                      random_state=self.seed)

        self.permutation = np.random.permutation(y.shape[1])
        permutation = self.permutation
        self.n = y.shape[1]
        pred = np.zeros_like(y)

        # self.models = np.empty((self.n, 1), dtype='object')
        self.models = []
        perm_ = 0

        while perm_ < len(permutation):
            perm = next(iter(permutation))
            pred[:, perm], model = self._init_estimator(X, y, perm)
            # self.models[permutation[0], :] = model
            self.models.append(model)
            X = self._augmenting(X, pred, perm)
            splits = list(kfold.split(X, y))
            permutation = permutation[1:]
            perm = next(iter(permutation))
            for (train_index, test_index) in splits:
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = self.model
                model.fit(x_train, y_train[:, perm])

                # meta-variable generation
                pred[test_index, perm] = model.predict(x_test)


    def fit(self, X, y):
        chains = np.random.permutation(self.chain)
        for chain in chains:
            self._fit_chain(self, X, y)

    def predict(self, X):
        pred = np.zeros((X.shape[0], self.n))

        perm = self.permutation[0]
        model_name = os.path.join(self.path, 'h'+str(perm))
        model = pickle.load(open(model_name, "rb"))

        pred[:, perm] = model.predict(X)

        X = self._augmenting(X, pred, perm)

        for i, perm in enumerate(self.permutation[1:]):
            model_name = os.path.join(self.path, 'h'+str(perm))
            # model = pickle.load(open(model_name, "rb"))
            pred[:, perm] = model.predict(X)

            X = self._augmenting(X, pred, perm)

            if i+1 == len(self.permutation[1:]):
                break
        return pred

    def _get_permutation(self):
        return self.permutation
