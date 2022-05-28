# Author: Seyedsaman Emami
# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import copy
import random
import numpy as np
from Base import _base
from sklearn.model_selection import KFold


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

        kfold = KFold(n_splits=self.cv,
                      shuffle=True,
                      random_state=self.seed)

        pred = np.zeros_like(y)

        models = np.empty((self.n, 1), dtype=object)

        for i, perm in enumerate(self.permutation):

            # Build a new variable and deep copy the object
            exec(f'model_{perm} = copy.deepcopy(self.model)')
            exec(f'model_{perm}.fit(X, y[:, perm])')
            # Save the trained model as a binary object in the NumPy array
            exec(f'models[perm, 0] = model_{perm}')

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

        return models

    def fit(self, X, y):

        self.n = y.shape[1]
        self.chains = []
        self.permutation_chains = np.zeros((self.n, self.chain))
        for chain in range(self.chain):
            self.permutation = np.random.permutation(self.n)
            self.permutation_chains[:, chain] = self.permutation
            self.chains.append(self._fit_chain(X, y))
        return self

    def predict(self, X):
        pred = np.zeros((X.shape[0], self.n))

        for ch, chain in enumerate(self.chains):
            permutation = self.permutation_chains[:, ch]
            for i, perm in enumerate(permutation):
                i += 1
                model = chain[perm][0]
                pred[:, perm] = model.predict(X)

                X = np.append(X, pred[:, perm][:, np.newaxis], axis=1)
                if i == len(permutation):
                    return exec(f'pred_{ch} = pred')

        return pred

    def _get_permutation(self):
        return self.permutation
