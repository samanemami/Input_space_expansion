# Author: Seyedsaman Emami
# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

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
                 verbose=0,
                 random_state=None,
                 path=None,
                 clear=False):

        self.model = model
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state
        self.path = path
        self.clear = clear

    def fit(self, X, y):

        np.random.seed(self.random_state)

        self._check_params()

        kfold = KFold(n_splits=self.cv,
                      shuffle=True,
                      random_state=self.random_state)

        permutation = np.random.permutation(y.shape[1])

        self.n = y.shape[1]
        pred = np.zeros_like(y)
        splits = []

        for i, perm in enumerate(permutation):
            if not i == 0:
                # Augment the input with the real values
                # of previous output in the permutation list
                X = np.append(X, y[:, permutation[perm-1]]
                              [:, np.newaxis], axis=1)
            splits_ = list(kfold.split(X, y))
            for _, (train_index, test_index) in enumerate(splits_):

                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = self.model
                model.fit(x_train, y_train[:, perm])

                # meta-variable generation
                pred[test_index, perm] = model.predict(x_test)

                # Dumping trained models of the 1st stage
                model_name = os.path.join(
                    self.path, 'h'+str(perm)+'s'+str(_))
                pickle.dump(model, open(model_name, "wb"))

            if self.verbose > 0:
                print("model_{0}, for target {1} is dumped.".format(
                    model_name, perm))

        splits.append(splits_)

    def predict(self, X):
        pass
