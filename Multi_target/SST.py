# Author: Seyedsaman Emami
# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import os
import pickle
import numpy as np
from Base import _base
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class sst(_base.BaseEstimator):
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

        self.n = y.shape[1]
        kfold = KFold(n_splits=self.cv,
                      shuffle=True,
                      random_state=self.random_state)

        pred = np.zeros_like(y)
        self.splits = list(kfold.split(X, y))

        # 1st training stage
        for i in range(self.n):
            for _, (train_index, test_index) in enumerate(self.splits):

                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = self.model
                model.fit(x_train, y_train[:, i])

                # meta-variable generation
                pred[test_index, i] = model.predict(x_test)

                # Dumping trained models of the 1st stage
                model_name = os.path.join(self.path, 'h'+str(i)+'s'+str(_))
                pickle.dump(model, open(model_name, "wb"))

            if self.verbose > 0:
                print("First stage model_{0}, for target {1} is dumped.".format(
                    model_name, i))
        self.score_ = mean_squared_error(y, pred)

        # 2nd training stage
        X = np.append(X, pred, axis=1)
        for i in range(self.n):

            model = self.model
            model.fit(X, y[:, i])

            # Dumping trained models of the 2nd stage
            model_name = os.path.join(self.path, 'h\''+str(i))
            pickle.dump(model, open(model_name, "wb"))

            if self.verbose > 0:
                print("Second stage model_{0}, for target {1} is dumped.".format(
                    model_name, i))

        return self

    def predict(self, X):

        pred_ = np.zeros((X.shape[0], self.n))
        pred = np.zeros_like(pred_)

        for i in range(self.n):
            for _, (train_index, test_index) in enumerate(self.splits):
                model_name = os.path.join(self.path, 'h'+str(i)+'s'+str(_))
                model = pickle.load(open(model_name, "rb"))
                pred_[test_index, i] = model.predict(X[test_index])

        self.pred = pred_
        X = np.append(X, pred_, axis=1)
        for i in range(self.n):
            model_name = os.path.join(self.path, 'h\''+str(i))
            model = pickle.load(open(model_name, "rb"))
            pred[:, i] = model.predict(X)

        # Removing the dumped models
        if self.clear:
            for root, _, files in os.walk(self.path):
                for models in files:
                    os.remove(os.path.join(root, models))

        return pred

    def _get_intrain_score(self):
        return self.score_

    def _get_first_stage_pred(self):
        return self.pred

    def get_split_indices(self):
        return self.splits
