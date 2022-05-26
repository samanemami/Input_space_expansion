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
                 *,
                 model,
                 cv=3,
                 seed=1,
                 path=None,
                 ):

        self.model = model
        self.cv = cv
        self.seed = seed
        self.path = path

    def fit(self, X, y):

        self._check_params()

        np.random.seed(self.seed)

        self.n = y.shape[1]
        kfold = KFold(n_splits=3,
                      shuffle=True,
                      random_state=self.seed
                      )

        pred = np.zeros_like(y)

        # 1st training stage
        for i in range(self.n):

            model = self.model
            model.fit(X, y[:, i])
            model_name = os.path.join(self.path, 'h'+str(i))
            pickle.dump(model, open(model_name, "wb"))

            for (train_index, test_index) in (kfold.split(X, y)):

                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = self.model
                model.fit(x_train, y_train[:, i])

                # meta-variable generation
                pred[test_index, i] = model.predict(x_test)

        self.score_ = mean_squared_error(y, pred)

        # 2nd training stage
        XX = np.append(X, pred, axis=1)
        for i in range(self.n):

            model = self.model
            model.fit(XX, y[:, i])

            # Dumping trained models of the 2nd stage
            model_name = os.path.join(self.path, 'h_'+str(i))
            pickle.dump(model, open(model_name, "wb"))

        return self

    def predict(self, X):

        # Use the save model
        pred_ = np.zeros((X.shape[0], self.n))
        pred = np.zeros_like(pred_)

        for i in range(self.n):
            model_name = os.path.join(self.path, 'h'+str(i))
            model = pickle.load(open(model_name, "rb"))
            pred_[:, i] = model.predict(X)
        self.pred = pred_

        X = np.append(X, pred_, axis=1)
        for i in range(self.n):
            model_name = os.path.join(self.path, 'h_'+str(i))
            model = pickle.load(open(model_name, "rb"))
            pred[:, i] = model.predict(X)

        return pred

    def _get_intrain_score(self):
        return self.score_

    def _get_first_stage_pred(self):
        return self.pred