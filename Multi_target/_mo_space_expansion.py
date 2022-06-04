# Author: Seyedsaman Emami
# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import random
import numpy as np
from Base import _base
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class erc(_base.BaseEstimator):
    def __init__(self,
                 *,
                 model,
                 cv=3,
                 chain=1,
                 seed=1,
                 ):

        self.model = model
        self.cv = cv
        self.chain = chain
        self.seed = seed

    """ Ensemble of Regressor Chains


    parameters
    ------------
    model : Sklrean ML class, 
        Sklearn ML model to build an ERC ensemble model.
    
    cv : int, default=3,
        The number of folds (disjoint parts) for 
        the KFold cross-validation.
    
    chain : int, default=1,
        The number of Ensemble chains.
        If the chain is equal to 1, the model 
        returns the RC model.
    
    seed : int, default=1,
        Seed value to generate a random number.

    """

    def _fit_chain(self, X, y, chain):

        n = random.randint(0, 100)
        np.random.seed(n)

        kfold = KFold(n_splits=self.cv,
                      shuffle=True,
                      random_state=self.seed)

        pred = np.zeros_like(y)

        models = np.empty((self.n, 1), dtype=object)
        permutation = self.permutation[:, chain]

        for perm in permutation[:-1]:

            # Build a new variable and deep copy the object
            exec(f'model_{perm} = clone(self.model)')
            exec(f'model_{perm}.fit(X, y[:, perm])')
            # Save the trained model as a binary object in the NumPy array
            exec(f'models[perm, 0] = model_{perm}')

            for (train_index, test_index) in kfold.split(X, y):
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model_ = clone(self.model)
                model_.fit(x_train, y_train[:, perm])

                # meta-variable generation
                pred[test_index, perm] = model_.predict(x_test)

            X = np.append(X, pred[:, perm][:, np.newaxis], axis=1)

        return models

    def fit(self, X, y):

        self.n = y.shape[1]
        # Chains include RC models
        self.chains = []
        self.permutation = np.zeros(
            (self.n, self.chain), dtype=np.int32)
        for chain in range(self.chain):
            self.permutation[:, chain] = np.random.permutation(self.n)
            self.chains.append(self._fit_chain(X, y, chain))
        return self

    def predict(self, X):
        pred = np.zeros((X.shape[0], self.n))
        for i, chain in enumerate(self.chains):
            pred_ = np.zeros_like(pred)
            permutation = self.permutation[:, i]
            XX = X
            for perm in permutation[:-1]:
                model = chain[perm][0]
                pred_[:, perm] = model.predict(XX)
                XX = np.append(XX, pred_[:, perm][:, np.newaxis], axis=1)
            pred += pred_

        return (pred_)/(self.n)

    def _get_permutation(self):
        return self.permutation


class sst(_base.BaseEstimator):
    def __init__(self,
                 *,
                 model,
                 cv=3,
                 seed=1,
                 ):

        self.model = model
        self.cv = cv
        self.seed = seed

    """ Stacked single-target

    parameters
    ------------
    model : Sklrean ML class, 
        Sklearn ML model to build a SST ensemble model.
    
    cv : int, default=3,
        The number of folds (disjoint parts) for 
        the KFold cross-validation.
    
    seed : int, default=1,
        Seed value to generate a random number.

    """

    def fit(self, X, y):

        np.random.seed(self.seed)

        self.n = y.shape[1]
        kfold = KFold(n_splits=3,
                      shuffle=True,
                      random_state=self.seed
                      )

        pred = np.zeros_like(y)
        self.models = np.empty((self.n, 2), dtype=object)

        # 1st training stage
        for i in range(self.n):

            exec(f'model_{i} = clone(self.model)')
            exec(f'self.models[i, 0] = model_{i}.fit(X, y[:, i])')

            for (train_index, test_index) in (kfold.split(X, y)):

                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = clone(self.model)
                model.fit(x_train, y_train[:, i])

                # meta-variable generation
                pred[test_index, i] = model.predict(x_test)

        self.score_ = mean_squared_error(y, pred)

        # 2nd training stage
        for i in range(self.n):
            XX = np.append(X, np.delete(pred, i, 1), axis=1)
            exec(f'model_{i} = clone(self.model)')
            exec(f'self.models[i, 1] = model_{i}.fit(XX, y[:, i])')

        return self

    def predict(self, X):

        # Use the save model
        pred_ = np.zeros((X.shape[0], self.n))
        pred = np.zeros_like(pred_)

        for i in range(self.n):
            model = self.models[i, 0]
            pred_[:, i] = model.predict(X)
        self.pred = pred_

        for i in range(self.n):
            XX = np.append(X, np.delete(pred_, i, 1), axis=1)
            model = self.models[i, 1]
            pred[:, i] = model.predict(XX)

        return pred

    def _get_intrain_score(self):
        return self.score_

    def _get_first_stage_pred(self):
        return self.pred
