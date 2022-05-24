# Author: Seyedsaman Emami
# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import pickle
from Base import _base
from sklearn.metrics import mean_squared_error


class erc(_base.BaseEstimator):
    def __init__(self,
                 model,
                 verbose=0):

        self.model = model
        self.verbose = verbose

    def init_estimating(self, X, y, i):
        model = self.model
        x_train, x_test, y_train, y_test = self._input(X, y[:, i])
        model.fit(x_train, y_train)
        model_name = 'erc_models\h'+str(i)
        pickle.dump(model, open(model_name, 'wb'))

        return mean_squared_error(y_test)
