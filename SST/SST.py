import pickle
import numpy as np
import pandas as pd
from inspect import isclass
from sklearn.model_selection import KFold, check_cv, train_test_split
from sklearn.metrics import mean_squared_error


class sst():
    def __init__(self,
                 model,
                 random_state=None,
                 verbose=0):

        self.model = model
        self.random_state = random_state
        self.verbose = verbose

    def _kfold(self, X, i=None):
        #  A K-Fold single shuffled train-test split
        cv = check_cv(cv=[train_test_split(X,
                                           shuffle=True,
                                           random_state=i)])

        index = next(cv.split())
        train, test = index[0], index[1]

        return train, test

    def _first_stage(self, X, y):

        y_hat = []
        self.score_1 = []
        n = y.shape[1]
        for i in range(n):
            x_train, x_test = self._kfold(X=X, i=i)
            y_train, y_test = self._kfold(X=y, i=i)

            model = self.model
            model.fit(x_train, y_train[:, i])
            self.score_1.append(np.sqrt(mean_squared_error(y_test[:, i],
                                                           model.predict(
                                                               x_test),
                                                           squared=False)))

            # Dumping the trained model in a binary mode
            model_name = 'h' + str(i)
            pickle.dump(model, open(model_name, 'wb'))

            # Generate the meta-variables
            model = pickle.load(open(model_name, 'rb'))
            y_hat.append(model.predict(x_train))

            if self.verbose > 0:
                print("First stage model_{0}, for target {1} is dumped.".format(
                    model_name, i))

        y_hat = (np.array(y_hat)).T

        if self.verbose > 0:
            print("{0} dumped models predicted {0} targets. \n The y_hat size is {1}".format(
                y_hat.shape[1], y_hat.shape))
            print('----------')

        return np.array(y_hat)

    def _second_stage(self, X, y):
        # Dumps trained model over augmented input variables
        y_hat = self._first_stage(X, y)
        n = y.shape[1]
        j = n
        for i in range(n):
            j += 1
            x_train = self._kfold(X=X, i=j)[0]
            y_train = self._kfold(X=y, i=j)[0]
            if x_train.shape[0] == y_hat.shape[0]:
                x_train_ = np.append(x_train, y_hat, axis=1)
            else:
                yhat, xtrain = pd.DataFrame(y_hat, columns=None, index=None), pd.DataFrame(
                    x_train, columns=None, index=None)
                x_train_ = (
                    (pd.concat([yhat, xtrain], axis=1)).fillna(0)).values

            model = self.model
            model.fit(x_train_, y_train[:, i])

            # Dumping the trained model
            model_name = 'h\'' + str(i)
            pickle.dump(model, open(model_name, 'wb'))
            if self.verbose > 0:
                print("Second stage model_{0}, for target {1} is dumped.".format(
                    model_name, i))

    def fit(self, X, y):
        self._second_stage(X, y)
        self.n = y.shape[1]

        return self

    def predict(self, X):
        pred_hat = np.zeros((X.shape[0], self.n))
        for i in range(self.n):
            # 1st stage of the prediction
            model = pickle.load(open('h'+str(i), 'rb'))
            pred_hat[:, i] = model.predict(X)

        pred = np.zeros_like(pred_hat)
        x_test = np.append(X, pred_hat, axis=1)
        for i in range(self.n):
            # final stage of the prediction
            model = pickle.load(open('h\''+str(i), 'rb'))
            pred[:, i] = model.predict(x_test)
        return pred

    def score(self, X, y):
        pred = self.predict(X)
        return mean_squared_error(y_true=y,
                                  y_pred=pred,
                                  multioutput="raw_values")

    def get_in_train_score(self):

        if isclass(self):
            raise ValueError('The model needs to be fitted first.')
        else:
            return self.score_1
