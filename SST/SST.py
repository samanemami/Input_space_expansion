import pickle
import numpy as np
from inspect import isclass
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class sst():
    def __init__(self,
                 model,
                 random_state=None,
                 verbose=0):

        self.model = model
        self.random_state = random_state
        self.verbose = verbose

    def _kfold(self, X, y, i):
        n = y.shape[1]
        kfold = KFold(n_splits=n, shuffle=True,
                      random_state=self.random_state)
        index = list(kfold.split(X, y))

        x_train, x_test = X[index[i][0]], X[index[i][1]]
        y_train, y_test = y[index[i][0]], y[index[i][1]]

        self.index = index

        return x_train, x_test, y_train, y_test

    def _first_stage(self, X, y):

        y_hat = []
        self.score_1 = []
        n = y.shape[1]
        for i in range(n):
            x_train, x_test, y_train, y_test = self._kfold(X=X, y=y, i=i)
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

        # Build an array from predicted arrays with different sizes
        # Due to the cross-validation indices.
        lens = [len(i) for i in y_hat]
        max_ = max(lens)
        arr = np.zeros((len(y_hat), max_), int)
        mask = np.arange(max_) < np.array(lens)[:, None]
        arr[mask] = (np.concatenate(y_hat))
        if self.verbose > 0:
            print("{0} dumped models predicted {0} targets. \n The y_hat size is {1}".format(
                arr.shape[0], arr.shape))
        return arr.T

    def _second_stage(self, X, y):
        # Dumps trained model over augmented input variables
        y_hat = self._first_stage(X, y)
        n = y.shape[1]
        for i in range(n):
            x_train, x_test, y_train, y_test = self._kfold(X=X, y=y, i=i)
            x_train_ = np.append(x_train, y_hat, axis=1)
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

        if not isclass(self):
            raise ValueError('The model needs to be fitted first.')
        else:
            return self.score_1
