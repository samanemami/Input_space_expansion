import pickle
import numpy as np
import multiprocessing as mpc
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import check_cv, train_test_split


class sst():
    def __init__(self,
                 model,
                 verbose=0):

        self.model = model
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
            pool = mpc.Pool(2)
            result = pool.starmap(self._kfold, [(X, i), (y, i)])
            x_train, x_test = result[0][0], result[0][1]
            y_train, y_test = result[1][0], result[1][1]

            model = self.model
            model.fit(x_train, y_train[:, i])
            self.score_1.append(mean_squared_error(y_test[:, i],
                                                   model.predict(
                x_test),
                squared=False))

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

        return y_hat

    def _second_stage(self, X, y):
        # Dumps trained model over augmented input variables
        y_hat = self._first_stage(X, y)
        n = y.shape[1]
        j = n
        for i in range(n):
            j += 1
            pool = mpc.Pool(2)
            result = pool.starmap(self._kfold, [(X, j), (y, j)])
            x_train = result[0][0]
            y_train = result[1][0]

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
        score = mean_squared_error(y_true=y,
                                   y_pred=pred,
                                   multioutput="raw_values")
        return score

    def _get_intrain_score(self):
        return self.score_1
