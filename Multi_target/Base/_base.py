from abc import abstractmethod
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import check_cv, train_test_split


class BaseEstimator:
    """Base class for SST and ERC."""

    def _kfold(self, X, i=None):

        #  A K-Fold single shuffled train-test split
        cv = check_cv(cv=[train_test_split(X,
                                           shuffle=True,
                                           random_state=i)])

        index = next(cv.split())
        train, test = index[0], index[1]

        return train, test

    @abstractmethod
    def fit(self, X, y):
        "To fit the estimator"

    @abstractmethod
    def predict(self, X):
        "To return the predicted values"

    def score(self, X, y):
        pred = self.predict(X)
        score = mean_squared_error(y_true=y,
                                   y_pred=pred,
                                   multioutput="raw_values")
        return score
