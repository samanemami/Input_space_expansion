import functools
from abc import abstractmethod
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import check_cv, train_test_split


class BaseEstimator:
    """Base class for SST and ERC."""

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
                                   multioutput="raw_values",
                                   squared=False)

        return score
