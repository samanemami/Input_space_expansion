import numpy as np
from abc import abstractmethod
from sklearn.metrics import mean_squared_error, r2_score


class BaseEstimator:
    """Base class for SST and ERC."""

    @abstractmethod
    def fit(self, X, y):
        "To fit the estimator"

    @abstractmethod
    def predict(self, X):
        "To return the predicted values"

    def score(self, X, y):
        # Returns RMSE of each target
        pred = self.predict(X)
        score = mean_squared_error(y_true=y,
                                   y_pred=pred,
                                   multioutput="raw_values",
                                   squared=False)

        return score

    def rrmse(self, X, y):

        # Returns Average of Relative Root Mean Squared Error
        pred = self.predict(X)
        score = r2_score(y, pred)

        return np.sqrt(np.abs(1-score))
