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

    def _check_params(self):
        if not self.path:
            raise ValueError("Define a path to dump the models.")

    def _clear(self):
        # To clear the dumped models in the directory
        for root, _, models in os.walk(self.path):
            for model in models:
                os.remove(os.path.join(root, model))
