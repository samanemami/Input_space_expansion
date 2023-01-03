import sys
import functools
import numpy as np
from abc import abstractmethod
from sklearn.metrics import mean_squared_error, r2_score


class BaseEstimator():
    """Abstract base class for SST and ERC."""

    @abstractmethod
    def __init__(self,
                 *,
                 model,
                 cv=3,
                 direct=False,
                 seed=1,
                 verbose=True
                 ):

        self.model = model
        self.cv = cv
        self.direct = direct
        self.seed = seed
        self.verbose = verbose

        """    
        parameters
        ------------
        model : Sklrean ML class, 
            Sklearn ML model to build a SST ensemble model.
        
        cv : int, default=3,
            The number of folds (disjoint parts) for 
            the KFold cross-validation.
        
        seed : int, default=1,
            Seed value to generate a random number.

        direct : bool, default=False,
            If False, returns the cv model.
            If True, returns the direct adaptations

        verbose : bool, default=True,
            Returns the verbosity
            
        """

    @abstractmethod
    def fit(self, X, y):
        "To fit the estimator"

    @abstractmethod
    def predict(self, X):
        "To return the predicted values"

    def score(self, X, y):
        '''Returns RMSE of each target'''
        pred = self.predict(X)
        score = mean_squared_error(y_true=y,
                                   y_pred=pred,
                                   multioutput="raw_values",
                                   squared=False)

        return score

    def rrmse(self, X, y):

        # Returns Average of Relative Root Mean Squared Error
        pred = self.predict(X)
        score = r2_score(y_true=y, y_pred=pred, multioutput='raw_values')

        return np.sqrt(np.abs(1-score))

    def ProgressBar(self, percent, barLen=20):
        sys.stdout.write("\r")
        progress = ""
        for i in range(barLen):
            if i < int(barLen * percent):
                progress += "="
            else:
                progress += " "
        sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
        sys.stdout.flush()

    def trackcalls(func):
        # Check if the function is called.
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wrapper.called = True
            return func(*args, **kwargs)
        wrapper.called = False
        return wrapper
