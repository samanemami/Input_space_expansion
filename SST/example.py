
from sklearn.model_selection import check_cv
from sklearn.model_selection import KFold
from SST import sst
import sklearn.datasets as dts
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
import numpy as np

X, y = dts.make_regression(n_targets=3, n_samples=1000)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
ss = sst(model=BaggingRegressor(n_estimators=100), verbose=1)
ss.fit(X, y)

