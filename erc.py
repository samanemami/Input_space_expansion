# %%
from pyexpat import model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from sklearn import datasets as dts

X, y = dts.make_regression(n_samples=7000, n_features=15, n_targets=3)


kfold = KFold(n_splits=10, shuffle=True)
for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = RandomForestRegressor()
    model.fit(x_train, y_train[:, 0])
    model.score
    model.fit(x_train, y_train[:, 1])

    model.fit(x_train, y_train[:, 2])