# %%
from pyexpat import model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
from sklearn import datasets as dts

X, y = dts.make_regression(n_samples=500, n_features=5, n_targets=3)


kfold = KFold(n_splits=2, shuffle=True)


score_t1 = []
score_t2 = []
score_t3 = []

for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = RandomForestRegressor()
    model.fit(x_train, y_train[:, 0])
    score_t1.append(model.score(x_test, y_test[:, 0]))

    model.fit(x_train, y_train[:, 1])
    score_t2.append(model.score(x_test, y_test[:, 1]))

    model.fit(x_train, y_train[:, 2])
    score_t3.append(model.score(x_test, y_test[:, 2]))

# %%
# np.column_stack((x_train, y_train[: 0]))
y_train[:, 0].shape
