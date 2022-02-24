# %%
from pyexpat import model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn import datasets as dts


def input(X, y, i):
    X = np.append(X, y[:, i][:, np.newaxis], axis=1)
    return X


m = 3
cv = 2
X, y = dts.make_regression(n_samples=500, n_features=5, n_targets=m)


kfold = KFold(n_splits=cv, shuffle=True)

scores = np.zeros((2, ((m-1) * m)))
scores = pd.DataFrame(scores)

for cv_, (train_index, test_index) in enumerate(kfold.split(X, y)):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = RandomForestRegressor()

    # train the model for different outputs
    for i in range(0, y_train.shape[1], 1):

        model.fit(x_train, y_train[:, i])
        score = model.score(x_test, y_test[:, i])
        scores.iloc[cv_, i] = score

    # train the model by considering the output as
    # an input

    for i in range(0, y_train.shape[1], 1):
        x_train = input(x_train, y_train, i)
        x_test = input(x_test, y_test, i)
        for j in range(0, y_train.shape[1], 1):
            if i != j:
                model.fit(x_train, y_train[:, j])
                score = model.score(x_test, y_test[:, j])
                scores.iloc[cv_, i+y.shape[1]] = score


# %%
