# %%
from itertools import permutations
from pyexpat import model
from matplotlib.pyplot import axis
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
from sklearn import datasets as dts


def input(X, y, i):
    X = np.append(X, y[:, i][:, np.newaxis], axis=1)
    return X


m = 3
X, y = dts.make_regression(n_samples=500, n_features=5, n_targets=m)


kfold = KFold(n_splits=2, shuffle=True)

# %%
score_x_t1 = []
score_x_t2 = []
score_x_t3 = []


score_xy1_t2 = []
score_xy1_t3 = []

score_xy_t1 = []
score_xy_t1 = []

score_xy2_t3 = []
score_xy2_t1 = []

score_xy3_t2 = []
score_xy3_t1 = []


for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = RandomForestRegressor()

    for i in range(0, y_train.shape[1], 1):

        model.fit(x_train, y_train[:, i])
        score_x_t1.append(model.score(x_test, y_test[:, i]))

    # train the model by considering the output as an input
    # Add the first output

    for i in range(0, y_train.shape[1], 1):
        x_train = input(x_train, y_train, i)
        x_test = input(x_test, y_test, i)
        for j in range(0, y_train.shape[1], 1):
            if i != j:
                model.fit(x_train, y_train[:, j])
                model.score(x_test, y_test[:, j])


# %%

df = pd.DataFrame([])
kfold = KFold(n_splits=5, shuffle=True)
for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = RandomForestRegressor()

    for i in range(0, y_train.shape[1], 1):

        model.fit(x_train, y_train[:, i])
        score = model.score(x_test, y_test[:, i])
        data = pd.Series(score, name=str(i))
        data.columns = data.iloc[0]
        # data = data.drop(data.index[[0]])
        df.iloc[:, 1] = df.append(data)

# %%
d = {'col1': [1, 2], 'col2': [3, 4]}
