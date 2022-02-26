# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn import datasets as dts
from training_erc import train_model as model
m = 5
X, y = dts.make_regression(n_samples=500, n_features=5, n_targets=m)


cv = 2


def input(X, y, i):
    X = np.append(X, y[:, i][:, np.newaxis], axis=1)
    return X


kfold = KFold(n_splits=cv, shuffle=True)
n_targest = ((m-1) * m) + y.shape[1]
scores = np.zeros((cv, n_targest))
scores = pd.DataFrame(scores, columns=pd.RangeIndex(0, scores.shape[1], 1))

for cv_, (train_index, test_index) in enumerate(kfold.split(X, y)):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = RandomForestRegressor()

    # train the model for different outputs
    for i in range(0, y_train.shape[1], 1):

        model.fit(x_train, y_train[:, i])
        score = model.score(x_test, y_test[:, i])
        scores.iloc[cv_, i] = score
        mapping = {scores.columns[i]: 'target_'+str(i)}
        scores = scores.rename(columns=mapping)

    # train the model by considering the output as
    # an input

    col = False
    for i in range(0, y_train.shape[1], 1):
        x_train = input(x_train, y_train, i)
        x_test = input(x_test, y_test, i)
        if col:
            if abs(col - col+1) < 1:
                col += 1
            else:
                col = col
        else:
            col = i+y.shape[1]
        for j in range(0, y_train.shape[1], 1):
            if i != j:
                model.fit(x_train, y_train[:, j])
                score = model.score(x_test, y_test[:, j])
                scores.iloc[cv_, col] = score
                mapping = {
                    scores.columns[col]: 'X+target_'+str(i) + '|target_'+str(j)}
                scores = scores.rename(columns=mapping)
                col += 1
# scores = scores.append(scores.mean(axis=0), ignore_index=True)
# scores.to_csv(title + "_score.csv", index=False)
# %%
i = 0
while i < y_train.shape[1]:
    x_train = input(x_train, y_train, i)
    if i > y_train.shape[1]:
        break
    i += 1
# %%
y_train.shape[1]


# %%
for i in range(y_train.shape[1]):
    x_train = input(x_train, y_train, i)
    y_train = y_train[:, ]


# %%
print(x_train.shape)
for i in range(0, y_train.shape[1], 1):
    for j in range(0, y_train.shape[1], 1):
        if i != j:

            x_train = input(x_train, y_train, i)
            #
            print(x_train.shape)

            # scores.iloc[cv_, col] = score
            # mapping = {
            #     scores.columns[col]: 'X+target_'+str(i) + '|target_'+str(j)}
            # scores = scores.rename(columns=mapping)
# input(X, y, )
# %%
print(x_train.shape)
