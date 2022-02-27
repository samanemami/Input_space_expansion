# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn import datasets as dts
from training_erc import train_model as model

np.random.seed(123)
random_state = 123

m = 3
X, y = dts.make_regression(n_samples=500, n_features=5, n_targets=m)

cv_in = 2
cv_out = 2


def input(X, y, i):
    X = np.append(X, y[:, i][:, np.newaxis], axis=1)
    return X


kfold = KFold(n_splits=cv_out, shuffle=True, random_state=random_state)
n_targest = y.shape[1] * 2
scores = np.zeros((cv_out, n_targest))
scores = pd.DataFrame(scores, columns=pd.RangeIndex(0, scores.shape[1], 1))

for _, (train_index, test_index) in enumerate(kfold.split(X, y)):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train m models for m outputs
    for i in range(0, y_train.shape[1], 1):

        model = RandomForestRegressor()
        model.fit(x_train, y_train[:, i])
        score = model.score(x_test, y_test[:, i])
        scores.iloc[_, i] = score
        mapping = {scores.columns[i]: 'target_'+str(i)}
        scores = scores.rename(columns=mapping)

    # ERC training
    # Train the firs model without augmentation

    i += 1
    score_ = []
    Y_train = y_train[:, 0]
    pred_ = np.zeros((y_test.shape[0], cv_in))

    # Train and predict the first output
    kfold_in = KFold(n_splits=cv_in, shuffle=True, random_state=random_state)
    for cv_, (train_index, test_index) in enumerate(kfold_in.split(x_train, Y_train)):
        dftrain, dfeval = x_train[train_index], x_train[test_index]
        ytrain, yeval = Y_train[train_index], Y_train[test_index]

        model = RandomForestRegressor()
        model.fit(dftrain, ytrain)

        pred_[:, cv_] = model.predict(x_test)

        score = model.score(x_test, y_test[:, 0])
        score_.append(score)

    pred_ = np.mean(pred_, axis=1)

    scores.iloc[_, i] = np.mean(score_)
    mapping = {scores.columns[i]: 'D\'_target_0'}
    scores = scores.rename(columns=mapping)

    # Train m models over transformed input data (augmented by m-1 output)

    j = 0
    score_ = []
    intrain_ = []
    pred = np.zeros((y_test.shape[0], cv_in))

    while j < y_train.shape[1]:
        i += 1
        x_train = input(x_train, y_train, j)
        x_test = np.append(x_test, pred_[:, np.newaxis], axis=1)

        if j+1 < y_train.shape[1]:
            Y_train = y_train[:, j+1]
        else:
            break

        for cv_, (train_index, test_index) in enumerate(kfold_in.split(x_train, Y_train)):
            dftrain, dfeval = x_train[train_index], x_train[test_index]
            ytrain, yeval = Y_train[train_index], Y_train[test_index]

            model = RandomForestRegressor()
            model.fit(dftrain, ytrain)

            pred[:, cv_] = model.predict(x_test)

            intrain_.append(model.score(dfeval, yeval))
            score = model.score(x_test, y_test[:, j+1])
            score_.append(score)


       
        pred_ = np.mean(pred, axis=1)

        intrain = np.mean(intrain_, axis=0)
        scores.iloc[_, i] = np.mean(score_)

        mapping = {scores.columns[i]: 'D\'_target_' + str(j+1)}
        scores = scores.rename(columns=mapping)
        j += 1
