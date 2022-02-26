# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
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
n_targest = y.shape[1] * 2
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
    i += 1
    cv_results = cross_val_score(estimator=model, X=x_train, y=y_train[:, 0],
                                 cv=3, scoring='r2')
    scores.iloc[:, i] = np.mean(cv_results)
    mapping = {scores.columns[i]: 'D\'_target_0'}
    scores = scores.rename(columns=mapping)
    j = 0
    X_train = x_train
    while j < y_train.shape[1]:
        i += 1
        X_train = input(X_train, y_train, j)
        if j+1 < y_train.shape[1]:
            Y_train = y_train[:, j+1]
        else:
            break
        cv_results = cross_val_score(estimator=model, X=X_train, y=Y_train,
                                     cv=10, scoring='r2')
        scores.iloc[:, i] = np.mean(cv_results)
        mapping = {scores.columns[i]: 'D\'_target_' + str(j+1)}
        scores = scores.rename(columns=mapping)
        j += 1
# %%
