# %%
import os
import numpy as np
import pickle
from posixpath import split
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from SST import sst

X, y = make_regression(n_targets=3)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
model = sst(model=GradientBoostingRegressor(n_estimators=100), cv=3,
            verbose=0,
            random_state=None,
            path='SST_Models',
            clear=False)
model.fit(x_train, y_train)
# model.score(x_test, y_test)
# model.predict(x_test)
# %%
X = x_test
pred_ = np.zeros((X.shape[0], y.shape[1], 3))
splits = model.get_split_indices()
for i in range(y.shape[1]):
    for _, (train_index, test_index) in enumerate(splits):
        model_name = os.path.join('SST_Models', 'h'+str(i)+'s'+str(_))
        model_ = pickle.load(open(model_name, "rb"))
        pred_[:, i, _] = model_.predict(X)

# %%
pred_.shape
pred = np.zeros_like(pred_)
X = np.append(X, pred_, axis=1)
for i in range(y.shape[1]):
    model_name = os.path.join('SST_Models', 'h\''+str(i))
    model = pickle.load(open(model_name, "rb"))
    pred[:, i] = model.predict(X)
