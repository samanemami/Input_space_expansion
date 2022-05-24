# %%

import os
from sklearn.datasets import make_regression
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from SST import sst
import pickle

X, y = make_regression(n_targets=3)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = sst(model=GradientBoostingRegressor(n_estimators=100), cv=3,
            verbose=0,
            random_state=None,
            path='SST_Models',
            clear=False)
model.fit(x_train, y_train)
model.score(x_test, y_test)
model.predict(x_test)
#%%
x_test.shape