# %%

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from ERC import erc
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
import warnings
warnings.simplefilter("ignore")

X, y = make_regression(n_targets=5)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


model_ = MLPRegressor()
# model_ = GradientBoostingRegressor(n_estimators=100)
model = erc(model=model_,
            cv=3,
            chain=1,
            seed=5,
            path="ERCModels",
            )
model.fit(X, y)
pred = model.predict(x_test)
# %%

X, y = make_regression(n_targets=3)

models = np.empty((10, 1), dtype=object)
# %%
for i in range(9):
    model = GradientBoostingRegressor(n_estimators=i+1)
    model.fit(X, y[:, 0])
    X = np.append(X, y[:, 0][:, np.newaxis], axis=1)

    models.copy
    # exec(f'model_{i} = model.copy')
    exec(f'model_{i} = model.fit(X, y[:, 0])')
    exec(f'models[i, 0] = model_{i}')
# %%
models
