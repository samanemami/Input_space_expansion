# %%
import numpy as np
from ERC import erc
from SST import sst
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


X, y = make_regression(n_targets=5)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# model = GradientBoostingRegressor(n_estimators=100)
# model = sst(model=model,
#             cv=3,
#             seed=2,
#             path='SSTModels',
#             )
# model._clear()
# model.fit(x_train, y_train)
# model.score(x_test, y_test)

model_ = GradientBoostingRegressor(n_estimators=100)
model = erc(model=model_,
            cv=3,
            chain=3,
            seed=5,
            path="ERCModels",
            )
model._clear()
# model.fit(X, y)
model._fit_chain(x_train, y_train)
# pred = model.predict(x_test)
# model.score(x_test, y_test)

