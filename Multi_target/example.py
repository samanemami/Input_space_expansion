# %%
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
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


model_ = BaggingRegressor(n_estimators=100, random_state=5)
model = erc(model=model_,
            cv=3,
            chain=3,
            seed=5,
            path="ERCModels",
            )
model.fit(X, y)

model.predict(x_test)



# %%
