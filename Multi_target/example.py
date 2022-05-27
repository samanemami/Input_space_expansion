# %%

from statistics import mode
from ERC import erc
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
import warnings
warnings.simplefilter("ignore")

X, y = make_regression(n_targets=5)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)



model_ = BaggingRegressor(n_estimators=120)
# model_ = GradientBoostingRegressor(n_estimators=100)
model = erc(model=model_,
            cv=3,
            chain=1,
            seed=5,
            path="ERCModels",
            )
# model._clear()
model.fit(X, y)
pred = model.predict(x_test)
