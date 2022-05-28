# %%
import warnings
from Multi_target import sst, erc
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

warnings.simplefilter("ignore")

X, y = make_regression(n_targets=3, n_samples=700)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
# %%




model = BaggingRegressor(n_estimators=100, random_state=5)
model = sst(model=model,
            cv=3,
            seed=1,
            path="ERCModels",)


model.fit(x_train, y_train)
pred = model.predict(x_test)
# a = model.score(x_test, y_test)

# %%

model = BaggingRegressor(n_estimators=100, random_state=5)
# model = erc(model=model,
#             cv=2,
#             chain=3,
#             seed=5,
#             path="ERCModels",
#             )
model.fit(x_train, y_train)
pred = model.predict(x_test)
# a = model.score(x_test, y_test)
