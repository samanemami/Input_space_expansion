
import warnings
from Multi_target import sst, erc
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")

X, y = make_regression(n_targets=3, n_samples=200)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


base_model = BaggingRegressor(n_estimators=100,
                              random_state=5)


model = erc(model=base_model,
            cv=2,
            chain=3,
            seed=1,
            direct=False,
            verbose=False,
            ranking=True
            )
model.fit(x_train, y_train)

