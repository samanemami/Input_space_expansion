import warnings
from Multi_target import sst, erc
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")

X, y = make_regression(n_targets=3, n_samples=700)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


base_model = BaggingRegressor(n_estimators=100,
                              random_state=5)


def sst_():
    model = sst(model=base_model,
                cv=3,
                seed=1
                )
    model.fit(x_train, y_train)

    return model


def erc_():
    model = erc(model=base_model,
                cv=2,
                chain=3,
                seed=5,
                )
    model.fit(x_train, y_train)

    return model


if __name__ == "__main__":
    model = sst_()
    print("\n", "RMSE_SST: \n", model.score(x_test, y_test))

    print("---------")

    model = erc_()
    print("\n", "RMSE_ERC: \n", model.score(x_test, y_test))
