# %%
from SST import sst
import sklearn.datasets as dts
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

X, y = dts.make_regression(n_targets=3, n_samples=1000)

x_train, x_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=1)

if __name__ == "__main__":

    model = sst(model=BaggingRegressor(n_estimators=100),
                verbose=1)
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
