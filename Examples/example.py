from sklearn import datasets as dts
from training_erc import train_model as model
m = 3
X, y = dts.make_regression(n_samples=500, n_features=5, n_targets=m)


if __name__ == '__main__':
    model(X=X, y=y, cv=3, m=m)
