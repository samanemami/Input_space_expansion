from sklearn import datasets as dts
from training_erc import train_model as model

X, y = dts.make_regression(n_samples=500,
                           n_features=5,
                           n_targets=3)


if __name__ == '__main__':

    model(X=X,
          y=y, cv_out=3,
          cv_in=4,
          random_state=123,
          title="title")
