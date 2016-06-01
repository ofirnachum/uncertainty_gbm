__doc__ = """Uncertainty-GBM applied to Boston real-estate data."""


import regressor

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.utils import shuffle


def main():
    boston = datasets.load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    offset = int(X.shape[0] * 0.7)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    clf = regressor.UncertaintyGBM(n_estimators=100, max_depth=4,
                                   learning_rate=0.01, verbose=True)
    clf.fit(X_train, y_train)

    pred_test = clf.predict(X_test)
    mu_test = pred_test[:, 0]
    std_test = pred_test[:, 1]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.title('Predicted mu_test against y_test')
    plt.scatter(y_test, mu_test)

    plt.subplot(2, 2, 2)
    plt.title('Predicted std_test against y_test')
    plt.scatter(y_test, std_test)

    plt.subplot(2, 2, 3)
    plt.title('High-risk/high-reward: mu_test + std_test')
    plt.scatter(y_test, mu_test + std_test)

    plt.subplot(2, 2, 4)
    plt.title('Low-risk/low-reward: mu_test - std_test')
    plt.scatter(y_test, mu_test - std_test)

    plt.show()


if __name__ == '__main__':
    main()
