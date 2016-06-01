import loss

import numpy as np
from scipy.stats import norm


def assert_approx_eq(a, b):
    assert np.all(np.abs(a - b) < 0.000001)


def test_normal_estimator():
    y = np.random.random(100)
    weight = np.random.random(100)
    X = np.zeros((100, 10))

    mean = np.mean(y)
    weighted_mean = np.average(y, weights=weight)
    std = np.std(y)
    weighted_std = np.average((y - weighted_mean) ** 2, weights=weight) ** 0.5

    estimator = loss.NormalEstimator()
    estimator.fit(X, y)
    pred_y = estimator.predict(X)
    assert np.all(pred_y[:, 0] == mean)
    assert np.all(pred_y[:, 1] == std)

    estimator = loss.NormalEstimator()
    estimator.fit(X, y, sample_weight=np.ones(100))
    pred_y = estimator.predict(X)
    assert np.all(pred_y[:, 0] == mean)
    assert np.all(pred_y[:, 1] == std)

    estimator = loss.NormalEstimator()
    estimator.fit(X, y, sample_weight=weight)
    pred_y = estimator.predict(X)
    assert np.all(pred_y[:, 0] == weighted_mean)
    assert np.all(pred_y[:, 1] == weighted_std)


def test_heteroscedastic_normal_loss_function():
    my_loss = loss.HeteroscedasticNormalLossFunction(2)
    y = np.zeros(100)
    pred = np.zeros((100, 2))
    pred[:, 1] = 1
    assert_approx_eq(-np.log(norm(0, 1).pdf(0)), my_loss(y, pred))
    assert_approx_eq(-np.log(norm(0, 1).pdf(0)), my_loss(y, pred, sample_weight=np.ones(100)))
