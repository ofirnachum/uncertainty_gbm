__doc__ = """Demonstration of model on constructed data with plots of mu, std.

Code adapted from
http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html

"""

import regressor

import numpy as np
import matplotlib.pyplot as plt

NUM_SAMPLES = 5000

def mu(x):
    """The mu function to predict."""
    return x * np.sin(x)


def std(x):
    """The std function to predict."""
    return np.abs(0.5 + x ** 1.4 / 6)


def main():
    np.random.seed(1)
    #  first the noiseless case
    X = np.atleast_2d(np.random.uniform(0, 10.0, size=NUM_SAMPLES)).T
    X = X.astype(np.float32)

    # observations
    y = mu(X).ravel()

    noise = np.random.normal(0, 1, size=y.shape) * std(X).ravel()
    y += noise
    y = y.astype(np.float32)

    # mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
    xx = xx.astype(np.float32)

    clf = regressor.UncertaintyGBM(n_estimators=250, max_depth=3,
                                   learning_rate=0.1, min_samples_leaf=9,
                                   min_samples_split=9, verbose=True)

    clf.fit(X, y)

    # make the prediction on the meshed x-axis
    pred = clf.predict(xx)
    pred_mu = pred[:, 0]
    pred_std = pred[:, 1]

    y_pred = pred_mu
    y_lower = pred_mu - 2 * pred_std
    y_upper = pred_mu + 2 * pred_std

    # plot the function, the prediction and the 95% confidence interval
    fig = plt.figure()
    plt.plot(X, y, 'b.', markersize=10, label=u'Observations')
    plt.plot(xx, mu(xx), 'g', linewidth=2, label=u'$f(x) = x\,\sin(x)$')
    plt.plot(xx, y_pred, 'r-', label=u'Prediction')
    plt.plot(xx, y_upper, 'k-')
    plt.plot(xx, y_lower, 'k-')
    plt.fill(np.concatenate([xx, xx[::-1]]),
             np.concatenate([y_upper, y_lower[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% prediction interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
