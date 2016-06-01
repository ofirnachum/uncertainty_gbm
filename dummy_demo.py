__doc__ = """Demonstration of uncertainty-GBM on constructed data.

In addition to negative log likelihood (NLL), results are also provided in
terms of R2 on mu and std to give a more interpretable indication that the
model is working as intended.

"""

import loss
import regressor

import numpy as np
import random
from sklearn.metrics import r2_score

NUM_TRAIN = 100000
NUM_TEST = 10000
NUM_FEATURES = 10
SEED = 88

MU_SCALE = 8  # scale for random coefficients for mu
STD_SCALE = 1  # scale for random coefficients for std


class SimpleLinear(object):
    def __init__(self, num_features):
        self.mu_coeffs = np.random.random(num_features) * MU_SCALE
        self.std_coeffs = np.random.random(num_features) * STD_SCALE

    def predict(self, X):
        return np.dot(X, self.mu_coeffs), np.dot(X, self.std_coeffs)


def get_trainable_model():
    return regressor.UncertaintyGBM(verbose=True)


def get_groundtruth_model():
    return SimpleLinear(NUM_FEATURES)


def get_data(num_samples, num_features, groundtruth):
    """Returns randomly sampled X, y."""
    X = np.random.random((num_samples, num_features))
    mu_y, std_y = groundtruth.predict(X)
    y = mu_y + std_y * np.random.normal(size=num_samples)
    return X, y


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    trainable_model = get_trainable_model()
    groundtruth_model = get_groundtruth_model()
    my_loss = loss.HeteroscedasticNormalLossFunction()

    train_X, train_y = get_data(NUM_TRAIN, NUM_FEATURES, groundtruth_model)
    test_X, test_y = get_data(NUM_TEST, NUM_FEATURES, groundtruth_model)
    trainable_model.fit(train_X, train_y)

    print 'train results'
    expected_mu, expected_std = groundtruth_model.predict(train_X)
    for i, pred in enumerate(trainable_model.staged_predict(train_X)):
        if i % 10 != 0:
            continue
        print 'stage %d: NLL = %.3f, R2 on mu(X) = %.3f, R2 on std(X) = %.3f' \
            % (i, my_loss(train_y, pred),
               r2_score(expected_mu, pred[:, 0]),
               r2_score(expected_std, pred[:, 1]))

    print 'test results'
    expected_mu, expected_std = groundtruth_model.predict(test_X)
    for i, pred in enumerate(trainable_model.staged_predict(test_X)):
        if i % 10 != 0:
            continue
        print 'stage %d: NLL = %.3f, R2 on mu(X) = %.3f, R2 on std(X) = %.3f' \
            % (i, my_loss(test_y, pred),
               r2_score(expected_mu, pred[:, 0]),
               r2_score(expected_std, pred[:, 1]))


if __name__ == '__main__':
    main()
