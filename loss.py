__doc__ = """Implements loss function for regression on heteroscedastic data."""

import numpy as np
import six
from abc import ABCMeta
from scipy.stats import norm
from sklearn.ensemble import gradient_boosting


class NormalEstimator(gradient_boosting.BaseEstimator):
    """Estimator for mean, std assuming normal distribution."""
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            self.mean = np.mean(y)
            self.std = np.std(y)
        else:
            self.mean = np.average(y, weights=sample_weight)
            self.std = np.average((y - self.mean) ** 2,
                                  weights=sample_weight) ** 0.5

    def predict(self, X):
        gradient_boosting.check_is_fitted(self, 'mean')
        gradient_boosting.check_is_fitted(self, 'std')

        y = np.empty((X.shape[0], 2), dtype=np.float64)
        y[:, 0] = self.mean
        y[:, 1] = self.std
        return y


class HeteroscedasticNormalLossFunction(
        six.with_metaclass(ABCMeta, gradient_boosting.LossFunction)):
    """A loss function corresponding to the likelihood of an observed y(x)
    given a predicted normal distribution described by mu_y(x) and std_y(x).

    """
    def __init__(self, n_classes=2):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes.".format(
                self.__class__.__name__))

        super(HeteroscedasticNormalLossFunction, self).__init__(n_classes)

    def init_estimator(self):
        return NormalEstimator()

    def __call__(self, y, pred, sample_weight=None):
        """Returns negative average log-likelihood."""
        if np.any(pred[:, 1] <= 0):
            print 'WARNING: non-positive predicted std on %d data points' \
                % np.sum(pred[:, 1] <= 0)
        nll = -np.log(norm(pred[:, 0], pred[:, 1]).pdf(y))
        if sample_weight is None:
            return np.mean(nll)
        else:
            return np.average(nll, weights=sample_weight)

    def negative_gradient(self, y, pred, k=0, **kwargs):
        """Compute negative gradient for the mean, std."""
        # our loss function is C + log(std) + (y - mu) ** 2 / (2 * std ** 2)
        # --> gradient with respect to mu is -(y - mu) / std ** 2
        # --> gradient with respect to std is 1 / std - (y - mu) ** 2 / std ** 3
        if k == 0:
            return (y - pred[:, 0]) / pred[:, 1] ** 2
        else:
            return - 1 / pred[:, 1] + (y - pred[:, 0]) ** 2 / pred[:, 1] ** 3

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_weight, sample_mask,
                                learning_rate=1.0, k=0):
        # compute leaf for each sample in ``X``.
        terminal_regions = tree.apply(X)

        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        # update each leaf (= perform line search)
        for leaf in np.where(tree.children_left == gradient_boosting.TREE_LEAF)[0]:
            if k == 0:
                self._update_terminal_region_for_mean(
                    tree, masked_terminal_regions,
                    leaf, X, y, residual,
                    y_pred, sample_weight)
            else:
                self._update_terminal_region_for_std(
                    tree, masked_terminal_regions,
                    leaf, X, y, residual,
                    y_pred, sample_weight)

        # update predictions (both in-bag and out-of-bag)
        y_pred[:, k] += (learning_rate
                         * tree.value[:, 0, 0].take(terminal_regions, axis=0))

    def _update_terminal_region_for_mean(self, tree, terminal_regions, leaf, X, y,
                                         residual, pred, sample_weight):
        """Make a single Newton-Raphson step.

        Estimate is
            sum(w * (y - mu) / std ** 2) / sum(w * 1 / std ** 2)

        """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        pred = pred.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        numerator = np.sum(sample_weight * residual)
        denominator = np.sum(sample_weight * (1 / pred[:, 1] ** 2))

        if denominator == 0.0:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _update_terminal_region_for_std(self, tree, terminal_regions, leaf, X, y,
                                        residual, pred, sample_weight):
        """Make a single Newton-Raphson step.

        Estimate is
            sum(w * (- 1 / std + (y - mu) ** 2 / std ** 3))
                / sum(w * (-2 / std ** 3 + 3 * (y - mu) ** 2 / std ** 4))

        """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        pred = pred.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        numerator = np.sum(sample_weight * residual)
        denominator = np.sum(np.abs(sample_weight *
            (-2 / pred[:, 1] ** 3 + 3 * (y - pred[:, 0]) ** 2 / pred[:, 1] ** 4)))

        if denominator == 0.0:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        pass


gradient_boosting.LOSS_FUNCTIONS['heteroscedastic_normal'] = \
    HeteroscedasticNormalLossFunction
