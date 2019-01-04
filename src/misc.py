import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity


class WhitenedKDE(BaseEstimator, DensityMixin):
    def __init__(self, **kwargs):
        self.kde = KernelDensity(**kwargs)
        self.pre_whiten = PCA(whiten=True)

    def fit(self, X, y=None, sample_weight=None):
        self.kde.fit(self.pre_whiten.fit_transform(X))
        return self

    def score_samples(self, X):
        return self.kde.score_samples(self.pre_whiten.transform(X))


class StatsmodelsKDE(BaseEstimator, DensityMixin):
    def __init__(self, bandwidth='normal_reference'):
        self.bandwidth = bandwidth

    def fit(self, X, y=None):
        n_features = X.shape[1]
        if not isinstance(self.bandwidth, str):
            bandwidth = self.bandwidth[-n_features:]
        else:
            bandwidth = self.bandwidth

        self.estimator_ = sm.nonparametric.KDEMultivariate(
            data=X, var_type='c' * n_features, bw=bandwidth)

        return self

    def score_samples(self, X):
        return np.log(self.estimator_.pdf(X))
