import os
import re
from shutil import copyfile

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


def copy_marks(src_directory, animal):
    target_directory = animal.directory
    mark_files = [os.path.join(root, f)
                  for root, _, files in os.walk(src_directory)
                  for f in files if (f.endswith('_params.mat') or
                                     f.endswith('_params_pc.mat'))
                  and not f.startswith('matclust')]
    new_mark_filenames = [rename_mark_file(mark_file, animal)
                          for mark_file in mark_files]

    for mark_file, new_filename in zip(mark_files, new_mark_filenames):
        mark_path = os.path.join(target_directory, 'EEG', new_filename)
        print(f'Copying {mark_file}\nto \n{new_filename}\n')
        copyfile(mark_file, mark_path)


def rename_mark_file(filename, animal):
    filename = os.path.splitext(os.path.basename(filename))[0]
    day, tetrode = [int(d) for d in re.findall(r'\d+', filename)]
    return f'{animal.short_name}marks{day:02d}-{tetrode:02d}.mat'
