import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

from src.load_data import load_data
from src.parameters import (ANIMALS, BRAIN_AREAS, SAMPLING_FREQUENCY,
                            USE_LIKELIHOODS)
from src.validate_position_decoding import cross_validate_position_decoding


def main():
    epoch_key = ('bon', 3, 2)
    speed_metric = 'linear_speed'
    position_metric = 'linear_distance'

    data_types = set(itertools.chain(*USE_LIKELIHOODS.values()))
    data = load_data(epoch_key, ANIMALS, SAMPLING_FREQUENCY, data_types,
                     BRAIN_AREAS, speed_metric)
    n_components = np.linspace(15, 60, 4, dtype=int)
    weight_concentration_prior = np.logspace(-2, 2, 5)

    grid_parameters = [
        dict(n_components=n, tol=1E-6, max_iter=200,
             weight_concentration_prior=w)
        for n, w in itertools.product(n_components, weight_concentration_prior)
    ]

    results = []
    for multiunit_model_kwargs in grid_parameters:
        results.append(cross_validate_position_decoding(
            data, multiunit_density_model=BayesianGaussianMixture,
            multiunit_model_kwargs=multiunit_model_kwargs,
            position_metric=position_metric, speed_metric=speed_metric,
            use_smoother=True, data_types={'multiunit'},
            n_splits=5))
    keys = list(itertools.product(n_components, weight_concentration_prior))
    results = pd.concat(results, axis=1, keys=keys,
                        names=['n_components', 'weight_concentration_prior'])
    results = (results
               .unstack()
               .reset_index()
               .rename(columns={0: 'root_mean_squared_error'})
               .drop(columns='level_2'))
    average_rmse = (results
                    .groupby(['n_components', 'weight_concentration_prior'])
                    .root_mean_squared_error
                    .agg(['mean', 'std']))
    print(average_rmse)


if __name__ == '__main__':
    sys.exit(main())
