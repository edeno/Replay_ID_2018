import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import TimeSeriesSplit

from replay_classification.core import filter, smooth
from replay_identification import ReplayDetector
from replay_identification.multiunit_likelihood import \
    estimate_replay_log_likelihood

_DEFAULT_MODEL_KWARGS = dict(n_components=30, tol=1E-6, max_iter=200,
                             weight_concentration_prior=1E1)


def predict_position_from_spikes(replay_detector, spikes, use_smoother):
    pass


def predict_position_from_multiunit(replay_detector, multiunits, use_smoother):
    multiunits = np.asarray(multiunits.copy())
    likelihood_model = replay_detector._multiunit_likelihood.keywords
    place_bin_centers = replay_detector.place_bin_centers
    occupancy_model = likelihood_model['occupancy_model']
    joint_models = likelihood_model['joint_models']
    marginal_models = likelihood_model['marginal_models']
    mean_rates = likelihood_model['mean_rates']
    time_bin_size = replay_detector.time_bin_size

    likelihood = np.exp(estimate_replay_log_likelihood(
        np.moveaxis(multiunits, -1, 0), place_bin_centers,
        occupancy_model, joint_models, marginal_models, mean_rates,
        time_bin_size))
    place_bin_size = np.diff(replay_detector.place_bin_edges)[0]

    state_transition = replay_detector._movement_state_transition
    initial_conditions = (np.ones_like(place_bin_centers) /
                          place_bin_centers.size / place_bin_size)
    results = filter(
        initial_conditions=initial_conditions,
        state_transition=state_transition,
        likelihood=likelihood,
        bin_size=place_bin_size)

    if use_smoother:
        results = smooth(
            filter_posterior=results['posterior_density'][..., np.newaxis, :],
            state_transition=state_transition,
            bin_size=place_bin_size)
        results['likelihood'] = likelihood

    max_ind = np.argmax(np.log(results['posterior_density'].squeeze()), axis=1)
    return place_bin_centers[max_ind]


def cross_validate_position_decoding(
        data, multiunit_density_model=BayesianGaussianMixture,
        multiunit_model_kwargs=_DEFAULT_MODEL_KWARGS,
        position_metric='linear_distance', speed_metric='linear_speed',
        use_smoother=True, data_types={'multiunit'},
        n_splits=5):

    cv = TimeSeriesSplit(n_splits=n_splits)
    time = data['position_info'].index
    rmse = []

    replay_detector = ReplayDetector(
        multiunit_density_model=multiunit_density_model,
        multiunit_model_kwargs=multiunit_model_kwargs, replay_speed=1)

    for train_index, test_index in cv.split(time):
        train_is_replay = data['is_ripple'].iloc[train_index]
        train_position_info = data['position_info'].iloc[train_index]
        if 'spikes' in data_types:
            train_spikes = data['spikes'].iloc[train_index]
        else:
            train_spikes = None
        if 'multiunit' in data_types:
            train_multiunit = data['multiunit'].isel(time=train_index)
        else:
            train_multiunit = None
        replay_detector.fit(
            is_replay=train_is_replay, speed=train_position_info[speed_metric],
            position=train_position_info[position_metric], lfp_power=None,
            spikes=train_spikes, multiunit=train_multiunit)

        if 'multiunit' in data_types:
            test_multiunit = data['multiunit'].isel(time=test_index)
            predicted_position = predict_position_from_multiunit(
                replay_detector, test_multiunit, use_smoother)
        elif 'spikes' in data_types:
            test_spikes = data['spikes'].iloc(time=test_index)
            predicted_position = predict_position_from_spikes(
                replay_detector, test_spikes, use_smoother)

        true_position = (data['position_info'][position_metric]
                         .iloc[test_index].values)
        rmse.append(np.sqrt(mean_squared_error(
            true_position, predicted_position)))

    return pd.DataFrame(rmse, columns=['root_mean_squared_error'],
                        index=pd.Index(np.arange(n_splits) + 1, name='fold'))
