
from logging import getLogger

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage.measurements import label
from scipy.stats import linregress

from replay_classification import SortedSpikeDecoder

logger = getLogger(__name__)


def start_time(s):
    return s.index[0]


def end_time(s):
    return s.index[-1]


def max_probability(df):
    return df.max()


def duration(df):
    return df.index[-1] - df.index[0]


def get_replay_times(results, probability_threshold=0.8,
                     minimum_duration=np.timedelta64(0, 'ms')):
    MINIMUM = 0.01
    labels = pd.DataFrame(
        {'replay_number': label(results.replay_probability > MINIMUM)[0],
         'replay_probability': results.replay_probability.values},
        index=results.time.to_index())

    grouper = labels.loc[labels.replay_number > 0].groupby('replay_number')
    replay_times = (grouper
                    .agg([start_time, end_time, duration, max_probability]))
    replay_times.columns = replay_times.columns.get_level_values(1)

    replay_times = (replay_times.loc[
        (replay_times.duration > minimum_duration) &
        (replay_times.max_probability > probability_threshold)])

    labels.loc[~labels.replay_number.isin(replay_times.index)] = 0
    labels.replay_number = label(labels.replay_number)[0]
    new_index = pd.Index(labels.replay_number.unique()[1:],
                         name='replay_number')
    replay_times = replay_times.set_index(new_index)

    return replay_times, labels


def summarize_replays(replay_info, detector_results, decoder_results, data):
    MOTION_BOUNDS = [-np.inf, -50, 50, np.inf]

    replay_position_info = (data['position_info']
                            .loc[replay_info.start_time]
                            .set_index(replay_info.index))
    replay_info = pd.concat((replay_info, replay_position_info), axis=1)

    detector_posterior = []
    decoder_posterior = []
    detector_likelihood = []
    replay_type = []
    replay_type_confidence = []
    motion_slope = []
    replay_movement_distance = []
    credible_interval_size = []

    for r, dr in zip(replay_info.itertuples(), decoder_results):
        # Get detector posterior
        cur_detector_results = (
            detector_results.sel(time=slice(r.start_time, r.end_time))
            .assign_coords(time=lambda da: da.time - r.start_time))
        density = dr.results.posterior_density.sum('state') / 4

        detector_posterior.append(cur_detector_results.replay_posterior)
        detector_likelihood.append(cur_detector_results.likelihood)

        # Get decoder posterior
        decoder_posterior.append(dr.results.posterior_density)

        # Classify Replay
        replay_type.append(dr.predicted_state())
        replay_type_confidence.append(dr.predicted_state_probability())

        # Replay Motion
        motion_slope.append(_get_replay_motion(r, density))

        # How far replay moves
        replay_movement_distance.append(_get_replay_movement(density))

        # How confident are we?
        credible_interval_size.append(_average_credible_interval_size(density))

    replay_info['replay_type'] = replay_type
    replay_info['replay_type_confidence'] = replay_type_confidence
    replay_info['replay_motion_slope'] = motion_slope
    replay_info['replay_motion_type'] = pd.cut(
        replay_info['replay_motion_slope'], MOTION_BOUNDS,
        labels=['Towards', 'Neither', 'Away'])
    replay_info['replay_movement_distance'] = replay_movement_distance
    replay_info['credible_interval_size'] = credible_interval_size

    detector_posterior = (xr.concat(detector_posterior, dim=replay_info.index)
                          .rename('detector_posterior'))
    detector_likelihood = (xr.concat(
        detector_likelihood, dim=replay_info.index)
                           .rename('detector_likelihood'))
    decoder_posterior = (xr.concat(decoder_posterior, dim=replay_info.index)
                         .rename('decoder_posterior'))
    replay_densities = xr.merge(
        (detector_posterior, detector_likelihood, decoder_posterior))

    return replay_info, replay_densities


def _get_test_spikes(data, is_replay, replay_number, sampling_frequency):
    test_spikes = data['spikes'][is_replay.replay_number == replay_number]
    n_time = test_spikes.shape[0]
    time = pd.TimedeltaIndex(np.arange(0, n_time) / sampling_frequency,
                             unit='s')
    return test_spikes, time


def decode_replays(data, replay_detector, is_replay, replay_info,
                   sampling_frequency):

    decoder = SortedSpikeDecoder(
        replay_speedup_factor=replay_detector.replay_speed,
        knot_spacing=replay_detector.spike_model_knot_spacing,
        spike_model_penalty=replay_detector.spike_model_penalty
    ).fit(
        position=data['position_info'].linear_distance.values,
        trajectory_direction=data['position_info'].task.values,
        spikes=data['spikes'],
        is_training=(is_replay.replay_number == 0))

    decoder_results = [
        decoder.predict(*_get_test_spikes(data, is_replay, replay_number,
                                          sampling_frequency))
        for replay_number in replay_info.index]

    return decoder_results, decoder


def _get_replay_motion(replay_info, posterior_density):
    '''

    Parameters
    ----------
    ripple_info : pandas dataframe row
    posterior_density : array, shape (n_time, n_position_bins)

    Returns
    -------
    slope : float
        Negative is Towards, Positive is Away

    '''
    posterior_density = posterior_density.dropna('time')
    replay_position = posterior_density.position.values[
        np.log(posterior_density).argmax('position').values]
    replay_distance_from_animal_position = np.abs(
        replay_position - replay_info.linear_distance)
    slope = linregress(
        posterior_density.get_index('time').total_seconds().values,
        replay_distance_from_animal_position).slope
    return slope


def _get_replay_movement(posterior_density):
    replay_position = posterior_density.position.values[
        np.log(posterior_density).argmax('position').values]
    return np.diff(np.quantile(replay_position, [0.25, 0.75]))[0]


def _average_credible_interval_size(posterior_density):
    credible_interval = np.percentile(posterior_density, [2.5, 97.5], axis=1)
    return np.mean(np.diff(credible_interval, axis=0))
