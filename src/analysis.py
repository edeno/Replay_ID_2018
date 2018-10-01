
from logging import getLogger

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage.measurements import label
from scipy.stats import linregress

from loren_frank_data_processing import (get_all_multiunit_indicators,
                                         get_all_spike_indicators,
                                         get_interpolated_position_dataframe,
                                         get_LFPs, make_neuron_dataframe,
                                         make_tetrode_dataframe)
from replay_classification import SortedSpikeDecoder
from replay_identification import ReplayDetector
from ripple_detection import Kay_ripple_detector
from spectral_connectivity import Connectivity, Multitaper

_MARKS = ['channel_1_max', 'channel_2_max', 'channel_3_max',
          'channel_4_max']

logger = getLogger(__name__)


def detect_epoch_ripples(epoch_key, animals, sampling_frequency,
                         position_info=None,
                         minimum_duration=np.timedelta64(15, 'ms'),
                         zscore_threshold=2,
                         close_ripple_threshold=np.timedelta64(0, 'ms')):
    '''Returns a list of tuples containing the start and end times of
    ripples. Candidate ripples are computed via the ripple detection
    function and then filtered to exclude ripples where the animal was
    still moving.
    '''

    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)
    tetrode_keys = tetrode_info[(tetrode_info.validripple == 1)].index
    lfps = get_LFPs(tetrode_keys, animals)
    time = lfps.index
    if position_info is None:
        position_info = get_interpolated_position_dataframe(
            epoch_key, animals)
    speed = position_info.linear_speed

    return Kay_ripple_detector(
        time, lfps.values, speed.values, sampling_frequency,
        minimum_duration=minimum_duration, zscore_threshold=zscore_threshold,
        close_ripple_threshold=close_ripple_threshold)


def get_ripple_indicator(epoch_key, animals, ripple_times, speed):
    time = speed.index
    ripple_indicator = pd.Series(np.zeros_like(time, dtype=bool), index=time)
    for _, start_time, end_time in ripple_times.itertuples():
        if speed.loc[start_time:end_time].max() <= 4:
            ripple_indicator.loc[start_time:end_time] = True

    return ripple_indicator


def estimate_ripple_band_power(lfps, sampling_frequency):
    m = Multitaper(lfps.values, sampling_frequency=sampling_frequency,
                   time_halfbandwidth_product=1,
                   time_window_duration=0.020,
                   time_window_step=0.020,
                   start_time=lfps.index[0].total_seconds())
    c = Connectivity.from_multitaper(m)
    closest_200Hz_freq_ind = np.argmin(np.abs(c.frequencies - 200))
    power = c.power()[..., closest_200Hz_freq_ind, :].squeeze()
    power = pd.DataFrame(power, index=pd.TimedeltaIndex(c.time, unit='s'))
    return power.reindex(lfps.index)


def load_data(epoch_key, animals, sampling_frequency, use_likelihoods):
    logger.info('Loading Data...')
    position_info = (
        get_interpolated_position_dataframe(epoch_key, animals)
        .assign(lagged_linear_distance=lambda df: df.linear_distance.shift(1))
        .dropna(subset=['linear_distance', 'linear_speed']))

    speed = position_info.linear_speed
    time = position_info.index

    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)
    tetrode_keys = tetrode_info[(tetrode_info.validripple == 1)].index
    lfps = get_LFPs(tetrode_keys, animals)
    lfps = lfps.reindex(time)

    if 'lfp_power' in use_likelihoods:
        power = estimate_ripple_band_power(lfps, sampling_frequency)
    else:
        power = None

    if 'spikes' in use_likelihoods:
        neuron_info = make_neuron_dataframe(animals).xs(
            epoch_key, drop_level=False)
        neuron_info = neuron_info[(neuron_info.numspikes > 0) &
                                  (neuron_info.type == 'principal')]
        spikes = (get_all_spike_indicators(neuron_info.index, animals)
                  .reindex(time)).values
    else:
        spikes = None

    if 'multiunit' in use_likelihoods:
        tetrode_keys = tetrode_info[(tetrode_info.numcells > 0)].index
        multiunit = (get_all_multiunit_indicators(tetrode_keys, animals)
                     .reindex(time)).values
    else:
        multiunit = None

    logger.info('Finding ripple times...')
    ripple_times = Kay_ripple_detector(
        time, lfps.values, speed.values, sampling_frequency,
        zscore_threshold=2, close_ripple_threshold=np.timedelta64(0, 'ms'),
        minimum_duration=np.timedelta64(15, 'ms'))
    is_ripple = get_ripple_indicator(
        epoch_key, animals, ripple_times, speed)

    return {
        'position_info': position_info,
        'is_ripple': is_ripple,
        'power': power,
        'spikes': spikes,
        'multiunit': multiunit,
    }


def identify_replays(data, use_likelihoods):
    detector = ReplayDetector()
    detector.fit(
        data['is_ripple'], data['position_info'].speed.values,
        data['position_info'].linear_distance.values, data['power'],
        data['spikes'], data['multiunit'])
    results = detector.predict(
        data['position_info'].speed.values,
        data['position_info'].linear_distance.values, data['power'],
        data['spikes'], data['multiunit'], time=data['position_info'].index,
        use_likelihoods=use_likelihoods)

    return results, detector


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
        density = cur_detector_results.replay_posterior

        detector_posterior.append(density)
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

    detector_posterior = (xr.concat(detector_posterior, dim=replay_info.index)
                          .rename('detector_posterior'))
    detector_likelihood = (xr.concat(detector_likelihood, dim=replay_info.index)
                           .rename('detector_likelihood'))
    decoder_posterior = (xr.concat(decoder_posterior, dim=replay_info.index)
                         .rename('decoder_posterior'))

    MOTION_BOUNDS = [-np.inf, -50, 50, np.inf]

    replay_info['replay_type'] = replay_type
    replay_info['replay_type_confidence'] = replay_type_confidence
    replay_info['replay_motion_slope'] = motion_slope
    replay_info['replay_motion_type'] = pd.cut(
        replay_info['replay_motion_slope'], MOTION_BOUNDS,
        labels=['Away', 'Neither', 'Towards'])
    replay_info['replay_movement_distance'] = replay_movement_distance
    replay_info['credible_interval_size'] = credible_interval_size

    return replay_info, detector_posterior, decoder_posterior, detector_likelihood


def _get_test_spikes(data, labels, replay_number, sampling_frequency):
    test_spikes = data['spikes'][labels.replay_number == replay_number].T
    n_time = test_spikes.shape[1]
    time = pd.TimeDeltaIndex(np.arange(0, n_time) / sampling_frequency,
                             unit='s')
    return test_spikes, time


def decode_replays(data, detector, labels, replay_info, sampling_frequency):
    is_training = ((data['position_info'].linear_speed > 4) &
                   data['position_info'].is_correct)
    train_position_info = data['position_info'].loc[is_training]
    train_spikes = data['spikes'][is_training]

    decoder = SortedSpikeDecoder(
        position=train_position_info.linear_distance.values,
        lagged_position=train_position_info.lagged_linear_distance.values,
        trajectory_direction=train_position_info.task.values,
        spikes=train_spikes.T,
        replay_speedup_factor=detector.replay_speed,
        n_position_bins=detector.place_bin_centers.size,
        confidence_threshold=0.8,
        knot_spacing=detector.spike_model_knot_spacing,
    ).fit()

    decoder_results = [
        decoder.predict(*_get_test_spikes(data, labels, replay_number,
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
        Negative is Away, Positive is Towards

    '''
    posterior_density = posterior_density.dropna('time')
    replay_position = posterior_density.position.values[
        np.log(posterior_density).argmax('position').values]
    replay_distance_from_animal_position = np.abs(
        replay_position - replay_info.linear_distance)
    slope = linregress(
        posterior_density.get_index('time').values,
        replay_distance_from_animal_position).slope
    return slope


def _get_replay_movement(posterior_density):
    replay_position = posterior_density.position.values[
        np.log(posterior_density).argmax('position').values]
    return np.diff(np.quantile(replay_position, [0.25, 0.75]))[0]


def _average_credible_interval_size(posterior_density):
    credible_interval = np.percentile(posterior_density, [2.5, 97.5], axis=1)
    return np.mean(np.diff(credible_interval, axis=0))


def generate_samples_from_posterior(posterior, n_values=100):
    '''
    https://stackoverflow.com/questions/17821458/random-number-from-histogram

    NOTE: could also use np.random.choice but posterior doesn't sum to 1 right now...
    '''
    value_bins = []
    for hist in posterior:
        cdf = np.cumsum(hist)
        cdf /= cdf[-1]
        value_bins.append(np.searchsorted(cdf, np.random.rand(n_values)))
    return np.array(value_bins)
