from logging import getLogger
from os.path import join

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from loren_frank_data_processing import (get_all_multiunit_indicators,
                                         get_all_spike_indicators,
                                         get_interpolated_position_dataframe,
                                         get_LFPs, get_trial_time,
                                         make_neuron_dataframe,
                                         make_tetrode_dataframe)
from ripple_detection import (Kay_ripple_detector, filter_ripple_band,
                              get_multiunit_population_firing_rate,
                              multiunit_HSE_detector)
from spectral_connectivity import Connectivity, Multitaper
from src.parameters import (ANIMALS, BRAIN_AREAS, PROCESSED_DATA_DIR,
                            SAMPLING_FREQUENCY)

logger = getLogger(__name__)

_MARKS = ['channel_1_max', 'channel_2_max', 'channel_3_max', 'channel_4_max']


def get_labels(times, time):
    ripple_labels = pd.DataFrame(np.zeros_like(time, dtype=np.int), index=time,
                                 columns=['replay_number'])
    for replay_number, start_time, end_time in times.itertuples():
        ripple_labels.loc[start_time:end_time] = replay_number

    return ripple_labels


def estimate_ripple_band_power(lfps, sampling_frequency):
    m = Multitaper(lfps.values, sampling_frequency=sampling_frequency,
                   time_halfbandwidth_product=1,
                   time_window_duration=0.020,
                   time_window_step=0.020,
                   start_time=lfps.index[0].total_seconds())
    c = Connectivity.from_multitaper(m)
    closest_200Hz_freq_ind = np.argmin(np.abs(c.frequencies - 200))
    power = c.power()[..., closest_200Hz_freq_ind, :].squeeze() + np.spacing(1)
    n_samples = int(0.020 * sampling_frequency)
    index = lfps.index[np.arange(1, power.shape[0] * n_samples + 1, n_samples)]
    power = pd.DataFrame(power, index=index)
    return power.reindex(lfps.index)


def get_adhoc_ripple(time, speed, tetrode_info):
    is_brain_areas = (
        tetrode_info.area.astype(str).str.upper().isin(BRAIN_AREAS))
    tetrode_keys = tetrode_info.loc[
        is_brain_areas & (tetrode_info.validripple == 1)].index

    ripple_lfps = get_LFPs(tetrode_keys, ANIMALS).reindex(time)
    ripple_lfps = (
        ripple_lfps.resample('1ms').mean().fillna(method='pad').reindex(time))
    ripple_times = Kay_ripple_detector(
        time, ripple_lfps.values, speed.values, SAMPLING_FREQUENCY,
        zscore_threshold=2.0, close_ripple_threshold=np.timedelta64(0, 'ms'),
        minimum_duration=np.timedelta64(15, 'ms'))

    ripple_times.index = ripple_times.index.rename('replay_number')
    ripple_labels = get_labels(ripple_times, time)
    ripple_times = ripple_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds())

    ripple_filtered_lfps = pd.DataFrame(
        np.stack([filter_ripple_band(ripple_lfps.values[:, ind])
                  for ind in np.arange(ripple_lfps.shape[1])], axis=1),
        index=ripple_lfps.index)

    ripple_power = estimate_ripple_band_power(ripple_lfps, SAMPLING_FREQUENCY)

    return (ripple_times, ripple_labels, ripple_filtered_lfps, ripple_power,
            ripple_lfps)


def get_adhoc_multiunit(speed, tetrode_info, time_function):
    time = time_function()
    is_brain_areas = (
        tetrode_info.area.astype(str).str.upper().isin(BRAIN_AREAS))
    tetrode_info = tetrode_info.loc[is_brain_areas]

    multiunit = (get_all_multiunit_indicators(
        tetrode_info.index, ANIMALS, time_function)
        .sel(features=_MARKS)
        .reindex({'time': time}))
    multiunit_spikes = (np.any(~np.isnan(multiunit.values), axis=1)
                        ).astype(np.float)
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(
            multiunit_spikes, SAMPLING_FREQUENCY), index=time,
        columns=['firing_rate'])

    multiunit_high_synchrony_times = multiunit_HSE_detector(
        time, multiunit_spikes, speed.values, SAMPLING_FREQUENCY,
        minimum_duration=np.timedelta64(15, 'ms'), zscore_threshold=2.0,
        close_event_threshold=np.timedelta64(0, 'ms'))
    multiunit_high_synchrony_times.index = (
        multiunit_high_synchrony_times.index.rename('replay_number'))
    multiunit_high_synchrony_labels = get_labels(
        multiunit_high_synchrony_times, time)
    multiunit_high_synchrony_times = multiunit_high_synchrony_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds())

    return (multiunit, multiunit_spikes, multiunit_firing_rate,
            multiunit_high_synchrony_times, multiunit_high_synchrony_labels)


def get_spikes(neuron_info, time_function):
    time = time_function()
    neuron_keys = neuron_info.loc[
        (neuron_info.numspikes > 100) &
        neuron_info.area.isin(BRAIN_AREAS) &
        (neuron_info.type == 'principal')].index
    spikes = get_all_spike_indicators(
        neuron_keys, ANIMALS, time_function).reindex(time)

    return spikes


def load_data(epoch_key):
    logger.info('Loading data...')
    time = get_trial_time(epoch_key, ANIMALS)
    time = (pd.Series(np.ones_like(time, dtype=np.float), index=time)
            .resample('1ms').mean()
            .index)

    def _time_function(*args, **kwargs):
        return time

    position_info = (
        get_interpolated_position_dataframe(
            epoch_key, ANIMALS, _time_function, spacing=30)
        .dropna(subset=['linear_distance', 'linear_speed']))

    time = position_info.index
    speed = position_info['speed']

    neuron_info = make_neuron_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False)

    spikes = get_spikes(neuron_info, _time_function)
    tetrode_info = make_tetrode_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False)

    logger.info('Finding multiunit high synchrony events...')
    (multiunit, multiunit_spikes, multiunit_firing_rate,
     multiunit_high_synchrony_times,
     multiunit_high_synchrony_labels) = get_adhoc_multiunit(
        speed, tetrode_info, _time_function)

    logger.info('Finding ripple times...')
    (ripple_times, ripple_labels, ripple_filtered_lfps,
     ripple_power, ripple_lfps) = get_adhoc_ripple(time, speed, tetrode_info)

    return {
        'position_info': position_info,
        'tetrode_info': tetrode_info,
        'neuron_info': neuron_info,
        'spikes': spikes,
        'ripple_times': ripple_times,
        'is_ripple': ripple_labels > 0,
        'ripple_labels': ripple_labels,
        'lfps': ripple_lfps,
        'ripple_filtered_lfps': ripple_filtered_lfps,
        'power': ripple_power,
        'multiunit': multiunit,
        'multiunit_high_synchrony_times': multiunit_high_synchrony_times,
        'is_multiunit_high_synchrony': multiunit_high_synchrony_labels > 0,
        'multiunit_high_synchrony_labels': multiunit_high_synchrony_labels,
        'multiunit_firing_rate': multiunit_firing_rate,
    }


def load_detector(epoch_key):
    animal, day, epoch = epoch_key
    file_name = f'{animal}_{day:02}_{epoch:02}_replay_detector.gz'
    return joblib.load(join(PROCESSED_DATA_DIR, file_name))
