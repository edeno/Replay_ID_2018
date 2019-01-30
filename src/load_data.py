from logging import getLogger
from os.path import join

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from loren_frank_data_processing import (get_all_multiunit_indicators,
                                         get_all_spike_indicators,
                                         get_interpolated_position_dataframe,
                                         get_LFPs, make_neuron_dataframe,
                                         make_tetrode_dataframe)
from ripple_detection import (Kay_ripple_detector, filter_ripple_band,
                              get_multiunit_population_firing_rate,
                              multiunit_HSE_detector)
from spectral_connectivity import Connectivity, Multitaper
from src.parameters import PROCESSED_DATA_DIR

logger = getLogger(__name__)

_MARKS = ['channel_1_max', 'channel_2_max', 'channel_3_max', 'channel_4_max']


def get_ripple_labels(ripple_times, time):
    ripple_labels = pd.DataFrame(np.zeros_like(time, dtype=np.int), index=time,
                                 columns=['replay_number'])
    for replay_number, start_time, end_time in ripple_times.itertuples():
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


def load_data(epoch_key, animals, sampling_frequency,
              brain_areas, speed_metric='linear_speed'):

    logger.info('Loading Data...')
    position_info = (
        get_interpolated_position_dataframe(epoch_key, animals)
        .dropna(subset=['linear_distance', 'linear_speed']))

    speed = position_info[speed_metric]
    time = position_info.index

    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)
    is_brain_areas = tetrode_info.area.isin(brain_areas)
    tetrode_keys = tetrode_info.loc[
        (tetrode_info.validripple == 1) & is_brain_areas].index
    lfps = get_LFPs(tetrode_keys, animals)
    lfps = lfps.reindex(time)

    power = estimate_ripple_band_power(lfps, sampling_frequency)

    neuron_info = make_neuron_dataframe(animals).xs(
        epoch_key, drop_level=False)
    neuron_info = neuron_info.loc[
        (neuron_info.numspikes > 0) &
        neuron_info.area.isin(brain_areas)]
    spikes = (get_all_spike_indicators(neuron_info.index, animals)
              .reindex(time))

    logger.info('Finding multiunit high synchrony events...')
    tetrode_info = tetrode_info.loc[
        (tetrode_info.numcells > 0) & is_brain_areas]
    multiunit = (get_all_multiunit_indicators(tetrode_info.index, animals)
                 .sel(features=_MARKS)
                 .reindex({'time': time}))
    multiunit_spikes = (np.any(~np.isnan(multiunit), axis=1)
                        .values).astype(np.float)
    multiunit_high_synchrony_times = multiunit_HSE_detector(
        time, multiunit_spikes, speed.values, sampling_frequency,
        minimum_duration=np.timedelta64(15, 'ms'), zscore_threshold=2.0,
        close_event_threshold=np.timedelta64(0, 'ms'))
    multiunit_high_synchrony_times.index = (
        multiunit_high_synchrony_times.index.rename('replay_number'))
    multiunit_high_synchrony_labels = get_ripple_labels(
        multiunit_high_synchrony_times, time)
    multiunit_high_synchrony_times = multiunit_high_synchrony_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds())
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(
            multiunit_spikes, sampling_frequency), index=time,
        columns=['firing_rate'])

    logger.info('Finding ripple times...')
    ripple_times = Kay_ripple_detector(
        time, lfps.values, speed.values, sampling_frequency,
        zscore_threshold=2.0, close_ripple_threshold=np.timedelta64(0, 'ms'),
        minimum_duration=np.timedelta64(15, 'ms'))
    ripple_times.index = ripple_times.index.rename('replay_number')
    ripple_labels = get_ripple_labels(ripple_times, time)
    ripple_times = ripple_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds())

    ripple_band_lfps = pd.DataFrame(
        np.stack([filter_ripple_band(lfps.values[:, ind])
                  for ind in np.arange(lfps.shape[1])], axis=1),
        index=lfps.index)

    return {
        'position_info': position_info,
        'ripple_times': ripple_times,
        'is_ripple': ripple_labels > 0,
        'ripple_labels': ripple_labels,
        'power': power,
        'spikes': spikes,
        'multiunit': multiunit,
        'lfps': lfps,
        'tetrode_info': tetrode_info,
        'ripple_band_lfps': ripple_band_lfps,
        'multiunit_high_synchrony_times': multiunit_high_synchrony_times,
        'is_multiunit_high_synchrony': multiunit_high_synchrony_labels > 0,
        'multiunit_high_synchrony_labels': multiunit_high_synchrony_labels,
        'multiunit_firing_rate': multiunit_firing_rate,
    }


def load_detector(epoch_key):
    animal, day, epoch = epoch_key
    file_name = f'{animal}_{day:02}_{epoch:02}_replay_detector.gz'
    return joblib.load(join(PROCESSED_DATA_DIR, file_name))
