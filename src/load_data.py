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
from ripple_detection import Kay_ripple_detector
from spectral_connectivity import Connectivity, Multitaper
from src.parameters import PROCESSED_DATA_DIR

logger = getLogger(__name__)

_MARKS = ['channel_1_max', 'channel_2_max', 'channel_3_max', 'channel_4_max']


def get_ripple_labels(ripple_times, time):
    ripple_labels = pd.Series(np.zeros_like(time, dtype=np.int), index=time,
                              name='ripple_number')
    for ripple_number, start_time, end_time in ripple_times.itertuples():
        ripple_labels.loc[start_time:end_time] = ripple_number

    return ripple_labels


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


def load_data(epoch_key, animals, sampling_frequency, data_types=None):
    if data_types is None:
        data_types = ['spikes', 'lfp_power']

    logger.info('Loading Data...')
    position_info = (
        get_interpolated_position_dataframe(epoch_key, animals)
        .dropna(subset=['linear_distance', 'linear_speed']))

    speed = position_info.linear_speed
    time = position_info.index

    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)
    tetrode_keys = tetrode_info[(tetrode_info.validripple == 1)].index
    lfps = get_LFPs(tetrode_keys, animals)
    lfps = lfps.reindex(time)

    if 'lfp_power' in data_types:
        power = estimate_ripple_band_power(lfps, sampling_frequency)
    else:
        power = None

    if 'spikes' in data_types:
        neuron_info = make_neuron_dataframe(animals).xs(
            epoch_key, drop_level=False)
        neuron_info = neuron_info[(neuron_info.numspikes > 0) &
                                  (neuron_info.type == 'principal')]
        spikes = (get_all_spike_indicators(neuron_info.index, animals)
                  .reindex(time)).values
    else:
        spikes = None

    if 'multiunit' in data_types:
        tetrode_keys = tetrode_info[(tetrode_info.numcells > 0)].index
        multiunit = (get_all_multiunit_indicators(tetrode_keys, animals)
                     .sel(features=_MARKS)
                     .reindex({'time': time})
                     .values)
    else:
        multiunit = None

    logger.info('Finding ripple times...')
    ripple_times = Kay_ripple_detector(
        time, lfps.values, speed.values, sampling_frequency,
        zscore_threshold=2, close_ripple_threshold=np.timedelta64(0, 'ms'),
        minimum_duration=np.timedelta64(15, 'ms'))
    ripple_labels = get_ripple_labels(ripple_times, time)

    return {
        'position_info': position_info,
        'ripple_times': ripple_times,
        'is_ripple': ripple_labels > 0,
        'ripple_labels': ripple_labels,
        'power': power,
        'spikes': spikes,
        'multiunit': multiunit,
    }


def load_detector(epoch_key):
    animal, day, epoch = epoch_key
    file_name = f'{animal}_{day:02}_{epoch:02}_replay_detector.gz'
    return joblib.load(join(PROCESSED_DATA_DIR, file_name))
