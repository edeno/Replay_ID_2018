from logging import getLogger
from os.path import join

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from loren_frank_data_processing import (get_all_multiunit_indicators,
                                         get_all_spike_indicators,
                                         get_interpolated_position_dataframe,
                                         get_LFPs, get_trial_time,
                                         make_neuron_dataframe,
                                         make_tetrode_dataframe)
from loren_frank_data_processing.position import make_track_graph
from ripple_detection import (Kay_ripple_detector, filter_ripple_band,
                              get_multiunit_population_firing_rate,
                              multiunit_HSE_detector)
from spectral_connectivity import Connectivity, Multitaper
from src.parameters import (ANIMALS, BRAIN_AREAS, MULTITAPER_PARAMETERS,
                            PROCESSED_DATA_DIR, SAMPLING_FREQUENCY)

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


def estimate_theta_power(time, tetrode_info, multitaper_params=None):
    if multitaper_params is None:
        multitaper_params = MULTITAPER_PARAMETERS['4Hz']

    is_brain_areas = (
        tetrode_info.area.astype(str).str.upper().isin(BRAIN_AREAS))
    tetrode_keys = tetrode_info.loc[is_brain_areas].index

    lfps = get_LFPs(tetrode_keys, ANIMALS).reindex(time)
    lfps = lfps.resample('2ms').mean().fillna(method='pad').reindex(time)

    m = Multitaper(lfps.values, **multitaper_params,
                   start_time=lfps.index[0].total_seconds())
    c = Connectivity.from_multitaper(m)
    dimension_names = ['time', 'frequency', 'tetrode']
    power = c.power()
    data_vars = {
        'power': (dimension_names, power)}

    n_samples = int(
        multitaper_params['time_window_duration'] * SAMPLING_FREQUENCY)
    index = lfps.index[np.arange(1, power.shape[0] * n_samples + 1, n_samples)]

    coordinates = {
        'time': index,
        'frequency': c.frequencies + np.diff(c.frequencies)[0] / 2,
        'tetrode': lfps.columns,
    }

    power = (xr.Dataset(data_vars, coords=coordinates)
             .sel(frequency=slice(0, 125)))
    power = power.reindex(time=lfps.index).interpolate_na('time')

    theta_power = (
        power.sel(frequency=slice(4, 12)).mean('frequency')
        .to_dataframe().unstack(level=0).interpolate())

    theta_power_change = theta_power.transform(
        lambda df: df / df.mean())
    theta_power_zscore = np.log(theta_power).transform(
        lambda df: (df - df.mean()) / df.std())

    return dict(
        theta_power=theta_power,
        theta_power_change=theta_power_change,
        theta_power_zscore=theta_power_zscore,
    )


def estimate_gamma_power(time, tetrode_info, multitaper_params=None):
    if multitaper_params is None:
        multitaper_params = MULTITAPER_PARAMETERS['10Hz']
    is_brain_areas = (
        tetrode_info.area.astype(str).str.upper().isin(BRAIN_AREAS))
    tetrode_keys = tetrode_info.loc[is_brain_areas].index

    lfps = get_LFPs(tetrode_keys, ANIMALS).reindex(time)
    lfps = lfps.resample('2ms').mean().fillna(method='pad').reindex(time)

    m = Multitaper(lfps.values, **multitaper_params,
                   start_time=lfps.index[0].total_seconds())
    c = Connectivity.from_multitaper(m)
    dimension_names = ['time', 'frequency', 'tetrode']
    power = c.power()
    data_vars = {
        'power': (dimension_names, power)}

    n_samples = int(
        multitaper_params['time_window_duration'] * SAMPLING_FREQUENCY)
    index = lfps.index[np.arange(1, power.shape[0] * n_samples + 1, n_samples)]

    coordinates = {
        'time': index,
        'frequency': c.frequencies + np.diff(c.frequencies)[0] / 2,
        'tetrode': lfps.columns,
    }

    power = (xr.Dataset(data_vars, coords=coordinates)
             .sel(frequency=slice(0, 125)))
    power = power.reindex(time=lfps.index).interpolate_na('time')

    low_gamma_power = (
        power.sel(frequency=slice(20, 50)).mean('frequency')
        .to_dataframe().unstack(level=0).interpolate())
    low_gamma_power_change = low_gamma_power.transform(
        lambda df: df / df.mean())
    low_gamma_power_zscore = np.log(low_gamma_power).transform(
        lambda df: (df - df.mean()) / df.std())

    high_gamma_power = (
        power.sel(frequency=slice(50, 125)).mean('frequency')
        .to_dataframe().unstack(level=0).interpolate())
    high_gamma_power_change = high_gamma_power.transform(
        lambda df: df / df.mean())
    high_gamma_power_zscore = np.log(high_gamma_power).transform(
        lambda df: (df - df.mean()) / df.std())

    return dict(low_gamma_power=low_gamma_power,
                low_gamma_power_change=low_gamma_power_change,
                low_gamma_power_zscore=low_gamma_power_zscore,
                high_gamma_power=high_gamma_power,
                high_gamma_power_change=high_gamma_power_change,
                high_gamma_power_zscore=high_gamma_power_zscore,
                lfps=lfps,
                )


def get_adhoc_ripple(epoch_key, tetrode_info, position_time):
    LFP_SAMPLING_FREQUENCY = 1500
    position_info = (
        get_interpolated_position_dataframe(epoch_key, ANIMALS)
        .dropna(subset=['linear_position', 'speed']))
    speed = position_info['speed']
    time = position_info.index

    if ~np.all(np.isnan(tetrode_info.validripple.astype(float))):
        tetrode_keys = tetrode_info.loc[
            (tetrode_info.validripple == 1)].index
    else:
        is_brain_areas = (
            tetrode_info.area.astype(str).str.upper().isin(BRAIN_AREAS))
        tetrode_keys = tetrode_info.loc[is_brain_areas].index

    ripple_lfps = get_LFPs(tetrode_keys, ANIMALS).reindex(time)
    ripple_filtered_lfps = pd.DataFrame(
        np.stack([filter_ripple_band(
            ripple_lfps.values[:, ind], sampling_frequency=1500)
            for ind in np.arange(ripple_lfps.shape[1])], axis=1),
        index=ripple_lfps.index)
    ripple_times = Kay_ripple_detector(
        time, ripple_lfps.values, speed.values, LFP_SAMPLING_FREQUENCY,
        zscore_threshold=2.0, close_ripple_threshold=np.timedelta64(0, 'ms'),
        minimum_duration=np.timedelta64(15, 'ms'))

    ripple_times.index = ripple_times.index.rename('replay_number')
    ripple_labels = get_labels(ripple_times, position_time)
    is_ripple = ripple_labels > 0
    ripple_times = ripple_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds())

    ripple_power = estimate_ripple_band_power(
        ripple_lfps, LFP_SAMPLING_FREQUENCY)
    interpolated_ripple_power = ripple_power.interpolate()

    ripple_power_change = interpolated_ripple_power.transform(
        lambda df: df / df.mean())
    ripple_power_zscore = np.log(interpolated_ripple_power).transform(
        lambda df: (df - df.mean()) / df.std())

    return dict(ripple_times=ripple_times,
                ripple_labels=ripple_labels,
                ripple_filtered_lfps=ripple_filtered_lfps,
                ripple_power=ripple_power,
                ripple_lfps=ripple_lfps,
                ripple_power_change=ripple_power_change,
                ripple_power_zscore=ripple_power_zscore,
                is_ripple=is_ripple)


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
    multiunit_rate_change = multiunit_firing_rate.transform(
        lambda df: df / df.mean())
    multiunit_rate_zscore = np.log(multiunit_firing_rate).transform(
        lambda df: (df - df.mean()) / df.std())

    multiunit_high_synchrony_times = multiunit_HSE_detector(
        time, multiunit_spikes, speed.values, SAMPLING_FREQUENCY,
        minimum_duration=np.timedelta64(15, 'ms'), zscore_threshold=2.0,
        close_event_threshold=np.timedelta64(0, 'ms'))
    multiunit_high_synchrony_times.index = (
        multiunit_high_synchrony_times.index.rename('replay_number'))
    multiunit_high_synchrony_labels = get_labels(
        multiunit_high_synchrony_times, time)
    is_multiunit_high_synchrony = multiunit_high_synchrony_labels > 0
    multiunit_high_synchrony_times = multiunit_high_synchrony_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds())

    return dict(
        multiunit=multiunit,
        multiunit_spikes=multiunit_spikes,
        multiunit_firing_rate=multiunit_firing_rate,
        multiunit_high_synchrony_times=multiunit_high_synchrony_times,
        multiunit_high_synchrony_labels=multiunit_high_synchrony_labels,
        multiunit_rate_change=multiunit_rate_change,
        multiunit_rate_zscore=multiunit_rate_zscore,
        is_multiunit_high_synchrony=is_multiunit_high_synchrony)


def get_spikes(neuron_info, time_function):
    time = time_function()
    neuron_info = neuron_info.loc[
        (neuron_info.numspikes > 100) &
        neuron_info.area.isin(BRAIN_AREAS) &
        (neuron_info.type == 'principal')]
    spikes = get_all_spike_indicators(
        neuron_info.index, ANIMALS, time_function).reindex(time)

    return spikes


def get_position_boundaries(position_info):
    max_df = position_info.groupby('arm_name').linear_position.max()
    min_df = position_info.groupby('arm_name').linear_position.min()
    return dict(center_well_position=min_df['Center Arm'],
                choice_position=max_df['Center Arm'],
                left_arm_start=min_df['Left Arm'],
                left_well_position=max_df['Left Arm'],
                right_arm_start=min_df['Right Arm'],
                right_well_position=max_df['Right Arm'],
                max_linear_distance=position_info.linear_distance.max())


def load_data(epoch_key):
    logger.info('Loading data...')
    time = get_trial_time(epoch_key, ANIMALS)
    time = (pd.Series(np.ones_like(time, dtype=np.float), index=time)
            .resample('2ms').mean()
            .index)

    def _time_function(*args, **kwargs):
        return time

    position_info = (
        get_interpolated_position_dataframe(
            epoch_key, ANIMALS, _time_function)
        .dropna(subset=['linear_position', 'speed']))

    time = position_info.index
    speed = position_info['speed']
    position_boundaries = get_position_boundaries(position_info)

    neuron_info = make_neuron_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False)
    spikes = get_spikes(neuron_info, _time_function)

    tetrode_info = make_tetrode_dataframe(ANIMALS, epoch_key=epoch_key)

    track_graph, _ = make_track_graph(epoch_key, ANIMALS)

    logger.info('Finding multiunit high synchrony events...')
    adhoc_multiunit = get_adhoc_multiunit(
        speed, tetrode_info, _time_function)

    logger.info('Finding ripple times...')
    adhoc_ripple = get_adhoc_ripple(epoch_key, tetrode_info, time)

    logger.info('Estimating gamma power...')
    gamma_power = estimate_gamma_power(time, tetrode_info)

    logger.info('Estimating theta power...')
    theta_power = estimate_theta_power(time, tetrode_info)

    return {
        'position_info': position_info,
        'tetrode_info': tetrode_info,
        'neuron_info': neuron_info,
        'spikes': spikes,
        'track_graph': track_graph,
        'sampling_frequency': SAMPLING_FREQUENCY,
        **position_boundaries,
        **adhoc_ripple,
        **adhoc_multiunit,
        **gamma_power,
        **theta_power,
    }


def load_detector(epoch_key):
    animal, day, epoch = epoch_key
    file_name = f'{animal}_{day:02}_{epoch:02}_replay_detector.gz'
    return joblib.load(join(PROCESSED_DATA_DIR, file_name))
