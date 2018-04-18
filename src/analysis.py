
import numpy as np
import pandas as pd

from loren_frank_data_processing import (get_interpolated_position_dataframe,
                                         get_LFPs,
                                         get_multiunit_indicator_dataframe,
                                         get_spike_indicator_dataframe,
                                         get_trial_time, make_neuron_dataframe,
                                         make_tetrode_dataframe)
from replay_identification import ReplayDetector
from replay_identification.lfp_likelihood import estimate_ripple_band_power
from ripple_detection import Kay_ripple_detector

_BRAIN_AREAS = 'CA1'


def detect_epoch_ripples(epoch_key, animals, sampling_frequency,
                         brain_areas=_BRAIN_AREAS,
                         minimum_duration=pd.Timedelta(milliseconds=15),
                         zscore_threshold=3, speed=None):
    '''Returns a list of tuples containing the start and end times of
    ripples. Candidate ripples are computed via the ripple detection
    function and then filtered to exclude ripples where the animal was
    still moving.
    '''

    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)
    brain_areas = [brain_areas] if isinstance(
        brain_areas, str) else brain_areas
    is_brain_areas = tetrode_info.area.isin(brain_areas)
    if 'CA1' in brain_areas:
        is_brain_areas = is_brain_areas & tetrode_info.validripple
    tetrode_keys = tetrode_info[is_brain_areas].index
    lfps = get_LFPs(tetrode_keys, animals)
    time = lfps.index
    if speed is None:
        speed = get_interpolated_position_dataframe(
            epoch_key, animals).speed

    return Kay_ripple_detector(
        time, lfps.values, speed.values, sampling_frequency,
        minimum_duration=minimum_duration, zscore_threshold=zscore_threshold)


def get_ripple_indicator(epoch_key, animals, ripple_times, position_info):
    time = get_trial_time(epoch_key, animals)
    ripple_indicator = pd.Series(np.zeros_like(time, dtype=bool), index=time)
    for _, start_time, end_time in ripple_times.itertuples():
        if position_info.loc[start_time:end_time].speed.max() <= 4:
            ripple_indicator.loc[start_time:end_time] = True

    return ripple_indicator


def get_ripple_power(epoch_key, animals, sampling_frequency):
    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)
    is_brain_areas = tetrode_info.area.isin(['CA1'])
    is_brain_areas = is_brain_areas & tetrode_info.validripple
    tetrode_keys = tetrode_info[is_brain_areas].index
    lfps = get_LFPs(tetrode_keys, animals)
    return estimate_ripple_band_power(lfps.values, sampling_frequency)


def get_spikes(epoch_key, animals):
    neuron_info = make_neuron_dataframe(animals).xs(
        epoch_key, drop_level=False)
    neuron_info = neuron_info[
        neuron_info.area.isin(['CA1', 'CA3']) &
        (neuron_info.numspikes > 0)]
    return np.stack([get_spike_indicator_dataframe(neuron_key, animals).values
                     for neuron_key in neuron_info.index], axis=-1)


def get_multiunit(epoch_key, animals):
    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)
    is_brain_areas = tetrode_info.area.isin(['CA1', 'CA3'])
    is_brain_areas = is_brain_areas & tetrode_info.validripple
    tetrode_keys = tetrode_info[is_brain_areas].index

    return np.stack([get_multiunit_indicator_dataframe(tetrode_key, animals)
                     for tetrode_key in tetrode_keys], axis=-1)


def identify_replays(epoch_key, animals, sampling_frequency,
                     use_likelihoods=['speed']):
    position_info = get_interpolated_position_dataframe(epoch_key, animals)
    speed = position_info.speed
    linear_distance = position_info.linear_distance.values
    power = get_ripple_power(epoch_key, animals, sampling_frequency)

    if 'spikes' in use_likelihoods:
        spikes = get_spikes(epoch_key, animals)
    else:
        spikes = None

    if 'multiunit' in use_likelihoods:
        multiunit = get_multiunit(epoch_key, animals)
    else:
        multiunit = None

    ripple_times = detect_epoch_ripples(epoch_key, animals, sampling_frequency,
                                        speed=speed)
    is_replay = get_ripple_indicator(
        epoch_key, animals, ripple_times, position_info)

    detector = ReplayDetector(replay_speed=20)
    detector.fit(
        is_replay, speed.values, power, linear_distance, spikes, multiunit)
    time = position_info.index.total_seconds().values
    return detector.predict(
        speed.values, power, linear_distance, spikes, multiunit, time=time,
        use_likelihoods=use_likelihoods), detector


def summarize_replays():
    pass
