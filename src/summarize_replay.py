
from logging import getLogger

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage.measurements import label
from scipy.stats import linregress

from loren_frank_data_processing import reshape_to_segments
from replay_classification import SortedSpikeDecoder

logger = getLogger(__name__)


def start_time(s):
    return s.index[0]


def end_time(s):
    return s.index[-1]


def max_probability(df):
    return df.max()


def duration(df):
    return (df.index[-1] - df.index[0]).total_seconds()


def get_replay_times(results, probability_threshold=0.8,
                     minimum_duration=0.000):

    replay_labels = label(
        results.replay_probability > probability_threshold)[0]
    labels = pd.DataFrame(
        {'replay_number': replay_labels,
         'replay_probability': results.replay_probability.values},
        index=results.time.to_index())

    grouper = labels.loc[labels.replay_number > 0].groupby('replay_number')
    replay_times = (grouper
                    .agg([start_time, end_time, duration, max_probability]))
    replay_times.columns = replay_times.columns.get_level_values(1)

    replay_times = replay_times.loc[replay_times.duration > minimum_duration]
    labels[~labels.isin(replay_times.index.values)] = 0

    return replay_times, labels


def summarize_replays(replay_info, detector_results, decoder_results, data,
                      sampling_frequency=1500,
                      position_metric='linear_distance'):
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
    n_unique_spiking = []
    pct_unique_spiking = []
    n_total_spikes = []
    pct_total_spikes = []
    avg_replay_speed = []

    for r, decoder_result in zip(replay_info.itertuples(), decoder_results):
        # Get detector posterior
        cur_detector_results = (
            detector_results.sel(time=slice(r.start_time, r.end_time))
            .assign_coords(time=lambda da: da.time - r.start_time))
        detector_posterior.append(
            cur_detector_results.sel(state='Replay').posterior.drop('state'))
        detector_likelihood.append(
            np.exp(np.log(cur_detector_results.likelihood)
                   .diff(dim='state')).squeeze())

        density = decoder_result.results.posterior_density.sum('state')

        # Get decoder posterior
        decoder_posterior.append(decoder_result.results.posterior_density)

        # Classify Replay
        replay_type.append(decoder_result.predicted_state())
        replay_type_confidence.append(
            decoder_result.predicted_state_probability())

        # Replay Motion
        motion_slope.append(_get_replay_motion(r, density, position_metric))

        # How far replay moves
        replay_movement_distance.append(_get_replay_movement(density))

        # Replay speed
        avg_replay_speed.append(np.mean(np.abs(_get_replay_velocity(
            density, sampling_frequency))))

        # Add stats about spikes
        n_unique = _n_unique_spiking(decoder_result.spikes)
        n_neurons = decoder_result.spikes.shape[1]
        n_total = _n_total_spikes(decoder_result.spikes)
        n_possible_spikes = decoder_result.spikes.size
        n_unique_spiking.append(n_unique)
        pct_unique_spiking.append(n_unique / n_neurons)
        n_total_spikes.append(n_total)
        pct_total_spikes.append(n_total / n_possible_spikes)

    replay_info['replay_type'] = replay_type
    replay_info['replay_type_confidence'] = replay_type_confidence
    replay_info['replay_motion_slope'] = motion_slope
    replay_info['replay_motion_type'] = pd.cut(
        replay_info['replay_motion_slope'], MOTION_BOUNDS,
        labels=['Towards', 'Neither', 'Away'])
    replay_info['replay_movement_distance'] = replay_movement_distance
    replay_info['n_unique_spiking'] = n_unique_spiking
    replay_info['pct_unique_spiking'] = pct_unique_spiking
    replay_info['n_spikes'] = n_total_spikes
    replay_info['pct_total_spikes'] = pct_total_spikes
    replay_info['avg_replay_speed'] = avg_replay_speed

    detector_posterior = (xr.concat(detector_posterior, dim=replay_info.index)
                          .rename('detector_posterior'))
    detector_likelihood = (xr.concat(
        detector_likelihood, dim=replay_info.index)
        .rename('detector_likelihood'))
    decoder_posterior = (xr.concat(decoder_posterior, dim=replay_info.index)
                         .rename('decoder_posterior'))
    replay_densities = xr.merge(
        (detector_posterior, detector_likelihood, decoder_posterior),
        join='inner')

    return replay_info, replay_densities


def decode_replays(data, replay_detector, is_replay, replay_info,
                   sampling_frequency, position_metric='linear_distance',
                   use_smoother=True):
    initial_conditions = {
        'linear_distance': 'Uniform',
        'linear_position': 'Uniform',
        'linear_position2': 'Uniform',
    }

    decoder = SortedSpikeDecoder(
        place_bin_size=replay_detector.place_bin_size,
        replay_speedup_factor=replay_detector.replay_speed,
        knot_spacing=replay_detector.spike_model_knot_spacing,
        spike_model_penalty=replay_detector.spike_model_penalty,
        replay_orders=['Forward', 'Reverse'],
    ).fit(
        position=data['position_info'][position_metric].values,
        experimental_condition=data['position_info'].task.values,
        spikes=data['spikes'].values,
        is_training=(is_replay.replay_number == 0).values,
        initial_conditions=initial_conditions[position_metric])

    test_spikes = reshape_to_segments(
        data['spikes'], replay_info.loc[:, ['start_time', 'end_time']],
        sampling_frequency=sampling_frequency)

    decoder_results = [
        decoder.predict(test_spikes.loc[replay_number].values,
                        test_spikes.loc[replay_number].index,
                        use_smoother=use_smoother)
        for replay_number in replay_info.index]

    return decoder_results, decoder


def _get_replay_motion(replay_info, posterior_density,
                       position_metric='linear_distance'):
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
        replay_position - getattr(replay_info, position_metric))
    slope = linregress(
        posterior_density.get_index('time').total_seconds().values,
        replay_distance_from_animal_position).slope
    return slope


def _get_replay_movement(posterior_density):
    replay_position = posterior_density.position.values[
        np.log(posterior_density).argmax('position').values]
    return np.diff(np.quantile(replay_position, [0.25, 0.75]))[0]


def _get_replay_velocity(posterior_density, sampling_frequency):
    replay_position = posterior_density.position.values[
        np.log(posterior_density).argmax('position').values]
    return np.diff(replay_position) * sampling_frequency


def _average_credible_interval_size(posterior_density):
    pass


def _n_unique_spiking(spikes):
    '''Number of units that spike per ripple
    '''
    if spikes.ndim > 2:
        return np.sum(~np.isnan(spikes), axis=(1, 2)).nonzero()[0].size
    else:
        return spikes.sum(axis=0).nonzero()[0].size


def _n_total_spikes(spikes):
    '''Total number of spikes per ripple
    '''
    if spikes.ndim > 2:
        return np.any(~np.isnan(spikes), axis=2).sum()
    else:
        return int(spikes.sum())


def _get_n_time_by_label(labels, overlap_info):
    is_overlap_labels = labels.isin(
        overlap_info.index.get_level_values(labels.name))
    overlap_labels1 = labels.loc[is_overlap_labels]
    overlap_labels1 = (overlap_labels1
                       .groupby(overlap_labels1)
                       .agg(len)
                       .rename(f'total_{overlap_labels1.name}'))

    return overlap_labels1


def compare_overlap(labels1, labels2, info1, info2, sampling_frequency,
                    epoch_key, name1, name2):
    labels1 = labels1.copy().rename('labels1')
    labels2 = labels2.copy().rename('labels2')
    is_overlap = (labels1 > 0) & (labels2 > 0)

    if any(is_overlap):
        overlap_info = pd.concat(
            (labels1.loc[is_overlap], labels2.loc[is_overlap]), axis=1)
        overlap_info = (overlap_info
                        .groupby(overlap_info.columns.tolist())
                        .agg(len)
                        .sort_index()
                        .rename('n_overlap')
                        .to_frame())
        overlap_info['overlap_duration'] = (
            overlap_info.n_overlap / sampling_frequency)

        overlap_labels1 = _get_n_time_by_label(labels1, overlap_info)
        overlap_labels2 = _get_n_time_by_label(labels2, overlap_info)

        percentage_overlap = {
            'jaccard_similarity': lambda df: (
                df.n_overlap / (
                    df.total_labels1 + df.total_labels2 - df.n_overlap))
        }

        overlap_info = (overlap_info
                        .join(overlap_labels1)
                        .join(overlap_labels2)
                        .assign(**percentage_overlap))

        start_time1 = (info1.set_index('replay_number').loc[
            overlap_info.index.get_level_values(0).values,
            ['start_time', 'end_time']])
        start_time2 = (info2.set_index('replay_number').loc[
            overlap_info.index.get_level_values(1).values,
            ['start_time', 'end_time']])

        time_difference = (start_time1.values - start_time2.values)

        overlap_info['start_time_difference'] = (
            time_difference[:, 0] / np.timedelta64(1, 's'))
        overlap_info['end_time_difference'] = (
            time_difference[:, 1] / np.timedelta64(1, 's'))

        replay_id1 = (info1
                      .reset_index()
                      .set_index('replay_number')
                      .replay_id
                      .loc[overlap_info.index.get_level_values(0)]
                      .values)
        replay_id2 = (info2
                      .reset_index()
                      .set_index('replay_number')
                      .replay_id
                      .loc[overlap_info.index.get_level_values(1)]
                      .values)
        replay_id_index = pd.MultiIndex.from_arrays(
            [replay_id1, replay_id2],
            names=['replay_number1', 'replay_number2'])

        overlap_info['animal'] = epoch_key[0]
        overlap_info['day'] = epoch_key[1]
        overlap_info['epoch'] = epoch_key[2]
        overlap_info['data_source1'] = name1
        overlap_info['data_source2'] = name2

        return overlap_info.set_index(replay_id_index)
    else:
        COLUMN_NAMES = ['replay_number1', 'replay_number2', 'n_overlap',
                        'overlap_duration', 'total_labels1', 'total_labels2',
                        'jaccard_similarity',
                        'start_time_difference', 'end_time_difference',
                        'animal', 'day', 'epoch', 'data_source1',
                        'data_source2']
        empty_df = (pd.DataFrame([], columns=COLUMN_NAMES)
                    .set_index(['replay_number1', 'replay_number2'])
                    )
        return empty_df


def add_epoch_info_to_dataframe(df, epoch_key, data_source):
    df = df.reset_index()
    df['animal'], df['day'], df['epoch'] = epoch_key
    df['data_source'] = data_source
    df['replay_id'] = (df.animal + '_' +
                       df.day.astype(str).str.zfill(2) + '_' +
                       df.epoch.astype(str).str.zfill(2) + '_' +
                       df.replay_number.astype(str).str.zfill(3) + '_' +
                       df.data_source)
    return df.set_index('replay_id')
