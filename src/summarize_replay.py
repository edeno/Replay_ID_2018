
from logging import getLogger

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage.measurements import label
from scipy.stats import linregress

from loren_frank_data_processing import reshape_to_segments
from loren_frank_data_processing.track_segment_classification import (
    get_track_segments_from_graph, project_points_to_segment)
from spectral_connectivity import Connectivity, Multitaper

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


def summarize_replays(replay_info, decoder_results, data,
                      sampling_frequency=1500,
                      position_metric='linear_distance'):
    replay_metrics = []

    for row, result in zip(replay_info.itertuples(), decoder_results):
        replay_metrics.append(
            get_replay_metrics(
                row.start_time, row.end_time,
                result.acausal_posterior, **data))

    replay_metrics = pd.DataFrame(replay_metrics, index=replay_info.index)
    replay_info = pd.concat((replay_info, replay_metrics), axis=1)

    return replay_info


def decode_replays(decoder, data, replay_info, use_smoother=True):
    decoder_results = []
    for replay_number in replay_info.index:
        start_time, end_time = replay_info.loc[
            replay_number, ['start_time', 'end_time']].values
        test_multiunit = data['multiunit'].sel(
            time=slice(start_time, end_time))
        decoder_results.append(
            decoder
            .predict(test_multiunit, time=test_multiunit.time)
            .drop(['likelihood', 'causal_posterior']))

    return decoder_results


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
                    df.total_labels1 + df.total_labels2 - df.n_overlap)),
            'overlap_coefficient': lambda df: (
                df.n_overlap / (
                    np.minimum(df.total_labels1, df.total_labels2)))
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


def _center_time(time):
    time_diff = np.diff(time)[0] if np.diff(time).size > 0 else 0
    return time + time_diff / 2


def get_replay_triggered_power(lfps, replay_info, tetrode_info,
                               multitaper_params,
                               window_offset=(-0.250, 0.250),
                               sampling_frequency=1500):
    ripple_locked_lfps = reshape_to_segments(
        lfps, replay_info.loc[:, ['start_time', 'end_time']],
        window_offset=window_offset, sampling_frequency=sampling_frequency)
    ripple_locked_lfps = (ripple_locked_lfps.to_xarray().to_array()
                          .rename({'variable': 'tetrodes'})
                          .transpose('time', 'replay_id', 'tetrodes')
                          .dropna('replay_id'))
    ripple_locked_lfps = (ripple_locked_lfps
                          - ripple_locked_lfps.mean(['replay_id']))
    start_time = ripple_locked_lfps.time.min().values / np.timedelta64(1, 's')
    m = Multitaper(ripple_locked_lfps.values, **multitaper_params,
                   start_time=start_time)
    c = Connectivity.from_multitaper(m)
    dimension_names = ['time', 'frequency', 'tetrode']
    data_vars = {
        'power': (dimension_names, c.power())}
    coordinates = {
        'time': _center_time(c.time),
        'frequency': c.frequencies + np.diff(c.frequencies)[0] / 2,
        'tetrode': lfps.columns,
    }
    return (xr.Dataset(data_vars, coords=coordinates)
            .sel(frequency=slice(0, 300)))


def maximum_a_posteriori_estimate(posterior_density):
    '''

    Parameters
    ----------
    posterior_density : xarray.DataArray, shape (n_time, n_x_bins, n_y_bins)

    Returns
    -------
    map_estimate : ndarray, shape (n_time,)

    '''
    try:
        stacked_posterior = np.log(posterior_density.stack(
            z=['x_position', 'y_position']))
        map_estimate = stacked_posterior.z[stacked_posterior.argmax('z')]
        map_estimate = np.asarray(map_estimate.values.tolist())
    except KeyError:
        map_estimate = posterior_density.position[
            np.log(posterior_density).argmax('position')]
        map_estimate = np.asarray(map_estimate)[:, np.newaxis]
    return map_estimate


def _get_closest_ind(map_estimate, all_positions):
    map_estimate = np.asarray(map_estimate)
    all_positions = np.asarray(all_positions)
    return np.argmin(np.linalg.norm(
        map_estimate[:, np.newaxis, :] - all_positions[np.newaxis, ...],
        axis=-2), axis=1)


def _get_projected_track_positions(position, track_segments, track_segment_id):
    projected_track_positions = project_points_to_segment(
        track_segments, position)
    n_time = projected_track_positions.shape[0]
    projected_track_positions = projected_track_positions[(
        np.arange(n_time), track_segment_id)]
    return projected_track_positions


def calculate_replay_distance(track_graph, map_estimate, actual_positions,
                              actual_track_segment_ids, position_info,
                              center_well_id=0):
    '''Calculate the linearized distance between the replay position and the
    animal's physical position for each time point.

    Parameters
    ----------
    track_graph : networkx.Graph
        Nodes and edges describing the track
    map_estimate : ndarray, shape (n_time, n_position_dims)
        Maximum aposterior estimate of the replay
    actual_positions : ndarray, shape (n_time, 2)
        Animal's physical position during the replay
    actual_track_segment_ids : ndarray, shape (n_time,)
        Animal's track segment ID during the replay
    position_info : pandas.DataFrame
    center_well_id : hasable, optional

    Returns
    -------
    replay_distance_from_actual_position : ndarray, shape (n_time,)
    replay_distance_from_center_well : ndarray, shape (n_time,)

    '''

    actual_track_segment_ids = (
        np.asarray(actual_track_segment_ids).squeeze().astype(int))

    # Find 2D position closest to replay position
    n_position_dims = map_estimate.shape[1]
    if n_position_dims == 1:
        closest_ind = _get_closest_ind(
            map_estimate, position_info.linear_position)
    else:
        closest_ind = _get_closest_ind(
            map_estimate, position_info.loc[:, ['x_position', 'y_position']])

    df = position_info.iloc[closest_ind]
    replay_positions = df.loc[:, ['x_position', 'y_position']].values
    replay_track_segment_ids = (
        df.loc[:, ['track_segment_id']].values.squeeze().astype(int))

    track_segments = get_track_segments_from_graph(track_graph)

    # Project positions to closest edge on graph
    replay_positions = _get_projected_track_positions(
        replay_positions, track_segments, replay_track_segment_ids)
    actual_positions = _get_projected_track_positions(
        actual_positions, track_segments, actual_track_segment_ids)

    edges = np.asarray(track_graph.edges)
    replay_edge_ids = edges[replay_track_segment_ids]
    actual_edge_ids = edges[actual_track_segment_ids]
    replay_distance_from_actual_position = []
    replay_distance_from_center_well = []

    zipped = zip(
        actual_edge_ids, replay_edge_ids, actual_positions, replay_positions,
        actual_track_segment_ids, replay_track_segment_ids)

    for (actual_edge_id, replay_edge_id, actual_pos, replay_pos,
         actual_id, replay_id) in zipped:
        track_graph1 = track_graph.copy()
        if actual_id != replay_id:
            # Add actual position node
            node_name = 'actual_position'
            node1, node2 = actual_edge_id
            nx.add_path(track_graph1, [node1, node_name, node2])
            track_graph1.remove_edge(node1, node2)
            track_graph1.nodes[node_name]['pos'] = tuple(actual_pos)

            # Add replay position node
            node_name = 'replay_position'
            node1, node2 = replay_edge_id
            nx.add_path(track_graph1, [node1, node_name, node2])
            track_graph1.remove_edge(node1, node2)
            track_graph1.nodes[node_name]['pos'] = tuple(replay_pos)
        else:
            node1, node2 = actual_edge_id

            nx.add_path(track_graph1,
                        [node1, 'actual_position', 'replay_position', node2])
            nx.add_path(track_graph1,
                        [node1, 'replay_position', 'actual_position', node2])

            track_graph1.nodes['actual_position']['pos'] = tuple(actual_pos)
            track_graph1.nodes['replay_position']['pos'] = tuple(replay_pos)
            track_graph1.remove_edge(node1, node2)

        # Calculate distance between all nodes
        for edge in track_graph1.edges(data=True):
            track_graph1.edges[edge[:2]]['distance'] = np.linalg.norm(
                track_graph1.nodes[edge[0]]['pos'] -
                np.array(track_graph1.nodes[edge[1]]['pos']))

        replay_distance_from_actual_position.append(
            nx.shortest_path_length(
                track_graph1, source='actual_position',
                target='replay_position', weight='distance'))
        replay_distance_from_center_well.append(
            nx.shortest_path_length(
                track_graph1, source=center_well_id,
                target='replay_position', weight='distance'))
    replay_distance_from_actual_position = np.asarray(
        replay_distance_from_actual_position)
    replay_distance_from_center_well = np.asarray(
        replay_distance_from_center_well)

    return (replay_distance_from_actual_position,
            replay_distance_from_center_well)


def get_replay_metrics(start_time, end_time, posterior, spikes,
                       ripple_power_change, ripple_power_zscore,
                       low_gamma_power_change, low_gamma_power_zscore,
                       high_gamma_power_change, high_gamma_power_zscore,
                       theta_power_change, theta_power_zscore,
                       multiunit_firing_rate, multiunit_rate_change,
                       multiunit_rate_zscore, position_info, track_graph,
                       sampling_frequency, max_linear_distance,
                       left_well_position, **kwargs):

    time_slice = slice(start_time, end_time)
    replay_spikes = spikes.loc[time_slice]
    time = replay_spikes.index / np.timedelta64(1, 's')

    replay_ripple_power_change = (ripple_power_change
                                  .loc[time_slice].values.mean())
    replay_ripple_power_zscore = (ripple_power_zscore
                                  .loc[time_slice].values.mean())

    replay_low_gamma_power_change = (
        low_gamma_power_change.loc[time_slice].values.mean())
    replay_low_gamma_power_zscore = (
        low_gamma_power_zscore.loc[time_slice].values.mean())

    replay_high_gamma_power_change = (
        high_gamma_power_change.loc[time_slice].values.mean())
    replay_high_gamma_power_zscore = (
        high_gamma_power_zscore.loc[time_slice].values.mean())

    replay_theta_power_change = (
        theta_power_change.loc[time_slice].values.mean())
    replay_theta_power_zscore = (
        theta_power_zscore.loc[time_slice].values.mean())

    replay_multiunit_firing_rate = (
        multiunit_firing_rate.loc[time_slice].values.mean())
    replay_multiunit_firing_rate_change = (
        multiunit_rate_change.loc[time_slice].values.mean())
    replay_multiunit_firing_rate_zscore = (
        multiunit_rate_zscore.loc[time_slice].values.mean())
    replay_position_info = position_info.loc[time_slice]

    map_estimate = maximum_a_posteriori_estimate(posterior)
    hpd_threshold = highest_posterior_density(
        posterior.sum("state"), coverage=0.95)
    isin_hpd = posterior.sum("state") >= hpd_threshold[:, np.newaxis]
    spatial_coverage = (
        isin_hpd * np.diff(posterior.position)[0]).sum("position").values
    n_position_bins = (posterior.sum("state", skipna=True)
                       > 0).sum("position").values[0]
    spatial_coverage_percentage = (isin_hpd.sum("position") /
                                   n_position_bins).values

    actual_positions = (position_info
                        .loc[time_slice, ['x_position', 'y_position']]
                        .values)
    actual_track_segment_ids = (position_info
                                .loc[time_slice, 'track_segment_id']
                                .values.squeeze().astype(int))

    (replay_distance_from_actual_position,
     replay_distance_from_center_well) = calculate_replay_distance(
        track_graph, map_estimate, actual_positions,
        actual_track_segment_ids, position_info)
    try:
        replay_total_displacement = np.abs(
            replay_distance_from_actual_position[-1] -
            replay_distance_from_actual_position[0])
    except IndexError:
        replay_total_displacement = np.nan

    map_estimate = map_estimate.squeeze()

    return {
        'replay_distance_from_actual_position': np.mean(
            replay_distance_from_actual_position),
        'replay_speed': np.abs(np.gradient(
            replay_distance_from_actual_position, time)),
        'replay_velocity_actual_position': np.mean(
            np.gradient(replay_distance_from_actual_position, time)),
        'replay_velocity_center_well': np.mean(
            np.gradient(replay_distance_from_center_well, time)),
        'replay_distance_from_center_well': np.mean(
            replay_distance_from_center_well),
        'replay_norm_distance_from_center_well': np.mean(
            replay_distance_from_center_well) / max_linear_distance,
        'replay_start_distance_from_center_well': (
            replay_distance_from_center_well[0]),
        'replay_norm_start_distance_from_center_well': (
            replay_distance_from_center_well[0] / max_linear_distance),
        'replay_end_distance_from_center_well': (
            replay_distance_from_center_well[-1]),
        'replay_norm_end_distance_from_center_well': (
            replay_distance_from_center_well[-1] / max_linear_distance),
        'replay_linear_position': np.mean(map_estimate),
        'replay_norm_linear_position': (
            np.mean(map_estimate) / left_well_position),
        'replay_start_linear_position': map_estimate[0],
        'replay_norm_start_linear_position': (
            map_estimate[0] / left_well_position),
        'replay_end_linear_position': map_estimate[-1],
        'replay_norm_end_linear_position': (
            map_estimate[-1] / left_well_position),
        'replay_total_distance': np.sum(
            np.abs(np.diff(replay_distance_from_actual_position))),
        'replay_total_displacement': replay_total_displacement,
        'ripple_power_change': replay_ripple_power_change,
        'ripple_power_zscore': replay_ripple_power_zscore,
        'low_gamma_power_change': replay_low_gamma_power_change,
        'low_gamma_power_zscore': replay_low_gamma_power_zscore,
        'high_gamma_power_change': replay_high_gamma_power_change,
        'high_gamma_power_zscore': replay_high_gamma_power_zscore,
        'theta_power_change': replay_theta_power_change,
        'theta_power_zscore': replay_theta_power_zscore,
        'multiunit_firing_rate': replay_multiunit_firing_rate,
        'multiunit_firing_rate_change': replay_multiunit_firing_rate_change,
        'multiunit_firing_rate_zscore': replay_multiunit_firing_rate_zscore,
        'n_unique_spiking': (replay_spikes.sum() > 0).sum(),
        'frac_unique_spiking': (replay_spikes.sum() > 0).mean(),
        'n_total_spikes': replay_spikes.sum().sum(),
        'sorted_spike_rate': (
            replay_spikes.mean() * sampling_frequency).mean(),
        'actual_linear_distance': replay_position_info.linear_distance.mean(),
        'actual_norm_linear_distance': (
            replay_position_info.linear_distance.mean() / max_linear_distance),
        'actual_linear_position': replay_position_info.linear_position.mean(),
        'actual_norm_linear_position': (
            replay_position_info.linear_position.mean() / left_well_position),
        'actual_speed': replay_position_info.speed.mean(),
        'actual_velocity_center_well': (
            replay_position_info.linear_velocity.mean()),
        'spatial_coverage': np.median(spatial_coverage),
        'spatial_coverage_percentage': np.median(spatial_coverage_percentage)
    }


def get_overlap_times(*args):
    is_overlap = args[0].squeeze() > 0
    for df in args[1:]:
        is_overlap = is_overlap & (df.squeeze() > 0)

    return is_overlap


def get_overlap_replay_number(label, is_overlap):
    return np.asarray(label.loc[is_overlap].unique())


def get_non_overlap_replay_number(label, is_overlap):
    all_id = label.unique()
    all_id = set(all_id[all_id > 0])
    overlap_id = set(get_overlap_replay_number(label, is_overlap))

    return np.asarray(list(all_id - overlap_id))


def convert_replay_number_to_id(replay_number, epoch_key, data_source):
    animal, day, epoch = epoch_key
    return pd.Index(replay_number).map(
        lambda number:
            f'{animal}_{day:02}_{epoch:02}_{number:03}_{data_source}')


def get_non_overlap_info(labels1, labels2, data_source1, data_source2,
                         epoch_key):
    is_overlap = get_overlap_times(labels1, labels2)

    no_overlap_id1 = convert_replay_number_to_id(
        get_non_overlap_replay_number(labels1, is_overlap), epoch_key,
        data_source1)

    no_overlap_id2 = convert_replay_number_to_id(
        get_non_overlap_replay_number(labels2, is_overlap), epoch_key,
        data_source2)

    return pd.concat(
        (pd.DataFrame(
            {'data_source': data_source1,
             'no_overlap_with': data_source2}, index=no_overlap_id1),
         pd.DataFrame(
            {'data_source': data_source2,
             'no_overlap_with': data_source1}, index=no_overlap_id2)),
        axis=0)


def highest_posterior_density(posterior_density, coverage=0.95):
    """
    Same as credible interval
    https://stats.stackexchange.com/questions/240749/how-to-find-95-credible-interval

    Parameters
    ----------
    posterior_density : xarray.DataArray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
    coverage : float, optional

    Returns
    -------
    threshold : ndarray, shape (n_time,)

    """
    try:
        posterior_density = posterior_density.stack(
            z=["x_position", "y_position"]
        ).values
    except KeyError:
        posterior_density = posterior_density.values
    const = np.sum(posterior_density, axis=1, keepdims=True)
    sorted_norm_posterior = np.sort(posterior_density, axis=1)[:, ::-1] / const
    posterior_less_than_coverage = np.cumsum(
        sorted_norm_posterior, axis=1) >= coverage
    crit_ind = np.argmax(posterior_less_than_coverage, axis=1)
    # Handle case when there are no points in the posterior less than coverage
    crit_ind[posterior_less_than_coverage.sum(axis=1) == 0] = (
        posterior_density.shape[1] - 1
    )

    n_time = posterior_density.shape[0]
    threshold = sorted_norm_posterior[(
        np.arange(n_time), crit_ind)] * const.squeeze()
    return threshold
