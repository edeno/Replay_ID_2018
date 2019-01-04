import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

from .parameters import USE_LIKELIHOODS

HUE_ORDER = {
    'replay_motion_type': ['Away', 'Neither', 'Towards'],
    'replay_type': ['Inbound-Forward', 'Inbound-Reverse',
                    'Outbound-Forward', 'Outbound-Reverse', 'Unclassified']
}


def plot_data_source_counts(replay_info, kind='violin', **kwargs):
    '''Compare number of replays by data source.'''
    df = (replay_info
          .groupby(['animal', 'day', 'epoch'])['data_source']
          .value_counts()
          .rename('counts_by_epoch')
          .reset_index())

    g = sns.catplot(x='data_source', y='counts_by_epoch', data=df, kind=kind,
                    order=USE_LIKELIHOODS.keys(), aspect=2,
                    **kwargs)

    return g


def plot_proportion_events_by_data_source(replay_info, covariate,
                                          kind='violin', **plot_kwargs):
    '''covariate = {'replay_type', 'replay_motion_type'}'''
    group = ['data_source', 'animal', 'day', 'epoch']
    df = (replay_info.groupby(group)[covariate]
          .value_counts(normalize=True)
          .rename('proportion')
          .reset_index())
    g = sns.catplot(x=covariate, y='proportion', data=df,
                    hue='data_source', kind=kind, aspect=2,
                    hue_order=USE_LIKELIHOODS.keys(),
                    order=HUE_ORDER[covariate],
                    **plot_kwargs)
    return g


def plot_continuous_by_data_source(replay_info, covariate, kind='violin',
                                   **kwargs):
    '''covariate = {'replay_movement_distance', 'credible_interval_size',
                    'duration'}'''
    g = sns.catplot(x='data_source', y=covariate, order=USE_LIKELIHOODS.keys(),
                    data=replay_info, aspect=2, kind=kind, **kwargs)
    return g


def compare_jaccard_similarity_of_replays(overlap_info, replay_info):
    '''Compare the number of replays that overlap for at least one time point
       vs. the union of the total number of replays for each data source.'''

    fig, ax = plt.subplots(1, 1)
    names = list(USE_LIKELIHOODS.keys())
    n_overlap = (overlap_info
                 .loc[:, ['data_source1', 'data_source2']]
                 .groupby(['data_source1', 'data_source2'])
                 .agg(len)
                 .to_xarray()
                 .reindex({'data_source1': names, 'data_source2': names})
                 .fillna(0.0)
                 )
    n_total = replay_info.loc[:, 'data_source'].value_counts().reindex(names)
    n_total1 = n_total.copy().rename('n_total1')
    n_total1.index = n_total1.index.rename('data_source1')
    n_total2 = n_total.copy().rename('n_total2')
    n_total2.index = n_total2.index.rename('data_source2')

    n_overlap += n_overlap.T.values + np.eye(len(names)) * n_total.values
    jaccard_similarity = n_overlap / (
        n_total1.to_xarray() + n_total2.to_xarray() - n_overlap)
    jaccard_similarity = (jaccard_similarity
                          .to_dataframe(name='jaccard')
                          .unstack())
    jaccard_similarity.columns = jaccard_similarity.columns.droplevel(0)
    jaccard_similarity = jaccard_similarity.reindex(index=names, columns=names)
    ax = sns.heatmap(jaccard_similarity, annot=True, annot_kws={'size': 16},
                     vmin=0.0, vmax=1.0, ax=ax)
    return ax


def _vertical_median_line(x, **kwargs):
    plt.axvline(x.median(), **kwargs)


def compare_time_difference_of_overlapping_replays(
        overlap_info, time_difference):
    '''Of the replays that do overlap, compare the start or end time'''
    g = sns.FacetGrid(data=overlap_info,  row='data_source1',
                      col='data_source2', height=5, aspect=1.1,
                      row_order=USE_LIKELIHOODS.keys(),
                      col_order=USE_LIKELIHOODS.keys(),
                      sharey=False, sharex=False)
    kde_kws = {'color': 'red', 'lw': 3, 'shade': True}
    g.map(plt.axvline, x=0, color='k', linewidth=1, zorder=1)

    g.map(_vertical_median_line, time_difference,
          color='blue', linewidth=3, zorder=100, linestyle='--')
    g.map(sns.distplot, time_difference, hist=False, rug=False, kde_kws=kde_kws
          ).set_titles(
        '{row_name} < {col_name} | {row_name} > {col_name}', size=12)
    g.set(xlim=(-0.2, 0.2))

    for row_ind, column_ind in zip(*np.tril_indices_from(g.axes, 0)):
        g.axes[row_ind, column_ind].set_visible(False)

    return g


def compare_similarity_of_overlapping_replays(overlap_info):
    '''Of the replays that do overlap, compare how much they overlap in time'''
    g = sns.FacetGrid(data=overlap_info,  row='data_source1',
                      col='data_source2', height=5, aspect=1,
                      row_order=USE_LIKELIHOODS.keys(),
                      col_order=USE_LIKELIHOODS.keys(),
                      sharex=False, sharey=False)
    kde_kws = {'color': 'red', 'lw': 3, 'clip': (0, 1), 'shade': True}
    g.map(sns.distplot, 'jaccard_similarity', hist=False, rug=False,
          kde_kws=kde_kws).set_titles('{row_name}, {col_name}', size=15)
    g.set(xlim=(0, 1))

    for row_ind, column_ind in zip(*np.tril_indices_from(g.axes, 0)):
        g.axes[row_ind, column_ind].set_visible(False)

    return g


def plot_behavior(position_info, position_metric='linear_distance'):

    time = position_info.index.total_seconds()
    is_inbound = position_info.task == 'Inbound'
    is_outbound = position_info.task == 'Outbound'
    position = position_info[position_metric].values
    is_correct = position_info.is_correct.values
    speed = position_info.linear_speed.values

    p_min, p_max = np.nanmin(position), np.nanmax(position)
    s_min, s_max = np.nanmin(speed), np.nanmax(speed)

    fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharex=True,
                             constrained_layout=True)

    axes[0].fill_between(time, is_correct * p_min,
                         is_correct * p_max, color='#7fc97f',
                         where=is_correct,
                         alpha=0.25, label='Correct')
    axes[0].plot(time, position, '.', color='lightgrey')
    axes[0].plot(time[is_inbound], position[is_inbound], '.',
                 label='Inbound')
    axes[0].plot(time[is_outbound], position[is_outbound], '.',
                 label='Outbound')
    axes[0].set_ylabel(f'{position_metric} (m)')
    axes[0].set_ylim((p_min, p_max))
    axes[0].legend()

    axes[1].fill_between(time, is_correct * s_min,
                         is_correct * s_max, color='#7fc97f',
                         where=is_correct, alpha=0.25)
    axes[1].plot(time, speed, color='#7570b3', linewidth=1)
    axes[1].axhline(4, linestyle='--', color='black')
    axes[1].set_ylabel('linear_speed (m / s)')
    axes[1].set_ylim((s_min, s_max))
    axes[1].set_xlabel('Time (s)')

    plt.xlim((time.min(), time.max()))
    sns.despine()


def plot_replay_by_place_field(spikes, place_field_firing_rates,
                               place_bin_centers, ax=None, cmap=None):
    '''

    Parameters
    ----------
    spikes : ndarray, shape (n_time, n_neurons)
    place_field_firing_rates: ndarray, shape (n_neurons, n_place_bins)
    place_bin_centers: ndarray, shape (n_place_bins,)

    '''
    ax = ax or plt.gca()
    AVG_PLACE_FIELD_SIZE = 25
    n_colors = int(np.ceil(np.ptp(place_bin_centers) / AVG_PLACE_FIELD_SIZE))
    cmap = cmap or ListedColormap(sns.color_palette('hls', n_colors))

    cmap = plt.get_cmap(cmap)
    place_colors = cmap(np.linspace(0.0, 1.0, place_bin_centers.size))
    neuron_to_place_bin = np.argmax(place_field_firing_rates, axis=1)

    time_ind, neuron_inds = np.nonzero(spikes)
    colors = place_colors[neuron_to_place_bin[neuron_inds]]

    for neuron_ind, color in zip(neuron_inds, colors):
        ax.plot(place_bin_centers, place_field_firing_rates[neuron_ind],
                color=color, linewidth=2)

    ax.set_xlim((place_bin_centers.min(), place_bin_centers.max()))
    ax.set_xlabel('Position')
    ax.set_ylabel('Firing Rate (spikes / s)')
    sns.despine()

    return ax


def plot_replay_spiking_ordered_by_place_fields(
        spikes, place_field_firing_rates, place_bin_centers,
        ax=None, cmap=None, sampling_frequency=1, time=None):
    '''Plot spikes by the positiion of their maximum place field firing rate.

    Parameters
    ----------
    spikes : ndarray, shape (n_time, n_neurons)
    place_field_firing_rates : ndarray, shape (n_neurons, n_place_bins)
    place_bin_centers : ndarray, shape (n_place_bins,)
    ax : None or matplotlib axis, optional
    cmap : None, str, or array, optional
    sampling_frequency : float, optional
    time : ndarray, shape (n_time,), optional

    Returns
    -------
    ax : matplotlib axis
    im : scatter plot handle

    '''
    ax = ax or plt.gca()
    AVG_PLACE_FIELD_SIZE = 25
    n_colors = int(np.ceil(np.ptp(place_bin_centers) / AVG_PLACE_FIELD_SIZE))
    cmap = cmap or ListedColormap(sns.color_palette('hls', n_colors))

    n_time, n_neurons = spikes.shape
    if time is None:
        time = np.arange(n_time) / sampling_frequency

    cmap = plt.get_cmap(cmap)
    neuron_to_place_bin = np.argmax(place_field_firing_rates, axis=1)
    ordered_place_field_to_neuron = np.argsort(neuron_to_place_bin)
    neuron_to_ordered_place_field = np.argsort(ordered_place_field_to_neuron)

    time_ind, neuron_ind = np.nonzero(spikes)
    im = ax.scatter(time[time_ind], neuron_to_ordered_place_field[neuron_ind],
                    c=place_bin_centers[neuron_to_place_bin[neuron_ind]],
                    cmap=cmap, vmin=np.floor(place_bin_centers.min()),
                    vmax=np.ceil(place_bin_centers.max()))
    plt.colorbar(im, ax=ax, label='position')

    ax.set_xlim(time[[0, -1]])
    ax.set_xlabel('Time')
    ax.set_ylabel('Neurons')
    ax.set_yticks(np.arange(n_neurons))
    ax.set_yticklabels(ordered_place_field_to_neuron + 1)
    ax.set_ylim((-0.25, n_neurons - 1 + 0.25))
    sns.despine()

    return ax, im


def plot_replay_position_by_spikes(
    spikes, place_field_firing_rates, place_bin_centers,
        ax=None, cmap=None, sampling_frequency=1):

    ax = ax or plt.gca()
    cmap = cmap or 'hsv'

    n_time, n_neurons = spikes.shape
    n_place_bins = place_bin_centers.size
    time = np.arange(n_time) / sampling_frequency

    cmap = plt.get_cmap(cmap)
    place_colors = cmap(np.linspace(0.0, 1.0, n_place_bins))
    neuron_to_place_bin = np.argmax(place_field_firing_rates, axis=1)

    time_ind, neuron_ind = np.nonzero(spikes)

    ax.scatter(time[time_ind],
               place_bin_centers[neuron_to_place_bin[neuron_ind]],
               c=place_colors[neuron_to_place_bin[neuron_ind]])

    ax.set_xlim(time[[0, -1]])
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.set_ylim(place_bin_centers[[0, -1]])
    sns.despine()

    return ax


def plot_detector_posteriors(replay_densities, col_wrap=5):
    ds = replay_densities.detector_posterior
    ds = ds.assign_coords(time=ds.time / np.timedelta64(1, 's'))

    n_plots = len(ds.replay_id)
    n_rows = int(np.ceil(n_plots / col_wrap))

    fig, axes = plt.subplots(n_rows, col_wrap,
                             figsize=(col_wrap * 3, n_rows * 3),
                             sharey=True, constrained_layout=True,
                             squeeze=False)
    for ax, id in zip(axes.flat, ds.replay_id.values):
        ds.sel(replay_id=id).dropna('time').plot(
            x='time', y='position', ax=ax, vmin=0, robust=True,
            add_colorbar=False)
        ax.set_title(id, fontsize=12)
    if n_plots < axes.size:
        for ax in axes.flat[n_plots:]:
            ax.axis('off')


def plot_replay_with_data(replay_number, data, replay_info, replay_detector,
                          spikes_detector_results,
                          lfp_power_detector_results,
                          multiunit_detector_results,
                          epoch_key,
                          sampling_frequency=1500,
                          offset=pd.Timedelta(0.250, 's'),
                          position_metric='linear_distance',
                          speed_metric='linear_speed',
                          is_relative_time=False):

    start_time = replay_info.loc[replay_number].start_time - offset
    end_time = replay_info.loc[replay_number].end_time + offset

    is_ripple = data['is_ripple'].loc[start_time:end_time].squeeze()
    is_multiunit_high_synchrony = (data['is_multiunit_high_synchrony']
                                   .loc[start_time:end_time].squeeze())
    position_info = data['position_info'].loc[start_time:end_time]
    spikes = data['spikes'].loc[start_time:end_time]
    multiunit_firing_rate = (
        data['multiunit_firing_rate'].loc[start_time:end_time].squeeze())
    place_fields = (replay_detector
                    ._spiking_likelihood
                    .keywords['place_conditional_intensity'].T
                    * sampling_frequency)
    lfps = data['lfps'].loc[start_time:end_time]
    lfps /= np.ptp(lfps, axis=0)
    ripple_band_lfps = data['ripple_band_lfps'].loc[start_time:end_time]
    ripple_band_lfps /= np.ptp(data['ripple_band_lfps'], axis=0)
    mean_power = data['power'].mean(axis=0).values
    power = data['power'].loc[start_time:end_time].dropna() / mean_power
    place_bin_centers = replay_detector.place_bin_centers

    time = position_info.index.total_seconds()
    power_time = power.index.total_seconds()

    if is_relative_time:
        time -= start_time.total_seconds()
        power_time -= start_time.total_seconds()

    spike_results = (spikes_detector_results
                     .sel(time=slice(start_time, end_time), state='Replay')
                     .assign_coords(
                         time=lambda ds: ds.time / np.timedelta64(1, 's')))
    lfp_results = (lfp_power_detector_results
                   .sel(time=slice(start_time, end_time), state='Replay')
                   .assign_coords(
                       time=lambda ds: ds.time / np.timedelta64(1, 's')))
    multiunit_results = (multiunit_detector_results
                         .sel(time=slice(start_time, end_time), state='Replay')
                         .assign_coords(
                             time=lambda ds: ds.time / np.timedelta64(1, 's')))
    fig, axes = plt.subplots(8, 1, figsize=(12, 16),
                             constrained_layout=True, sharex=True)

    axes[0].plot(time, is_ripple,
                 label='Ad-hoc ripple', linewidth=3)
    axes[0].plot(time, is_multiunit_high_synchrony,
                 label='Ad-hoc multiunit HSE', linewidth=3)
    lfp_results.replay_probability.plot(x='time', label='lfp_power',
                                        ax=axes[0], linewidth=3)
    spike_results.replay_probability.plot(x='time', label='spikes',
                                          ax=axes[0], linewidth=3)
    multiunit_results.replay_probability.plot(x='time', label='multiunit',
                                              ax=axes[0], linewidth=3)

    axes[0].axhline(0.8, linestyle='--', color='black')
    axes[0].set_title('')
    axes[0].legend()

    spike_results.posterior.plot(x='time', y='position',
                                 vmin=0.0, robust=True, ax=axes[1])
    axes[1].plot(position_info.index.total_seconds(),
                 position_info[position_metric].values,
                 linewidth=3, linestyle='--', color='white')
    axes[1].set_title('')

    multiunit_results.posterior.plot(x='time', y='position',
                                     vmin=0.0, robust=True, ax=axes[2])
    axes[2].plot(position_info.index.total_seconds(),
                 position_info[position_metric].values,
                 linewidth=3, linestyle='--', color='white')
    axes[2].set_title('')

    plot_replay_spiking_ordered_by_place_fields(
        spikes.values, place_fields, place_bin_centers, ax=axes[3],
        time=time)

    axes[4].plot(time, multiunit_firing_rate.values)
    axes[4].set_ylabel('Multiunit firing rate (spikes / s)')

    for lfp_ind, lfp in enumerate(lfps.values.T):
        axes[5].plot(time, lfp + lfp_ind + 1)
    axes[5].set_ylabel('LFP')

    for lfp_ind, lfp in enumerate(ripple_band_lfps.values.T):
        axes[6].plot(time, lfp + lfp_ind + 1)
    axes[6].set_ylabel('Ripple\nBandpassed LFP')

    axes[7].plot(time,
                 position_info[speed_metric].values,
                 linewidth=3)
    axes[7].axhline(4, color='black', linestyle='--')
    axes[7].set_ylabel('Speed (m/s)')

    animal, day, epoch = epoch_key
    plt.suptitle(
        f'replay_number = {animal}_{day:02d}_{epoch:02d}_{replay_number:03d}',
        fontsize=14, y=1.01)
    axes[-1].set_xlabel('Time (s)')

    return fig, axes
