import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, LogNorm

from .parameters import USE_LIKELIHOODS

HUE_ORDER = {
    'replay_motion_type': ['Away', 'Neither', 'Towards'],
    'replay_type': ['Inbound-Forward', 'Inbound-Reverse',
                    'Outbound-Forward', 'Outbound-Reverse', 'Unclassified']
}

_DEFAULT_DATA_TYPES = ['spikes', 'multiunit', 'lfps', 'bandpassed_lfps',
                       'speed']
_DEFAULT_RESULT_TYPES = ['spikes', 'ripple_power', 'clusterless',
                         'ad_hoc_ripple', 'ad_hoc_multiunit']
RESULT_COLORS_MAP = {
    'spikes': '#ff7f0e',
    'ripple_power': '#2ca02c',
    'clusterless': '#d62728',
    'ad_hoc_ripple': '#1f77b4',
    'ad_hoc_multiunit': '#17becf',
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


def plot_behavior(position_info, position_metric='linear_position2',
                  speed_metric='speed'):

    time = np.asarray(position_info.index.total_seconds())
    is_inbound = position_info.task == 'Inbound'
    is_outbound = position_info.task == 'Outbound'
    position = position_info[position_metric].values
    is_correct = position_info.is_correct.values
    speed = position_info[speed_metric].values

    p_min, p_max = np.nanmin(position), np.nanmax(position)
    s_min, s_max = np.nanmin(speed), np.nanmax(speed)

    fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharex=True,
                             constrained_layout=True)

    axes[0].fill_between(time, is_correct * p_min,
                         is_correct * p_max, color='#7fc97f',
                         where=is_correct,
                         alpha=0.25, label='Correct')
    axes[0].scatter(time, position, color='lightgrey')
    axes[0].scatter(time[is_inbound], position[is_inbound], label='Inbound')
    axes[0].scatter(time[is_outbound], position[is_outbound], label='Outbound')
    axes[0].set_ylabel(f'{position_metric} (cm)')
    axes[0].set_ylim((p_min, p_max))
    axes[0].legend()

    axes[1].fill_between(time, is_correct * s_min,
                         is_correct * s_max, color='#7fc97f',
                         where=is_correct, alpha=0.25)
    axes[1].plot(time, speed, color='#7570b3', linewidth=1)
    axes[1].axhline(4, linestyle='--', color='black')
    axes[1].set_ylabel(f'{speed_metric} (cm / s)')
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
                    vmax=np.ceil(place_bin_centers.max()), s=25)
    plt.colorbar(im, ax=ax, label='position')

    ax.set_xlim(time[[0, -1]])
    ax.set_xlabel('Time')
    ax.set_ylabel('Neurons')
    ax.set_yticks(np.arange(n_neurons))
    ax.set_yticklabels(ordered_place_field_to_neuron + 1)
    ax.set_ylim((-0.25, n_neurons + 1.00))
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


def plot_replay_with_data(replay_number, data, replay_info, epoch_key=None,
                          replay_detector=None,
                          spikes_detector_results=None,
                          lfp_power_detector_results=None,
                          multiunit_detector_results=None,
                          sampling_frequency=1500,
                          offset=pd.Timedelta(0.250, 's'),
                          position_metric='linear_distance',
                          speed_metric='linear_speed',
                          is_relative_time=False,
                          show_data_types=None, show_result_types=None):

    if show_data_types is None:
        show_data_types = _DEFAULT_DATA_TYPES
    if show_result_types is None:
        show_result_types = _DEFAULT_RESULT_TYPES

    start_time = replay_info.loc[replay_number].start_time - offset
    end_time = replay_info.loc[replay_number].end_time + offset
    position_info = data['position_info'].loc[start_time:end_time]
    time = position_info.index.total_seconds()
    if is_relative_time:
        time -= start_time.total_seconds()

    if 'spikes' in show_result_types:
        spike_results = (spikes_detector_results
                         .sel(time=slice(start_time, end_time), state='Replay')
                         .assign_coords(
                             time=lambda ds: ds.time / np.timedelta64(1, 's')))
    else:
        spike_results = None

    if 'ripple_power' in show_result_types:
        lfp_results = (lfp_power_detector_results
                       .sel(time=slice(start_time, end_time), state='Replay')
                       .assign_coords(
                           time=lambda ds: ds.time / np.timedelta64(1, 's')))
    else:
        lfp_results = None

    if 'clusterless' in show_result_types:
        multiunit_results = (
            multiunit_detector_results
            .sel(time=slice(start_time, end_time), state='Replay')
            .assign_coords(time=lambda ds: ds.time / np.timedelta64(1, 's')))
    else:
        multiunit_results = None

    n_plots = len(show_data_types)
    if len(show_result_types) > 0:
        n_plots += 1
        if 'spikes' in show_result_types:
            n_plots += 1
        if 'clusterless' in show_result_types:
            n_plots += 1

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.25 * n_plots),
                             constrained_layout=True, sharex=True)
    plt.ticklabel_format(useOffset=False)

    if 'ad_hoc_ripple' in show_result_types:
        is_ripple = data['is_ripple'].loc[start_time:end_time].squeeze()
        axes[0].plot(time, is_ripple, label='Ad-hoc ripple', linewidth=3,
                     color=RESULT_COLORS_MAP['ad_hoc_ripple'])
    if 'ad_hoc_multiunit' in show_result_types:
        is_multiunit_high_synchrony = (data['is_multiunit_high_synchrony']
                                       .loc[start_time:end_time].squeeze())
        axes[0].plot(time, is_multiunit_high_synchrony,
                     label='Ad-hoc multiunit HSE', linewidth=3,
                     color=RESULT_COLORS_MAP['ad_hoc_multiunit'])
    if 'ripple_power' in show_result_types:
        lfp_results.replay_probability.plot(
            x='time', label='ripple power', ax=axes[0], linewidth=3,
            color=RESULT_COLORS_MAP['ripple_power'])
    if 'spikes' in show_result_types:
        spike_results.replay_probability.plot(
            x='time', label='spikes', ax=axes[0], linewidth=3,
            color=RESULT_COLORS_MAP['spikes'])
    if 'clusterless' in show_result_types:
        multiunit_results.replay_probability.plot(
            x='time', label='clusterless', ax=axes[0], linewidth=3,
            color=RESULT_COLORS_MAP['clusterless'])

    if len(show_result_types) > 0:
        axes[0].set_title('')
        axes[0].set_ylabel('Replay\nProbability')
        axes[0].tick_params(length=0)
        axes[0].set_xlabel('')
        axes[0].legend(loc='upper right', frameon=False, fontsize=12)

    if 'spikes' in show_result_types:
        spike_results.acausal_posterior.plot(x='time', y='position',
                                             vmin=0.0, vmax=0.1, ax=axes[1])
        axes[1].plot(position_info.index.total_seconds(),
                     position_info[position_metric].values,
                     linewidth=3, linestyle='--', color='white')
        axes[1].set_title('')
        max_df = (data['position_info']
                  .groupby('arm_name')[position_metric].max())
        min_time = np.asarray(spike_results.time.min())
        for arm_name, max_position in max_df.iteritems():
            axes[1].axhline(max_position, color='grey',
                            linestyle='-', linewidth=1)
            axes[1].text(min_time, max_position - 5, arm_name, color='white',
                         fontsize=11, verticalalignment='top')
        min_df = (data['position_info']
                  .groupby('arm_name')[position_metric].min())
        for arm_name, min_position in min_df.iteritems():
            axes[1].axhline(min_position, color='grey',
                            linestyle='-', linewidth=1)
        axes[1].tick_params(length=0)
        axes[1].set_xlabel('')

    if 'clusterless' in show_result_types:
        multiunit_results.acausal_posterior.plot(x='time', y='position',
                                                 vmin=0.0, robust=True, ax=axes[2])
        axes[2].plot(position_info.index.total_seconds(),
                     position_info[position_metric].values,
                     linewidth=3, linestyle='--', color='white')
        axes[2].set_title('')

    ax_ind = 0
    if len(show_result_types) > 0:
        ax_ind += 1
    if 'spikes' in show_result_types:
        ax_ind += 1
    if 'clusterless' in show_result_types:
        ax_ind += 1

    if 'spikes' in show_data_types:
        place_bin_centers = replay_detector.place_bin_centers_.squeeze(axis=-1)
        place_bin_edges = replay_detector.place_bin_edges_.squeeze(axis=-1)
        is_track = (np.histogram(data['position_info'][position_metric],
                                 bins=place_bin_edges)[0] > 0)
        place_fields = (replay_detector
                        ._spiking_likelihood
                        .keywords['place_conditional_intensity'].T
                        * sampling_frequency)
        place_fields[:, ~is_track] = 0.0

        spikes = data['spikes'].loc[start_time:end_time]
        ax = axes if n_plots == 1 else axes[ax_ind]
        plot_replay_spiking_ordered_by_place_fields(
            spikes.values, place_fields, place_bin_centers, ax=ax,
            time=time)
        ax.set_yticks([])
        ax.set_yticklabels([], [])
        ax.tick_params(length=0)
        ax.set_xlabel('')
        ax_ind += 1

    if 'multiunit' in show_data_types:
        multiunit_firing_rate = (
            data['multiunit_firing_rate'].loc[start_time:end_time].squeeze())
        ax = axes if n_plots == 1 else axes[ax_ind]
        ax.plot(time, multiunit_firing_rate.values)
        ax.set_ylabel('Multiunit firing rate (spikes / s)')
        ax_ind += 1

    if 'lfps' in show_data_types:
        lfps = data['lfps'].loc[start_time:end_time]
        lfps /= np.ptp(np.asarray(lfps), axis=0)
        ax = axes if n_plots == 1 else axes[ax_ind]
        for lfp_ind, lfp in enumerate(lfps.values.T):
            ax.plot(time, lfp + lfp_ind + 1)
        ax.set_ylabel('LFPs')
        ax.set_yticks([])
        ax.set_yticklabels([], [])
        ax_ind += 1

    if 'bandpassed_lfps' in show_data_types:
        ripple_band_lfps = np.asarray(
            data['ripple_filtered_lfps'].loc[start_time:end_time])
        ripple_band_lfps /= np.ptp(
            np.asarray(data['ripple_filtered_lfps']), axis=0)
        ax = axes if n_plots == 1 else axes[ax_ind]
        for lfp_ind, lfp in enumerate(ripple_band_lfps.T):
            ax.plot(time, lfp + lfp_ind + 1)
        ax.set_ylabel('Ripple\nBandpassed LFP')
        ax_ind += 1

    if 'speed' in show_data_types:
        ax = axes if n_plots == 1 else axes[ax_ind]
        ax.plot(time, position_info[speed_metric].values, linewidth=3)
        ax.axhline(4, color='black', linestyle='--')
        ax.set_ylabel('Speed (cm/s)')

    if epoch_key is not None:
        animal, day, epoch = epoch_key
        plt.suptitle(
            ('replay_number = '
             f'{animal}_{day:02d}_{epoch:02d}_{replay_number:03d}'),
            fontsize=14, y=1.05)

    ax = axes if n_plots == 1 else axes[-1]
    plt.xlim((time.min(), time.max()))
    ax.set_xlabel('Time (s)')
    sns.despine()

    return fig, axes


def plot_power_change(power, data_source1, data_source2,
                      frequency=None, vmin=0.5, vmax=2):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    (power[data_source1].mean('tetrode')
     .sel(frequency=frequency)
     .plot(x='time', y='frequency', ax=axes[0]))
    axes[0].set_title(data_source1)

    (power[data_source2].mean('tetrode')
     .sel(frequency=frequency)
     .plot(x='time', y='frequency', ax=axes[1]))
    axes[1].set_title(data_source2)

    ((power[data_source1] / power[data_source2]).mean('tetrode')
     .sel(frequency=frequency)
     .plot(x='time', y='frequency', ax=axes[2],
           norm=LogNorm(vmin=vmin, vmax=vmax), cmap='RdBu_r',
           vmin=vmin, vmax=vmax, center=0))
    axes[2].set_title(f'{data_source1} vs. {data_source2}')

    return fig, axes


def plot_linearized_place_fields(replay_detector, position_info,
                                 position_metric='linear_position2',
                                 sort_order=None):
    place_bin_centers = replay_detector.place_bin_centers_.squeeze(axis=-1)
    place_fields = (replay_detector
                    ._spiking_likelihood
                    .keywords['place_conditional_intensity'].T)
    if sort_order is None:
        sort_order = np.argsort(place_fields.argmax(axis=1))
    place_fields = place_fields[sort_order]
    n_neurons = place_fields.shape[0]
    fig, axes = plt.subplots(n_neurons, 1, constrained_layout=True,
                             figsize=(10, n_neurons / 3), sharex=True)

    for ax, place_field in zip(axes, place_fields):
        ax.fill_between(place_bin_centers, place_field)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.tick_params(length=0)
        max_df = (position_info
                  .groupby('arm_name')[position_metric].max())
        for arm_name, max_position in max_df.iteritems():
            ax.axvline(max_position, color='grey',
                       linestyle='-', linewidth=1)
        min_df = (position_info
                  .groupby('arm_name')[position_metric].min())
        for arm_name, min_position in min_df.iteritems():
            ax.axvline(min_position, color='grey',
                       linestyle='-', linewidth=1)

    sns.despine()
    plt.xlim((place_bin_centers.min(), place_bin_centers.max()))

    return axes
