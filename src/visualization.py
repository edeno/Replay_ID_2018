import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .parameters import USE_LIKELIHOODS

HUE_ORDER = {
    'replay_motion_type': ['Away', 'Neither', 'Towards'],
    'replay_type': ['Inbound-Forward', 'Inbound-Reverse',
                    'Outbound-Forward', 'Outbound-Reverse', 'Unclassified']
}


def plot_data_source_counts(replay_info, kind='bar'):
    '''Compare number of replays by data source.'''
    df = (replay_info
          .groupby(['animal', 'day', 'epoch'])['data_source']
          .value_counts()
          .rename('counts_by_epoch')
          .reset_index())

    g = sns.catplot(x='data_source', y='counts_by_epoch', data=df, kind=kind,
                    col='animal', order=USE_LIKELIHOODS.keys(), aspect=2,
                    col_wrap=2)

    return g


def plot_proportion_events_by_data_source(replay_info, covariate,
                                          kind='bar', **plot_kwargs):
    '''covariate = {'replay_type', 'replay_motion_type'}'''
    group = ['data_source', 'animal', 'day', 'epoch']
    df = (replay_info.groupby(group)[covariate]
          .value_counts(normalize=True)
          .rename('proportion')
          .reset_index())
    g = sns.catplot(x=covariate, y='proportion', data=df, col='animal',
                    hue='data_source', kind=kind, aspect=2, col_wrap=1,
                    hue_order=USE_LIKELIHOODS.keys(),
                    order=HUE_ORDER[covariate],
                    **plot_kwargs)
    return g


def plot_continuous_by_data_source(replay_info, covariate, kind='bar'):
    '''covariate = {'replay_movement_distance', 'credible_interval_size',
                    'duration'}'''
    g = sns.catplot(x='data_source', y=covariate, order=USE_LIKELIHOODS.keys(),
                    data=replay_info, aspect=2, kind=kind, col='animal',
                    col_wrap=2)
    return g


def compare_jaccard_similarity_of_replays(overlap_info, replay_info):
    '''Compare the number of replays that overlap for at least one time point
       vs. the union of the total number of replays for each data source.'''
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
                     vmin=0.0, vmax=1.0)
    return ax


def compare_time_difference_of_overlapping_replays(
        overlap_info, time_difference):
    '''Of the replays that do overlap, compare the start or end time'''
    g = sns.FacetGrid(data=overlap_info,  row='data_source1',
                      col='data_source2', height=5, aspect=1,
                      row_order=USE_LIKELIHOODS.keys(),
                      col_order=USE_LIKELIHOODS.keys(), sharey=False)
    kde_kws = {'color': 'red', 'lw': 3, 'shade': True}
    g.map(plt.axvline, x=0, color='k', linestyle='--', linewidth=3, zorder=100)
    g.map(sns.distplot, time_difference, hist=False, rug=False, kde_kws=kde_kws
          ).set_titles('{row_name} - {col_name}', size=15)
    return g


def compare_similarity_of_overlapping_replays(overlap_info):
    '''Of the replays that do overlap, compare how much they overlap in time'''
    g = sns.FacetGrid(data=overlap_info,  row='data_source1',
                      col='data_source2', height=5, aspect=1,
                      row_order=USE_LIKELIHOODS.keys(),
                      col_order=USE_LIKELIHOODS.keys())
    kde_kws = {'color': 'red', 'lw': 3, 'clip': (0, 1), 'shade': True}
    g.map(sns.distplot, 'jaccard_similarity', hist=False, rug=False,
          kde_kws=kde_kws).set_titles('{row_name}, {col_name}', size=15)
    g.set(xlim=(0, 1))

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
                         alpha=0.25)
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
    cmap = cmap or 'hsv'

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
        ax=None, cmap=None, sampling_frequency=1):

    ax = ax or plt.gca()
    cmap = cmap or 'hsv'

    n_time, n_neurons = spikes.shape
    n_place_bins = place_bin_centers.size
    time = np.arange(n_time) / sampling_frequency

    cmap = plt.get_cmap(cmap)
    place_colors = cmap(np.linspace(0.0, 1.0, n_place_bins))
    neuron_to_place_bin = np.argmax(place_field_firing_rates, axis=1)
    ordered_place_field_to_neuron = np.argsort(neuron_to_place_bin)
    neuron_to_ordered_place_field = np.argsort(ordered_place_field_to_neuron)

    time_ind, neuron_ind = np.nonzero(spikes)

    ax.scatter(time[time_ind], neuron_to_ordered_place_field[neuron_ind],
               c=place_colors[neuron_to_place_bin[neuron_ind]])

    ax.set_xlim(time[[0, -1]])
    ax.set_xlabel('Time')
    ax.set_ylabel('Neurons')
    ax.set_yticks(np.arange(n_neurons))
    ax.set_yticklabels(ordered_place_field_to_neuron)
    sns.despine()

    return ax


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
