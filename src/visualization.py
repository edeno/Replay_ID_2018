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
          .rename('counts by epoch')
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
                    data=replay_info, aspect=2, kind=kind, col='animal')
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
    g.map(sns.distplot, 'jaccard', hist=False, rug=False, kde_kws=kde_kws
          ).set_titles('{row_name}, {col_name}', size=15)
    g.set(xlim=(0, 1))

    return g
