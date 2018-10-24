import itertools
import logging
import sys
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from src.parameters import FIGURE_DIR, PROCESSED_DATA_DIR, USE_LIKELIHOODS
from src.visualization import (compare_jaccard_similarity_of_replays,
                               compare_similarity_of_overlapping_replays,
                               compare_time_difference_of_overlapping_replays,
                               plot_continuous_by_data_source,
                               plot_data_source_counts,
                               plot_proportion_events_by_data_source)

COLUMNS = ['animal', 'day', 'epoch',
           'replay_type', 'replay_motion_type', 'replay_movement_distance',
           'credible_interval_size', 'data_source', 'duration']

logging.basicConfig(level='INFO')


def _add_name(figure_name, name=None):
    if name is None:
        figure_name = join(FIGURE_DIR, f'{figure_name}.png')
    else:
        figure_name = join(FIGURE_DIR, f'{figure_name}_{name}.png')
    return figure_name


def save_overlap_figures(overlap_info, replay_info, name=None):
    compare_jaccard_similarity_of_replays(overlap_info, replay_info)
    figure_name = 'jaccard_similarity_of_replays'
    figure_name = _add_name(figure_name, name)
    plt.savefig(figure_name)

    for time_difference in ['start_time_difference', 'end_time_difference']:
        compare_time_difference_of_overlapping_replays(
            overlap_info, time_difference)
        figure_name = f'{time_difference}_of_overlapping_replays'
        figure_name = _add_name(figure_name, name)
        plt.savefig(figure_name)

    compare_similarity_of_overlapping_replays(overlap_info)
    figure_name = 'similarity_of_overlapping_replays'
    figure_name = _add_name(figure_name, name)
    plt.savefig(figure_name)


def save_replay_info_figures(replay_info, name=None):
    # Compare number of events
    plot_data_source_counts(replay_info)
    figure_name = 'n_events'
    figure_name = _add_name(figure_name, name)
    plt.savefig(figure_name)

    plot_data_source_counts(replay_info, col='animal', col_wrap=2)
    figure_name = 'n_events_by_animal'
    figure_name = _add_name(figure_name, name)
    plt.savefig(figure_name)

    # Compare replay types
    for covariate in ['replay_type', 'replay_motion_type']:
        plot_proportion_events_by_data_source(replay_info, covariate)
        figure_name = f'prop_{covariate}_by_data_source'
        figure_name = _add_name(figure_name, name)
        plt.savefig(figure_name)

        plot_proportion_events_by_data_source(replay_info, covariate,
                                              col='animal', col_wrap=1)
        figure_name = f'prop_{covariate}_by_data_source_by_animal'
        figure_name = _add_name(figure_name, name)
        plt.savefig(figure_name)

    # Compare how far the replay position moves/posterior confidence bounds
    continuous_covariates = [
        'replay_movement_distance', 'credible_interval_size',
        'duration', 'pct_unique_spiking', 'pct_total_spikes']
    for covariate in continuous_covariates:
        plot_continuous_by_data_source(replay_info, covariate)
        figure_name = f'{covariate}_by_data_source'
        figure_name = _add_name(figure_name, name)
        plt.savefig(figure_name)

        plot_continuous_by_data_source(replay_info, covariate,
                                       col='animal', col_wrap=2)
        figure_name = f'{covariate}_by_data_source_by_animal'
        figure_name = _add_name(figure_name, name)
        plt.savefig(figure_name)


def main():
    logging.info('Gathering replay info...')
    filenames = join(PROCESSED_DATA_DIR, '*.nc')
    replay_info = pd.concat(
        [xr.open_mfdataset(
            filenames, group=f'{name}/replay_info', autoclose=True
        ).to_dataframe().loc[:, COLUMNS]
            for name in USE_LIKELIHOODS])

    logging.info('Comparing all replay events...')
    save_replay_info_figures(replay_info)
    logging.info(replay_info.info(verbose=False, memory_usage='deep'))

    logging.info('Gathering overlap info...')
    names = list(USE_LIKELIHOODS)
    combination = itertools.combinations(names, 2)
    overlap_info = []

    for name1, name2 in combination:
        overlap_info.append(xr.open_mfdataset(
            filenames, group=f'/overlap/{name1}/{name2}', autoclose=True
        ).to_dataframe())

    overlap_info = pd.concat(overlap_info)

    logging.info('Comparing overlapping replay events...')
    save_overlap_figures(overlap_info, replay_info)

    overlap_grouper = overlap_info.groupby('animal')
    replay_grouper = replay_info.groupby('animal')
    for (animal_name, oi), (_, ri) in zip(overlap_grouper, replay_grouper):
        save_overlap_figures(oi, ri, name=animal_name)

    logging.info(overlap_info.info(verbose=False, memory_usage='deep'))

    logging.info('Comparing non-overlapping replay events...')
    grouper = overlap_info.groupby(['data_source1', 'data_source2'])
    for (name1, name2), df in grouper:
        is_not_overlap = ((
            ~replay_info.index.isin(df.replay_number2) &
            (replay_info.data_source == name1)) |
            (~replay_info.index.isin(df.replay_number1) &
             (replay_info.data_source == name2)))
        name = f'no_overlap_{name1}_vs_{name2}'
        save_replay_info_figures(replay_info[is_not_overlap], name)

    logging.info('Done...')


if __name__ == '__main__':
    sys.exit(main())
