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


def save_overlap_figures(overlap_info, replay_info, name=None):
    compare_jaccard_similarity_of_replays(overlap_info, replay_info)
    figure_name = 'jaccard_similarity_of_replays'
    if name is None:
        figure_name = join(FIGURE_DIR, f'{figure_name}.png')
    else:
        figure_name = join(FIGURE_DIR, f'{figure_name}_{name}.png')
    plt.savefig(figure_name)

    for time_difference in ['start_time_difference', 'end_time_difference']:
        compare_time_difference_of_overlapping_replays(
            overlap_info, time_difference)
        figure_name = f'{time_difference}_of_overlapping_replays'
        if name is None:
            figure_name = join(FIGURE_DIR, f'{figure_name}.png')
        else:
            figure_name = join(FIGURE_DIR, f'{figure_name}_{name}.png')
        plt.savefig(figure_name)

    compare_similarity_of_overlapping_replays(overlap_info)
    figure_name = 'similarity_of_overlapping_replays'
    if name is None:
        figure_name = join(FIGURE_DIR, f'{figure_name}.png')
    else:
        figure_name = join(FIGURE_DIR, f'{figure_name}_{name}.png')
    plt.savefig(figure_name)


def main():
    logging.info('Gathering replay info...')
    filenames = join(PROCESSED_DATA_DIR, '*.nc')
    replay_info = pd.concat(
        [xr.open_mfdataset(
            filenames, group=f'{name}/replay_info', autoclose=True
        ).to_dataframe().loc[:, COLUMNS]
            for name in USE_LIKELIHOODS])

    logging.info('Compare replay events...')
    # Compare number of events
    plot_data_source_counts(replay_info)
    plt.savefig(join(FIGURE_DIR, 'n_events.png'))

    # Compare replay types
    for covariate in ['replay_type', 'replay_motion_type']:
        plot_proportion_events_by_data_source(replay_info, covariate)
        plt.savefig(join(FIGURE_DIR, f'prop_{covariate}_by_data_source.png'))
    # Compare how far the replay position moves/posterior confidence bounds
    continuous_covariates = [
        'replay_movement_distance', 'credible_interval_size',
        'overlap_duration']
    for covariate in continuous_covariates:
        plot_continuous_by_data_source(replay_info, covariate)
        plt.savefig(join(FIGURE_DIR, f'{covariate}_by_data_source.png'))

    logging.info('Gathering overlap info...')
    names = list(USE_LIKELIHOODS.keys())
    combination = itertools.combinations(names, 2)
    overlap_info = []

    for name1, name2 in combination:
        overlap_info.append(xr.open_mfdataset(
            filenames, group=f'/overlap/{name1}/{name2}', autoclose=True
        ).to_dataframe())

    overlap_info = pd.concat(overlap_info)

    logging.info('Comparing overlap of replay events...')
    save_overlap_figures(overlap_info, replay_info)

    overlap_grouper = overlap_info.groupby('animal')
    replay_grouper = replay_info.groupby('animal')
    for (animal_name, oi), (_, ri) in zip(overlap_grouper, replay_grouper):
        save_overlap_figures(oi, ri, name=animal_name)


if __name__ == '__main__':
    sys.exit(main())
