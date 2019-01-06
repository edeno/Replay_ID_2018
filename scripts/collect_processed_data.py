import itertools
import logging
import sys
from os.path import join

import pandas as pd
import xarray as xr

from src.parameters import PROCESSED_DATA_DIR, USE_LIKELIHOODS

COLUMNS = ['animal', 'day', 'epoch',
           'replay_type', 'replay_motion_type', 'replay_movement_distance',
           'data_source', 'duration',
           'pct_unique_spiking', 'pct_total_spikes', 'avg_replay_speed']

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level='INFO', format=FORMAT)


def _preprocess(ds):
    return ds[COLUMNS]


def collect_data(use_smoother=True):
    decoder_type = 'smoother' if use_smoother else 'filter'
    logging.info('Gathering replay info...')
    filenames = join(PROCESSED_DATA_DIR, '*.nc')
    replay_info = pd.concat(
        [xr.open_mfdataset(
            filenames, group=f'{decoder_type}/{data_source}/replay_info',
            autoclose=True, preprocess=_preprocess,
            parallel=True).to_dataframe()
         for data_source in USE_LIKELIHOODS])

    logging.info(replay_info.info(verbose=False, memory_usage='deep'))
    replay_info_path = join(
        PROCESSED_DATA_DIR, f'{decoder_type}_replay_info.csv')
    replay_info.to_csv(replay_info_path, mode='w')

    logging.info('Gathering overlap info...')
    names = list(USE_LIKELIHOODS)
    combination = itertools.combinations(names, 2)
    overlap_info = []

    for data_source1, data_source2 in combination:
        logging.info(f'...{data_source1} vs {data_source2}')
        group = f'{decoder_type}/overlap/{data_source1}/{data_source2}'
        overlap_info.append(xr.open_mfdataset(
            filenames, group=group, autoclose=True,
            parallel=True).to_dataframe())

    overlap_info = pd.concat(overlap_info)

    logging.info(overlap_info.info(verbose=False, memory_usage='deep'))
    overlap_info_path = join(
        PROCESSED_DATA_DIR, f'{decoder_type}_overlap_info.csv')
    overlap_info.to_csv(overlap_info_path, mode='w')

    logging.info('Done...')


def main():
    collect_data(use_smoother=True)
    collect_data(use_smoother=False)


if __name__ == '__main__':
    sys.exit(main())
