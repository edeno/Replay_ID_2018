import itertools
import logging
import sys
from os.path import join

import pandas as pd
import xarray as xr

from src.parameters import PROCESSED_DATA_DIR, USE_LIKELIHOODS

COLUMNS = ['animal', 'day', 'epoch',
           'replay_type', 'replay_motion_type', 'replay_movement_distance',
           'credible_interval_size', 'data_source', 'duration',
           'pct_unique_spiking', 'pct_total_spikes']

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level='INFO', format=FORMAT)


def _preprocess(ds):
    return ds[COLUMNS]


def main():
    logging.info('Gathering replay info...')
    filenames = join(PROCESSED_DATA_DIR, '*.nc')
    replay_info = pd.concat(
        [xr.open_mfdataset(
            filenames, group=f'{name}/replay_info', autoclose=True,
            preprocess=_preprocess, parallel=True).to_dataframe()
         for name in USE_LIKELIHOODS])

    logging.info(replay_info.info(verbose=False, memory_usage='deep'))
    replay_info_path = join(PROCESSED_DATA_DIR, 'replay_info.csv')
    replay_info.to_csv(replay_info_path, mode='w')

    logging.info('Gathering overlap info...')
    names = list(USE_LIKELIHOODS)
    combination = itertools.combinations(names, 2)
    overlap_info = []

    for name1, name2 in combination:
        logging.info('...{name1} vs {name2}')
        overlap_info.append(xr.open_mfdataset(
            filenames, group=f'/overlap/{name1}/{name2}', autoclose=True,
            parallel=True).to_dataframe())

    overlap_info = pd.concat(overlap_info)

    logging.info(overlap_info.info(verbose=False, memory_usage='deep'))
    overlap_info_path = join(PROCESSED_DATA_DIR, 'overlap_info.csv')
    overlap_info.to_csv(overlap_info_path, mode='w')

    logging.info('Done...')


if __name__ == '__main__':
    sys.exit(main())
