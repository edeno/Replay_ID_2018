from os.path import join

from loren_frank_data_processing import save_xarray
from src.parameters import PROCESSED_DATA_DIR


def save_replay_data(data_source, epoch_key, replay_info, is_replay,
                     use_smoother):
    decoder_type = 'smoother' if use_smoother else 'filter'

    save_xarray(PROCESSED_DATA_DIR, epoch_key,
                replay_info.to_xarray(),
                f'{decoder_type}/{data_source}/replay_info')
    save_xarray(PROCESSED_DATA_DIR, epoch_key,
                is_replay.to_xarray(),
                f'{decoder_type}/{data_source}/is_replay')


def save_detector_parameters(epoch_key, replay_detector):
    animal, day, epoch = epoch_key
    file_name = f'{animal}_{day:02}_{epoch:02}_replay_detector.gz'
    replay_detector.save_model(join(PROCESSED_DATA_DIR, file_name))


def save_overlap(overlap, epoch_key, data_source1, data_source2, use_smoother):
    decoder_type = 'smoother' if use_smoother else 'filter'
    save_xarray(PROCESSED_DATA_DIR, epoch_key,
                overlap.reset_index().to_xarray(),
                f'{decoder_type}/overlap/{data_source1}/{data_source2}')


def save_power(power, epoch_key, use_smoother, data_source):
    decoder_type = 'smoother' if use_smoother else 'filter'
    group = f'{decoder_type}/{data_source}/power'
    save_xarray(PROCESSED_DATA_DIR, epoch_key, power, group)
