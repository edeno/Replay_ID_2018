import os

from loren_frank_data_processing import save_xarray
from src.parameters import PROCESSED_DATA_DIR


def save_replay_data(data_source, epoch_key, replay_info, is_replay):
    epoch_identifier = _get_epoch_identifier(epoch_key)

    replay_info_filename = os.path.join(
        PROCESSED_DATA_DIR,
        f'{epoch_identifier}_{data_source}_replay_info.csv')
    replay_info.to_csv(replay_info_filename)

    is_replay_filename = os.path.join(
        PROCESSED_DATA_DIR, f'{epoch_identifier}_{data_source}_is_replay.csv')
    is_replay.to_csv(is_replay_filename)


def save_detector_parameters(epoch_key, replay_detector):
    epoch_identifier = _get_epoch_identifier(epoch_key)
    file_name = f'{epoch_identifier}_replay_detector.gz'
    replay_detector.save_model(os.path.join(PROCESSED_DATA_DIR, file_name))


def save_overlap_info(overlap_info, epoch_key, data_source1, data_source2):
    epoch_identifier = _get_epoch_identifier(epoch_key)
    overlap_filename = (
        f'{epoch_identifier}_{data_source1}_{data_source2}_overlap_info.csv')
    overlap_info.to_csv(overlap_filename)


def save_non_overlap_info(
        non_overlap_info, epoch_key, data_source1, data_source2):
    epoch_identifier = _get_epoch_identifier(epoch_key)
    non_overlap_filename = (
        f'{epoch_identifier}_{data_source1}_{data_source2}'
        '_non_overlap_info.csv')
    non_overlap_info.to_csv(non_overlap_filename)


def save_power(power, epoch_key, data_source):
    group = f'{data_source}/power'
    save_xarray(PROCESSED_DATA_DIR, epoch_key, power, group)


def _get_epoch_identifier(epoch_key):
    animal, day, epoch = epoch_key
    return f'{animal}_{day:02d}_{epoch:02d}'
