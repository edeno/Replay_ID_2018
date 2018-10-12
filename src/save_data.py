from os.path import join

from loren_frank_data_processing import save_xarray
from src.parameters import PROCESSED_DATA_DIR


def save_replay_data(name, epoch_key, replay_info, replay_densities,
                     is_replay):
    save_xarray(PROCESSED_DATA_DIR, epoch_key, replay_densities,
                f'{name}/replay_densities')
    save_xarray(PROCESSED_DATA_DIR, epoch_key,
                replay_info.reset_index().to_xarray(), f'{name}/replay_info')
    save_xarray(PROCESSED_DATA_DIR, epoch_key,
                is_replay.reset_index().to_xarray(), f'{name}/is_replay')


def save_ripple_data(epoch_key, data):
    save_xarray(PROCESSED_DATA_DIR, epoch_key,
                data['ripple_labels'].reset_index().to_xarray(),
                '/ripple_labels')
    save_xarray(PROCESSED_DATA_DIR, epoch_key,
                data['ripple_times'].reset_index().to_xarray(),
                '/ripple_times')


def save_detector_parameters(epoch_key, replay_detector):
    animal, day, epoch = epoch_key
    file_name = f'{animal}_{day:02}_{epoch:02}_replay_detector.gz'
    replay_detector.save_model(join(PROCESSED_DATA_DIR, file_name))


def save_overlap(overlap, epoch_key, name1, name2):
    save_xarray(PROCESSED_DATA_DIR, epoch_key,
                overlap.reset_index().to_xarray(), f'/overlap/{name1}/{name2}')
