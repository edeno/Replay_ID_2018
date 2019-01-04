import itertools
import logging
import sys
from argparse import ArgumentParser
from os.path import join

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from replay_identification import ReplayDetector
from src.load_data import load_data
from src.parameters import (ANIMALS, BRAIN_AREAS, FIGURE_DIR,
                            SAMPLING_FREQUENCY, USE_LIKELIHOODS)
from src.summarize_replay import get_replay_times
from src.visualization import plot_replay_with_data
from src.misc import WhitenedKDE

logging.basicConfig(level=logging.INFO)


def main(epoch_key, speed_metric='linear_speed',
         position_metric='linear_distance', use_smoother=False):
    data_types = set(itertools.chain(*USE_LIKELIHOODS.values()))
    data = load_data(epoch_key, ANIMALS, SAMPLING_FREQUENCY, data_types,
                     BRAIN_AREAS, speed_metric)

    replay_detector = ReplayDetector(
        multiunit_density_model=WhitenedKDE,
        multiunit_model_kwargs=dict(bandwidth=1.10, kernel='epanechnikov'))
    replay_detector.fit(
        is_replay=data['is_ripple'], speed=data['position_info'].linear_speed,
        position=data['position_info'][position_metric],
        lfp_power=data['power'], spikes=data['spikes'],
        multiunit=data['multiunit'])

    lfp_power_detector_results = replay_detector.predict(
        speed=data['position_info'].linear_speed,
        position=data['position_info'][position_metric],
        lfp_power=data['power'],
        spikes=data['spikes'], multiunit=data['multiunit'],
        time=data['position_info'].index,
        use_likelihoods=['lfp_power'],
        use_smoother=use_smoother)

    spikes_detector_results = replay_detector.predict(
        speed=data['position_info'].linear_speed,
        position=data['position_info'][position_metric],
        lfp_power=data['power'],
        spikes=data['spikes'], multiunit=data['multiunit'],
        time=data['position_info'].index,
        use_likelihoods=['spikes'],
        use_smoother=use_smoother)

    multiunit_detector_results = replay_detector.predict(
        speed=data['position_info'].linear_speed,
        position=data['position_info'][position_metric],
        lfp_power=data['power'],
        spikes=data['spikes'], multiunit=data['multiunit'],
        time=data['position_info'].index,
        use_likelihoods=['multiunit'],
        use_smoother=use_smoother)

    spikes_replay_info, spikes_is_replay = get_replay_times(
        spikes_detector_results)
    lfp_power_replay_info, lfp_power_is_replay = get_replay_times(
        lfp_power_detector_results)
    multiunit_replay_info, multiunit_is_replay = get_replay_times(
        multiunit_detector_results)

    animal, day, epoch = epoch_key
    folder = 'replays_smoother' if use_smoother else 'replays_filter'

    for replay_number in tqdm(spikes_replay_info.index):
        fig, _ = plot_replay_with_data(
            replay_number, data, spikes_replay_info, replay_detector,
            spikes_detector_results, lfp_power_detector_results,
            multiunit_detector_results, epoch_key)

        figure_name = f'{animal}_{day:02d}_{epoch:02d}_{replay_number:03d}.png'
        figure_path = join(FIGURE_DIR, folder, figure_name)
        plt.savefig(figure_path, bbox_inches='tight')
        plt.close(fig)


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('Animal', type=str, help='Short name of animal')
    parser.add_argument('Day', type=int, help='Day of recording session')
    parser.add_argument('Epoch', type=int,
                        help='Epoch number of recording session')
    parser.add_argument('--use_smoother', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_command_line_arguments()
    epoch_key = (args.Animal, args.Day, args.Epoch)
    use_smoother = args.use_smoother
    sys.exit(main(epoch_key, use_smoother=use_smoother))
