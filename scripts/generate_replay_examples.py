import logging
import os
import sys
from argparse import ArgumentParser
from collections import OrderedDict
from os.path import join

import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from tqdm.auto import tqdm

from replay_identification import ReplayDetector
from src.load_data import load_data
from src.parameters import ANIMALS, BRAIN_AREAS, FIGURE_DIR, SAMPLING_FREQUENCY
from src.summarize_replay import get_replay_times
from src.visualization import plot_replay_with_data

logging.basicConfig(level=logging.INFO)

USE_LIKELIHOODS = OrderedDict(
    [('lfp_power', ['lfp_power']),
     ('spikes', ['spikes']),
     ('multiunit', ['multiunit']),
     ('ad_hoc_ripple', ['ad_hoc_ripple']),
     ('ad_hoc_multiunit', ['ad_hoc_multiunit'])]
)


def main(epoch_key, speed_metric='linear_speed',
         position_metric='linear_distance', use_smoother=False):
    data = load_data(epoch_key, ANIMALS, SAMPLING_FREQUENCY,
                     BRAIN_AREAS, speed_metric)

    replay_detector = ReplayDetector(
        multiunit_density_model=BayesianGaussianMixture,
        multiunit_model_kwargs=dict(n_components=300, tol=1E-8))
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

    replay_infos = OrderedDict(
        [('ad_hoc_ripple', data['ripple_times']),
         ('ad_hoc_multiunit', data['multiunit_high_synchrony_times']),
         ('lfp_power', lfp_power_replay_info),
         ('spikes', spikes_replay_info),
         ('multiunit', multiunit_replay_info)],
    )

    animal, day, epoch = epoch_key
    smoother_type = 'smoother' if use_smoother else 'filter'
    for data_source, replay_info in replay_infos.items():
        logging.info(f'{data_source}...')
        folder = join(FIGURE_DIR, f'replays_{data_source}_{smoother_type}')
        os.makedirs(folder, exist_ok=True)
        for replay_number in tqdm(replay_info.index):
            fig, _ = plot_replay_with_data(
                replay_number, data, replay_info, replay_detector,
                spikes_detector_results, lfp_power_detector_results,
                multiunit_detector_results, epoch_key)
            figure_name = (
                f'{animal}_{day:02d}_{epoch:02d}_{replay_number:03d}.png')
            figure_path = join(folder, figure_name)
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
