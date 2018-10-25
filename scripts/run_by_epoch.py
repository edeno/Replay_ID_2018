import itertools
import logging
import sys
from argparse import ArgumentParser
from os.path import join
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run

import matplotlib.pyplot as plt

from replay_identification import ReplayDetector
from src.load_data import load_data
from src.parameters import (ANIMALS, FIGURE_DIR, SAMPLING_FREQUENCY,
                            USE_LIKELIHOODS, BRAIN_AREAS)
from src.save_data import save_overlap, save_replay_data
from src.summarize_replay import (add_epoch_info_to_dataframe, compare_overlap,
                                  decode_replays, get_replay_times,
                                  summarize_replays)
from src.visualization import plot_behavior


def run_analysis(epoch_key, animals, sampling_frequency, use_likelihoods,
                 position_metric='linear_distance',
                 speed_metric='linear_speed'):
    animal, day, epoch = epoch_key
    data_types = set(itertools.chain(*use_likelihoods.values()))
    data = load_data(epoch_key, animals, sampling_frequency, data_types,
                     BRAIN_AREAS, speed_metric)
    plot_behavior(data['position_info'], position_metric)
    figure_name = f'behavior_{animal}_{day:02d}_{epoch:02d}.png'
    plt.savefig(join(FIGURE_DIR, 'behavior', figure_name))

    replay_detector = ReplayDetector()
    replay_detector.fit(
        is_replay=data['is_ripple'], speed=data['position_info'][speed_metric],
        position=data['position_info'][position_metric],
        lfp_power=data['power'], spikes=data['spikes'],
        multiunit=data['multiunit'])

    # Plot detector fits
    if 'spikes' in data_types:
        axes = replay_detector.plot_spikes(
            data['spikes'], data['position_info'][position_metric],
            data['is_ripple'], sampling_frequency=SAMPLING_FREQUENCY)
        replay_detector.plot_fitted_place_fields(
            sampling_frequency=SAMPLING_FREQUENCY, axes=axes)
        figure_name = f'spikes_{animal}_{day:02d}_{epoch:02d}.png'
        plt.savefig(join(FIGURE_DIR, 'detector', figure_name))

    if 'lfp_power' in data_types:
        replay_detector.plot_lfp_power(data['power'], data['is_ripple'])
        figure_name = f'lfp_power_{animal}_{day:02d}_{epoch:02d}.png'
        plt.savefig(join(FIGURE_DIR, 'detector', figure_name))

        replay_detector.plot_fitted_lfp_power_model()
        figure_name = f'fitted_lfp_power_{animal}_{day:02d}_{epoch:02d}.png'
        plt.savefig(join(FIGURE_DIR, 'detector', figure_name))

    if 'multiunit' in data_types:
        replay_detector.plot_multiunit(
            data['multiunit'],
            data['position_info'][position_metric], data['is_ripple'])
        figure_name = f'multiunit_{animal}_{day:02d}_{epoch:02d}.png'
        plt.savefig(join(FIGURE_DIR, 'detector', figure_name))

        replay_detector.plot_fitted_multiunit_model()
        figure_name = f'fitted_multiunit_{animal}_{day:02d}_{epoch:02d}.png'
        plt.savefig(join(FIGURE_DIR, 'detector', figure_name))

    data_sources = []
    labels = []
    infos = []

    for data_source, likelihoods in use_likelihoods.items():
        logging.info(f'Finding replays with {data_source}...')
        if data_source != 'ripple':
            detector_results = replay_detector.predict(
                speed=data['position_info'][speed_metric],
                position=data['position_info'][position_metric],
                lfp_power=data['power'],
                spikes=data['spikes'], multiunit=data['multiunit'],
                time=data['position_info'].index,
                use_likelihoods=likelihoods)
            replay_info, is_replay = get_replay_times(detector_results)
        else:
            replay_info = data['ripple_times'].copy()
            is_replay = data['ripple_labels'].copy()

        logging.info(f'Classifying replays with {data_source}...')
        replay_info = add_epoch_info_to_dataframe(replay_info, epoch_key,
                                                  data_source)
        decoder_results, _ = decode_replays(
            data, replay_detector, is_replay, replay_info, sampling_frequency,
            position_metric)
        logging.info(f'Summarizing replays with {data_source}...')
        replay_info, replay_densities = summarize_replays(
            replay_info, detector_results, decoder_results, data,
            position_metric)

        # Save Data
        save_replay_data(data_source, epoch_key, replay_info, replay_densities,
                         is_replay)
        data_sources.append(data_source)
        labels.append(is_replay.replay_number)
        infos.append(replay_info)

    combination = itertools.combinations(zip(labels, infos, data_sources), 2)
    for (labels1, info1, name1), (labels2, info2, name2) in combination:
        logging.info(
            f'Analyzing replay overlap between {name1} and {name2}...')
        overlap_info = compare_overlap(
            labels1, labels2, info1, info2, SAMPLING_FREQUENCY)
        overlap_info['animal'] = animal
        overlap_info['day'] = day
        overlap_info['epoch'] = epoch
        overlap_info['data_source1'] = name1
        overlap_info['data_source2'] = name2
        save_overlap(overlap_info, epoch_key, name1, name2)
    logging.info('Done...')


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('Animal', type=str, help='Short name of animal')
    parser.add_argument('Day', type=int, help='Day of recording session')
    parser.add_argument('Epoch', type=int,
                        help='Epoch number of recording session')
    parser.add_argument(
        '-d', '--debug',
        help='More verbose output for debugging',
        action='store_const',
        dest='log_level',
        const=logging.DEBUG,
        default=logging.INFO,
    )
    return parser.parse_args()


def main():
    args = get_command_line_arguments()
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=FORMAT, level=args.log_level)

    def _signal_handler(signal_code, frame):
        logging.error('***Process killed with signal {signal}***'.format(
            signal=signal_code))
        exit()

    for code in [SIGUSR1, SIGUSR2]:
        signal(code, _signal_handler)

    epoch_key = (args.Animal, args.Day, args.Epoch)
    logging.info(
        'Processing epoch: Animal {0}, Day {1}, Epoch #{2}...'.format(
            *epoch_key))
    git_hash = run(['git', 'rev-parse', 'HEAD'],
                   stdout=PIPE, universal_newlines=True).stdout
    logging.info('Git Hash: {git_hash}'.format(git_hash=git_hash.rstrip()))

    # Analysis Code
    run_analysis(epoch_key, ANIMALS, SAMPLING_FREQUENCY, USE_LIKELIHOODS)


if __name__ == '__main__':
    sys.exit(main())
