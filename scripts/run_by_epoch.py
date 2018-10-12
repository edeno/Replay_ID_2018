import itertools
import logging
import sys
from argparse import ArgumentParser
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run

from replay_identification import ReplayDetector
from src.load_data import load_data
from src.parameters import ANIMALS, SAMPLING_FREQUENCY, USE_LIKELIHOODS
from src.save_data import save_overlap, save_replay_data, save_ripple_data
from src.summarize_replay import (add_epoch_info_to_dataframe, compare_overlap,
                                  decode_replays, get_replay_times,
                                  summarize_replays)


def run_analysis(epoch_key, animals, sampling_frequency, use_likelihoods,
                 position_metric='linear_distance'):
    data_types = set(itertools.chain(*use_likelihoods.values()))
    data = load_data(epoch_key, animals, sampling_frequency, data_types)

    if data['spikes'] is None:
        s = data['spikes']
    else:
        s = data['spikes'].values

    if data['multiunit'] is None:
        m = data['multiunit']
    else:
        m = data['multiunit'].values

    replay_detector = ReplayDetector()
    replay_detector.fit(
        data['is_ripple'].values, data['position_info'].speed.values,
        data['position_info'][position_metric].values, data['power'],
        s, m)

    names = []
    labels = []
    infos = []

    for name, likelihoods in use_likelihoods.items():
        logging.info(f'Finding replays with {name}...')
        detector_results = replay_detector.predict(
            data['position_info'].speed.values,
            data['position_info'][position_metric].values, data['power'],
            s, m, time=data['position_info'].index,
            use_likelihoods=likelihoods)

        replay_info, is_replay = get_replay_times(detector_results)
        logging.info(f'Classifying replays with {name}...')
        decoder_results, _ = decode_replays(
            data, replay_detector, is_replay, replay_info, sampling_frequency,
            position_metric)
        logging.info(f'Summarizing replays with {name}...')
        replay_info, replay_densities = summarize_replays(
            replay_info, detector_results, decoder_results, data,
            position_metric)

        add_epoch_info_to_dataframe(replay_info, epoch_key)

        # Save Data
        save_replay_data(name, epoch_key, replay_info, replay_densities,
                         is_replay)
        names.append(name)
        labels.append(is_replay.replay_number)
        infos.append(replay_info)

    add_epoch_info_to_dataframe(data['ripple_times'], epoch_key)
    save_ripple_data(epoch_key, data)
    names.append('ripples')
    labels.append(data['ripple_labels'])
    infos.append(data['ripple_times'])

    combination = itertools.combinations(zip(labels, infos, names), 2)
    for (labels1, info1, name1), (labels2, info2, name2) in combination:
        logging.info(
            f'Analyzing replay overlap between {name1} and {name2}...')
        overlap = compare_overlap(labels1, labels2, info1, info2)
        save_overlap(overlap, epoch_key, name1, name2)
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
