import itertools
import logging
import sys
from argparse import ArgumentParser
from os.path import join
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run

import matplotlib.pyplot as plt
from replay_trajectory_classification import SortedSpikesDecoder

from replay_identification import ReplayDetector
from src.load_data import load_data
from src.parameters import (FIGURE_DIR, MULTITAPER_PARAMETERS,
                            SAMPLING_FREQUENCY, USE_LIKELIHOODS,
                            detector_parameters)
from src.save_data import save_overlap, save_power, save_replay_data
from src.summarize_replay import (add_epoch_info_to_dataframe, compare_overlap,
                                  decode_replays, get_non_overlap_info,
                                  get_replay_times, get_replay_triggered_power,
                                  save_non_overlap, summarize_replays)
from src.visualization import plot_behavior

logging.basicConfig(level='INFO', format='%(asctime)s %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def decode(data, replay_detector, track_labels, use_likelihoods,
           epoch_key, sampling_frequency, use_smoother, position_metric,
           speed_metric):
    is_training = data['position_info'][speed_metric] > 4
    decoder = SortedSpikesDecoder(
        place_bin_size=replay_detector.place_bin_size,
        replay_speed=replay_detector.replay_speed,
        movement_var=replay_detector.movement_var,
        knot_spacing=replay_detector.spike_model_knot_spacing,
        spike_model_penalty=replay_detector.spike_model_penalty,
        transition_type='w_track_1D_random_walk')
    decoder.fit(
        position=data['position_info'][position_metric],
        spikes=data['spikes'], is_training=is_training,
        track_labels=track_labels)

    data_sources = []
    labels = []
    infos = []

    for data_source, likelihoods in use_likelihoods.items():
        logging.info(f'Finding replays with {data_source}...')
        if data_source == 'ad_hoc_ripple':
            replay_info = data['ripple_times'].copy()
            is_replay = data['ripple_labels'].copy()
            detector_results = []
        elif data_source == 'ad_hoc_multiunit':
            replay_info = data['multiunit_high_synchrony_times'].copy()
            is_replay = data['multiunit_high_synchrony_labels'].copy()
            detector_results = []
        else:
            detector_results = replay_detector.predict(
                speed=data['position_info'][speed_metric],
                position=data['position_info'][position_metric],
                lfp_power=data['ripple_power'],
                spikes=data['spikes'], multiunit=data['multiunit'],
                time=data['position_info'].index,
                use_likelihoods=likelihoods,
                use_smoother=use_smoother)
            replay_info, is_replay = get_replay_times(detector_results)

        logging.info(f'Classifying replays with {data_source}...')
        replay_info = add_epoch_info_to_dataframe(replay_info, epoch_key,
                                                  data_source)
        if data_source in ['sorted_spikes', 'clusterless']:
            decoder_results = [(detector_results
                                .sel(time=slice(row.start_time, row.end_time),
                                     state='Replay'))
                               for row in replay_info.itertuples()]
        else:
            decoder_results = decode_replays(
                decoder, data, replay_info, sampling_frequency, use_smoother)
        logging.info(f'Summarizing replays with {data_source}...')
        replay_info = summarize_replays(
            replay_info, decoder_results, data,
            SAMPLING_FREQUENCY, position_metric)

        power = get_replay_triggered_power(
            data['lfps'], replay_info, data['tetrode_info'],
            MULTITAPER_PARAMETERS['4Hz'], window_offset=(-0.250, 0.250),
            sampling_frequency=SAMPLING_FREQUENCY)
        save_power(power, epoch_key, use_smoother, data_source)

        # Save Data
        save_replay_data(data_source, epoch_key, replay_info,
                         is_replay, use_smoother)
        data_sources.append(data_source)
        labels.append(is_replay.replay_number)
        infos.append(replay_info)

    comb = itertools.combinations(zip(labels, infos, data_sources), 2)
    for (labels1, info1, data_source1), (labels2, info2, data_source2) in comb:
        logging.info(
            'Analyzing replay overlap between '
            f'{data_source1} and {data_source2}...')
        overlap_info = compare_overlap(
            labels1, labels2, info1, info2, SAMPLING_FREQUENCY,
            epoch_key, data_source1, data_source2)
        if overlap_info.shape[0] == 0:
            logging.warn('No overlap detected.')
        save_overlap(
            overlap_info, epoch_key, data_source1, data_source2, use_smoother)

        logging.info('Analyzing replay non-overlap ...')
        non_overlap_info = get_non_overlap_info(
            labels1, labels2, data_source1, data_source2, epoch_key)
        save_non_overlap(
            non_overlap_info, epoch_key, data_source1, data_source2,
            use_smoother)


def run_analysis(epoch_key, use_likelihoods,
                 position_metric='linear_position2',
                 speed_metric='speed', use_smoother=True):
    animal, day, epoch = epoch_key
    data_types = set(itertools.chain(*use_likelihoods.values()))
    data = load_data(epoch_key)
    plot_behavior(data['position_info'], position_metric)
    figure_name = f'behavior_{animal}_{day:02d}_{epoch:02d}.png'
    plt.savefig(join(FIGURE_DIR, 'behavior', figure_name))

    replay_detector = ReplayDetector(**detector_parameters)
    logging.info(replay_detector)

    track_labels = data['position_info'].arm_name
    replay_detector.fit(
        is_ripple=data['is_ripple'], speed=data['position_info'][speed_metric],
        position=data['position_info'][position_metric],
        lfp_power=data['ripple_power'], spikes=data['spikes'],
        multiunit=data['multiunit'], track_labels=track_labels)

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
        replay_detector.plot_lfp_power(data['ripple_power'], data['is_ripple'])
        figure_name = f'lfp_power_{animal}_{day:02d}_{epoch:02d}.png'
        plt.savefig(join(FIGURE_DIR, 'detector', figure_name))

        replay_detector.plot_fitted_lfp_power_model()
        figure_name = f'fitted_lfp_power_{animal}_{day:02d}_{epoch:02d}.png'
        plt.savefig(join(FIGURE_DIR, 'detector', figure_name))

    decode(data, replay_detector, track_labels, use_likelihoods,
           epoch_key, SAMPLING_FREQUENCY, use_smoother, position_metric,
           speed_metric)

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
    run_analysis(epoch_key, USE_LIKELIHOODS)


if __name__ == '__main__':
    sys.exit(main())
