import itertools
import logging
import os
import sys
from argparse import ArgumentParser
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run

import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client
from loren_frank_data_processing.position import EDGE_ORDER, EDGE_SPACING
from replay_identification import ReplayDetector
from replay_trajectory_classification import ClusterlessDecoder
from src.load_data import load_data
from src.parameters import (FIGURE_DIR, MULTITAPER_PARAMETERS,
                            SAMPLING_FREQUENCY, USE_LIKELIHOODS,
                            detector_parameters)
from src.save_data import (save_non_overlap_info, save_overlap_info,
                           save_posterior, save_power, save_replay_info)
from src.summarize_replay import (add_epoch_info_to_dataframe, compare_overlap,
                                  decode_replays, get_non_overlap_info,
                                  get_replay_times, get_replay_triggered_power,
                                  summarize_replays)
from src.visualization import plot_behavior, plot_detector
from tqdm.autonotebook import tqdm

logging.basicConfig(level='INFO', format='%(asctime)s %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def decode(data, replay_detector, use_likelihoods,
           epoch_key, sampling_frequency):
    animal, day, epoch = epoch_key
    is_training = data['position_info'].speed > 4
    decoder = ClusterlessDecoder(
        place_bin_size=replay_detector.place_bin_size,
        replay_speed=replay_detector.replay_speed,
        movement_var=replay_detector.movement_var,
        transition_type='random_walk',
    )
    logging.info(decoder)
    decoder.fit(
        position=data['position_info'].linear_position,
        multiunits=data['multiunit'],
        is_training=is_training,
        track_graph=data['track_graph'],
        center_well_id=0,
        edge_order=EDGE_ORDER,
        edge_spacing=EDGE_SPACING)

    data_sources = []
    labels = []
    infos = []
    results = {}
    detector = {}

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
            detector_results = (
                replay_detector
                .predict(
                    speed=data['position_info'].speed,
                    position=data['position_info'].linear_position,
                    spikes=data['spikes'],
                    multiunit=data['multiunit'],
                    time=data['position_info'].index,
                    use_likelihoods=likelihoods,
                    use_smoother=True)
                .drop(['causal_posterior', 'likelihood']))
            replay_info, is_replay = get_replay_times(detector_results)
            results[data_source] = detector_results
            detector[data_source] = replay_detector
            save_posterior(detector_results.sum("state", skipna=False),
                           epoch_key, data_source)

        logging.info(f'Classifying replays with {data_source}...')
        replay_info = add_epoch_info_to_dataframe(replay_info, epoch_key,
                                                  data_source)

        if data_source in ['sorted_spikes', 'clusterless']:
            latent_position = [(detector_results.sum("state", skipna=False)
                                .sel(time=slice(row.start_time, row.end_time)))
                               for row in replay_info.itertuples()]
        else:
            latent_position = decode_replays(
                decoder, data, replay_info, sampling_frequency)
        logging.info(f'Summarizing replays with {data_source}...')
        replay_info = summarize_replays(
            replay_info, latent_position, data,
            SAMPLING_FREQUENCY)

        # Save Data
        logging.info(f'Saving {data_source}...')
        save_replay_info(data_source, epoch_key, replay_info,
                         is_replay)
        data_sources.append(data_source)
        labels.append(is_replay.replay_number)
        infos.append(replay_info)

        try:
            logging.info(f'Estimating {data_source} replay-triggered power...')
            power = get_replay_triggered_power(
                data['lfps'], replay_info, data['tetrode_info'],
                MULTITAPER_PARAMETERS['4Hz'], window_offset=(-0.250, 0.250),
                sampling_frequency=SAMPLING_FREQUENCY)
            save_power(power, epoch_key, data_source)
        except ValueError:
            pass

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
        save_overlap_info(
            overlap_info, epoch_key, data_source1, data_source2)

        logging.info('Analyzing replay non-overlap ...')
        non_overlap_info = get_non_overlap_info(
            labels1, labels2, data_source1, data_source2, epoch_key)
        save_non_overlap_info(
            non_overlap_info, epoch_key, data_source1, data_source2)

    # Plot each non-local chunk
    time_index = np.arange(data['position_info'].shape[0])

    for data_source, replay_info, is_replay in zip(
            data_sources, infos, labels):
        if data_source in ['sorted_spikes', 'clusterless']:
            logging.info(f'Plotting {data_source}...')
            folder = os.path.join(FIGURE_DIR, f'non_local_{data_source}')
            os.makedirs(folder, exist_ok=True)
            for replay_name, df in tqdm(replay_info.iterrows()):
                start_ind, end_ind = time_index[is_replay == df.replay_number][
                    [0, -1]]
                time_ind = slice(start_ind - 125, end_ind + 125)
                figsize = (20 * (end_ind - start_ind) / 1000, 7.5)
                plot_detector(time_ind, data, detector[data_source],
                              results[data_source],
                              figsize=figsize, data_source=data_source)
                figure_name = f'{replay_name}.png'
                figure_path = os.path.join(folder, figure_name)
                plt.savefig(figure_path, bbox_inches='tight')
                plt.close(plt.gcf())


def run_analysis(epoch_key, use_likelihoods):
    animal, day, epoch = epoch_key
    data_types = set(itertools.chain(*use_likelihoods.values()))
    data = load_data(epoch_key)
    assert np.allclose(data['is_ripple'].shape[0],
                       data['position_info'].shape[0],
                       data['spikes'].shape[0],
                       data['multiunit'].shape[0])
    plot_behavior(data['position_info'])
    figure_name = f'behavior_{animal}_{day:02d}_{epoch:02d}.png'
    plt.savefig(os.path.join(FIGURE_DIR, 'behavior', figure_name))

    replay_detector = ReplayDetector(**detector_parameters)
    logging.info(replay_detector)

    replay_detector.fit(
        is_ripple=data['is_ripple'],
        speed=data['position_info'].speed,
        position=data['position_info'].linear_position,
        spikes=data['spikes'],
        multiunit=data['multiunit'],
        track_graph=data['track_graph'],
        center_well_id=0,
        edge_order=EDGE_ORDER,
        edge_spacing=EDGE_SPACING,
    )

    # Plot detector fits
    if 'spikes' in data_types:
        axes = replay_detector.plot_spikes(
            data['spikes'], data['position_info'].linear_position,
            data['is_ripple'], sampling_frequency=SAMPLING_FREQUENCY)
        replay_detector.plot_fitted_place_fields(
            sampling_frequency=SAMPLING_FREQUENCY, axes=axes)
        figure_name = f'spikes_{animal}_{day:02d}_{epoch:02d}.png'
        plt.savefig(os.path.join(FIGURE_DIR, 'detector', figure_name))

    decode(data, replay_detector, use_likelihoods,
           epoch_key, SAMPLING_FREQUENCY)

    logging.info('Done...')


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('Animal', type=str, help='Short name of animal')
    parser.add_argument('Day', type=int, help='Day of recording session')
    parser.add_argument('Epoch', type=int,
                        help='Epoch number of recording session')
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--threads_per_worker', type=int, default=1)
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

    client_params = dict(n_workers=args.n_workers,
                         threads_per_worker=args.threads_per_worker,
                         processes=True,
                         memory_limit='25GB')
    with Client(**client_params) as client:
        logging.info(client)
        # Analysis Code
        run_analysis(epoch_key, USE_LIKELIHOODS)


if __name__ == '__main__':
    sys.exit(main())
