import itertools
import logging

from replay_identification import ReplayDetector
from src.load_data import load_data
from src.parameters import ANIMALS, SAMPLING_FREQUENCY, USE_LIKELIHOODS
from src.save_data import save_overlap, save_replay_data, save_ripple_data
from src.summarize_replay import (compare_overlap, decode_replays,
                                  get_replay_times, summarize_replays)


def run_analysis(epoch_key, animals, sampling_frequency, use_likelihoods,
                 position_metric='linear_distance'):
    data_types = set(itertools.chain(*use_likelihoods.values()))
    data = load_data(epoch_key, animals, sampling_frequency, data_types)

    replay_detector = ReplayDetector()
    replay_detector.fit(
        data['is_ripple'], data['position_info'].speed.values,
        data['position_info'][position_metric].values, data['power'],
        data['spikes'], data['multiunit'])

    names = []
    labels = []
    infos = []

    for name, likelihoods in use_likelihoods.items():
        logging.info(f'Finding replays with {name}...')
        detector_results = replay_detector.predict(
            data['position_info'].speed.values,
            data['position_info'][position_metric].values, data['power'],
            data['spikes'], data['multiunit'],
            time=data['position_info'].index, use_likelihoods=likelihoods)

        replay_info, is_replay = get_replay_times(detector_results)
        decoder_results, _ = decode_replays(
            data, replay_detector, is_replay, replay_info, sampling_frequency,
            position_metric)
        replay_info, replay_densities = summarize_replays(
            replay_info, detector_results, decoder_results, data,
            position_metric)

        # Save Data
        save_replay_data(name, epoch_key, replay_info, replay_densities,
                         is_replay)
        names.append(name)
        labels.append(is_replay.replay_number)
        infos.append(replay_info)

    save_ripple_data(epoch_key, data)
    names.append('ripples')
    labels.append(data['ripple_labels'])
    infos.append(data['ripple_times'])

    combination = itertools.combinations(zip(labels, infos, names), 2)
    for (labels1, info1, name1), (labels2, info2, name2) in combination:
        logging.info(f'Analyzing overlap between {name1} and {name2}...')
        overlap = compare_overlap(labels1, labels2, info1, info2)
        save_overlap(overlap, epoch_key, name1, name2)


if __name__ == '__main__':
    epoch_key = ('bon', 3, 2)
    run_analysis(epoch_key, ANIMALS, SAMPLING_FREQUENCY, USE_LIKELIHOODS)
