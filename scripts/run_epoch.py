import itertools
from replay_identification import ReplayDetector
from src.load_data import load_data
from src.parameters import ANIMALS, SAMPLING_FREQUENCY
from src.save_data import (save_replay_data, save_ripple_data)
from src.summarize_replay import (decode_replays, get_replay_times,
                                  summarize_replays)


def run_analysis(epoch_key, animals, sampling_frequency, use_likelihoods,
                 position_metric='linear_distance'):
    data_types = set(itertools.chain(*use_likelihoods.values()))
    data = load_data(epoch_key, animals, sampling_frequency, data_types)

    replay_detector = ReplayDetector()
    replay_detector.fit(
        data['is_ripple'], data['position_info'].speed.values,
        data['position_info'][position_metric].values, data['power'],
        data['spikes'], data['multiunit'])

    for name, likelihoods in use_likelihoods.items():
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
            detector_results, decoder_results, data, position_metric)

        # Save Data
        save_replay_data(name, epoch_key, replay_info, replay_densities,
                         is_replay)

    save_ripple_data(epoch_key, data)


if __name__ == '__main__':
    epoch_key = ('bon', 3, 2)
    use_likelihoods = {
        'lfp_power': ['lfp_power'],
        'spikes': ['spikes'],
        'spikes_and_lfp_power': ['spikes', 'lfp_power']
    }
    run_analysis(epoch_key, ANIMALS, SAMPLING_FREQUENCY, use_likelihoods)
