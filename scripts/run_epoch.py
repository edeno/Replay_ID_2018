from src.analysis import (decode_replays, get_replay_times, identify_replays,
                          load_data, summarize_replays, save_data)
from src.parameters import ANIMALS, SAMPLING_FREQUENCY

if __name__ == '__main__':
    epoch_key = ('bon', 3, 2)
    use_likelihoods = ['spikes', 'lfp_power']

    data = load_data(epoch_key, ANIMALS, SAMPLING_FREQUENCY, use_likelihoods)

    detector_results, replay_detector = identify_replays(data, use_likelihoods)
    replay_info, is_replay = get_replay_times(detector_results)

    decoder_results, replay_decoder = decode_replays(
        data, replay_detector, is_replay, replay_info, SAMPLING_FREQUENCY)

    replay_info, detector_posterior, decoder_posterior = summarize_replays(
        detector_results, decoder_results, data)

    # Save Data
    save_data(replay_info, detector_posterior, decoder_posterior, is_replay,
              use_likelihoods, epoch_key)
