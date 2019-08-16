from collections import OrderedDict
from os.path import abspath, dirname, join, pardir

from loren_frank_data_processing import Animal

# LFP sampling frequency
SAMPLING_FREQUENCY = 1000

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')
FIGURE_DIR = join(ROOT_DIR, 'figures')

ANIMALS = {
    'bon': Animal(directory=join(RAW_DATA_DIR, 'Bond'), short_name='bon'),
    'cha': Animal(directory=join(RAW_DATA_DIR, 'Chapati'), short_name='cha'),
    # 'con': Animal(directory=join(RAW_DATA_DIR, 'Conley'), short_name='con'),
    'Cor': Animal(directory=join(RAW_DATA_DIR, 'Corriander'),
                  short_name='Cor'),
    'dav': Animal(directory=join(RAW_DATA_DIR, 'Dave'), short_name='dav'),
    'dud': Animal(directory=join(RAW_DATA_DIR, 'Dudley'), short_name='dud'),
    'egy': Animal(directory=join(RAW_DATA_DIR, 'Egypt'), short_name='egy'),
    'fra': Animal(directory=join(RAW_DATA_DIR, 'Frank'), short_name='fra'),
    'gov': Animal(directory=join(RAW_DATA_DIR, 'Government'),
                  short_name='gov'),
    'hig': Animal(directory=join(RAW_DATA_DIR, 'Higgs'), short_name='hig'),
}

USE_LIKELIHOODS = OrderedDict(
    [('lfp_power', ['lfp_power']),
     ('sorted_spikes', ['spikes']),
     ('clusterless', ['multiunit']),
     ('ad_hoc_ripple', ['ad_hoc_ripple']),
     ('ad_hoc_multiunit', ['ad_hoc_multiunit'])]
)

BRAIN_AREAS = ['CA1', 'CA2', 'CA3']

_10Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.100,
    time_window_step=0.100,
    time_halfbandwidth_product=1,
)
_4Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.250,
    time_window_step=0.250,
    time_halfbandwidth_product=1,
)
_2Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.500,
    time_window_step=0.500,
    time_halfbandwidth_product=1,
)

_12Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.250,
    time_window_step=0.250,
    time_halfbandwidth_product=3,
)

MULTITAPER_PARAMETERS = {
    '2Hz': _2Hz_Res,
    '4Hz': _4Hz_Res,
    '10Hz': _10Hz_Res,
    '12Hz': _12Hz_Res,
}

detector_parameters = {
    'movement_var': 4.0,
    'replay_speed': 1,
    'place_bin_size': 1.0,
    'lfp_model_kwargs': {'n_components': 1, 'max_iter': 200, 'tol': 1e-06},
    'spike_model_knot_spacing': 10,
    'spike_model_penalty': 0.5,
    'movement_state_transition_type': 'w_track_1D_random_walk'
}
