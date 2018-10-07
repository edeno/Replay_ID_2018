from collections import OrderedDict
from os.path import abspath, dirname, join, pardir

from loren_frank_data_processing import Animal

# LFP sampling frequency
SAMPLING_FREQUENCY = 1500

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')
FIGURE_DIR = join(ROOT_DIR, 'figures')

ANIMALS = {
    'bon': Animal(directory=join(RAW_DATA_DIR, 'Bond'),
                  short_name='bon'),
    'fra': Animal(directory=join(RAW_DATA_DIR, 'frank'),
                  short_name='fra'),
    'Cor': Animal(directory=join(RAW_DATA_DIR, 'CorrianderData'),
                  short_name='Cor'),
}

USE_LIKELIHOODS = OrderedDict(
    [('lfp_power', ['lfp_power']),
     ('spikes', ['spikes']),
     ('spikes_and_lfp_power', ['spikes', 'lfp_power'])]
)
