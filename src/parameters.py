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
     ('spikes', ['spikes']),
     ('spikes_and_lfp_power', ['spikes', 'lfp_power']),
     ('multiunit', ['multiunit']),
     ('ripple', ['ripple'])]
)

BRAIN_AREAS = ['CA1', 'CA2', 'CA3']
