#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy', 'scipy', 'xarray', 'seaborn', 'pandas',
                    'loren_frank_data_processing', 'ripple_detection',
                    'replay_identification', 'replay_classification',
                    'spectral_connectivity']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='replay_id_2018',
    version='0.1.0.dev0',
    license='MIT',
    description=('Analysis of replays identified from different sources.'),
    author='Eric Denovellis, Long Tao',
    author_email='edeno@bu.edu, longtao2@bu.edu',
    url='https://github.com/edeno/Replay_ID_2018',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
