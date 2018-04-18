#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['loren_frank_data_processing', 'ripple_detection',
                    'replay_identification', 'replay_classification']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='replay_id_2018',
    version='0.1.0.dev0',
    license='GPL-3.0',
    description=('Analysis of replays identified from different sources.'),
    author='Eric Denovellis',
    author_email='edeno@bu.edu',
    url='https://github.com/edeno/Replay_ID_2018',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
