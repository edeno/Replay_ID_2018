#!/bin/bash -l

#$ -P braincom
#$ -l h_rt=5:00:00
#$ -l mem_total=125G
#$ -j y
#$ -o create_figure.log
#$ -pe omp 16

python generate_replay_examples.py bon 3 2 --use_smoother
