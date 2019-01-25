#!/bin/bash -l

#$ -P braincom
#$ -l h_rt=12:00:00
#$ -l mem_total=125G
#$ -j y
#$ -o replay_examples.log
#$ -pe omp 16
export OPENBLAS_NUM_THREADS=16
python generate_replay_examples.py bon 3 2 --use_smoother
