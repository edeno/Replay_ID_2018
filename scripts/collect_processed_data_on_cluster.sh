#!/bin/bash -l

#$ -P braincom
#$ -l h_rt=18:00:00
#$ -l mem_total=125G
#$ -j y
#$ -o create_figure.log
#$ -pe omp 16

python collect_processed_data.py
