#!/bin/bash -l

#$ -P braincom
#$ -l h_rt=4:00:00
#$ -l mem_total=125G
#$ -j y
#$ -o create_figure.qlog
#$ -pe omp 4

python create_figures.py

# echo python create_figures.py | qsub -l h_rt=4:00:00 -pe omp 4 -P braincom -l mem_total=125G -j y -o create_figure.log
