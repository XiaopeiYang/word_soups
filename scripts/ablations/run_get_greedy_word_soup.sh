#!/bin/bash -l

python preprocess/get_greedy_word_soup.py --dataset $1 --seed 1 --n_descriptors 8 --subsample_classes base
python preprocess/get_greedy_word_soup.py --dataset $1 --seed 2 --n_descriptors 8 --subsample_classes base
python preprocess/get_greedy_word_soup.py --dataset $1 --seed 3 --n_descriptors 8 --subsample_classes base