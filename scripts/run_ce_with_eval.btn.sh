#!/bin/bash

for seed in 1 2 3;

do
   
    python main_novelclasses.py --dataset FungiSmall  \
    --n_epochs 1 --iters_per_epoch 30 --lr $1 --seed $seed \
    --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --raw_gpt_centroid_eval 1 --raw_gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1 
    
done