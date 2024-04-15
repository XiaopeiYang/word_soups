#!/bin/bash

for seed in 1 2 3;

do

    python main_novelclasses.py --prompt_lr_multi 1 --maple 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 3  --dataset FungiSmall  \
    --n_epochs 1 --iters_per_epoch 30 --lr $1 --seed $seed \
    --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --raw_gpt_centroid_eval 1 --raw_gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1 
    
done