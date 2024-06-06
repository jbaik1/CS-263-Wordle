#!/bin/bash

# MODEL=meta-llama/Meta-Llama-3-8B-Instruct
MODEL=HuggingFaceH4/zephyr-7b-beta
GAMES=488
TURNS=5

for SHOTS in 0
do
CUDA_VISIBLE_DEVICES=0 python3 run_models.py \
    --model $MODEL \
    --num_games $GAMES \
    --shots $SHOTS \
    --max_turns $TURNS \
    --use_vllm
done