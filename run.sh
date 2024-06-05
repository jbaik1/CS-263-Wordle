#!/bin/bash

MODEL=meta-llama/Meta-Llama-3-8B-Instruct
GAMES=1
SHOTS=0
TURNS=5

CUDA_VISIBLE_DEVICES=5 python3 run_models.py \
    --model $MODEL \
    --num_games $GAMES \
    --shots $SHOTS \
    --max_turns $TURNS \
    --use_vllm