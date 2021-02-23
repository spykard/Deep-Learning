#!/bin/bash

# OUTPUTDIR is directory containing this run.sh script
OUTPUTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python train.py \
  --num_hidden_layers 6 \
  --num_attention_heads 9 \
  --optimizer_cosine_lr True \
  --optimizer_warmup_ratio 0.05 \
  --batch_size 100 \
  --num_epochs 300 \
  --hidden_size 400 \
  --use_learned_2d_encoding True \
  --use_gaussian_attention False \
  --num_keep_checkpoints 30 \
  --output_dir $OUTPUTDIR

python train.py --num_hidden_layers 6 --num_attention_heads 9 --optimizer_cosine_lr True --optimizer_warmup_ratio 0.05 --batch_size 32 --num_epochs 300 --hidden_size 400 --use_learned_2d_encoding True --use_gaussian_attention False --num_keep_checkpoints 30 --dataset FER2013