#!/bin/bash

pretrained_model_path="openai/whisper-large-v3"
data_dir="data"
batch_size=2 
task="asr"

python bin/main_predictions.py \
            --data_dir $data_dir \
            --pretrained_model_path $pretrained_model_path \
            --task $task \
            --batch_size $batch_size
