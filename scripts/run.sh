#!/bin/bash

pretrained_model_path="openai/whisper-large-v3"
data_dir="data"
batch_size=2 

python bin/main_predictions.py \
            --data_dir $data_dir \
            --pretrained_model_path $pretrained_model_path \
            --batch_size $batch_size
