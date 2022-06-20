#!/bin/bash

# Choose the specific GPU
cuda=3
# The path of python
python="/anaconda3/envs/torch/bin/python3.8"

# The output path of data_prepare.sh
dataset_path="/data/SoDA"
# Store the trained model
check_point_path="/data/SoDA/log"

preprocess_method="padding"
preprocess_strategy="normal_0"

# Apply specific model.
# assert seq_len // patch_size == 0
# ResNet: resnet18/resnet34/resnet50/resnet101
# LSTM: lstm_es/lstm_ms/lstm_s/lstm_b
# MLP-Mixer: mixer_es_(patch_size)/mixer_ms_(patch_size)/mixer_s_(patch_size)/mixer_b_(patch_size)
# ViT: vit_es_(patch_size)/vit_ms_(patch_size)/vit_s_(patch_size)/vit_b_(patch_size)
model_name="resnet18"

# training
echo "${preprocess_method} and ${preprocess_strategy} - training"
CUDA_VISIBLE_DEVICES=${cuda} ${python} training.py --dataset_path ${dataset_path} \
--preprocess_method ${preprocess_method} --preprocess_strategy ${preprocess_strategy} --seq_len 224 \
--train_batch_size 512 --eval_batch_size 512 --num_epoch 400 --opt_method "adamw" \
--lr_rate 5e-4 --lr_rate_adjust_epoch 40 --lr_rate_adjust_factor 0.5 --weight_decay 1e-4 \
--save_epoch 401 --eval_epoch 10 --patience 40 \
--check_point_path ${check_point_path} --use_gpu true --gpu_device ${cuda} \
--model_name ${model_name} --head_name "span_cls" --strategy_name "span_cls"

# testing
echo "${preprocess_method} and ${preprocess_strategy} - testing"
CUDA_VISIBLE_DEVICES=${cuda} ${python} testing.py --dataset_path ${dataset_path} \
--preprocess_method ${preprocess_method} --preprocess_strategy ${preprocess_strategy} --seq_len 224 \
--check_point_path ${check_point_path} --use_gpu true --gpu_device ${cuda} \
--model_name ${model_name} --head_name "span_cls" --strategy_name "span_cls" \
--test_batch_size 512