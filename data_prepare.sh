#!/bin/bash

# Create a new folder "SoDA" on the disk where the data is stored
# and unzip the download dataset into "SoDA".
datasource_path="/data/SoDA/datasource"
output_path="/data/SoDA"

# Slice the whole Time-Series into several segment as "clip".
# If clip length is less than seq_len, then upsample or pad to the corresponding length.
method="padding" # upsampling or padding
# After slicing, based on different strategy, divided clip into train set or test set.
# normal_(0-4): Test every person's i*2 ~ i*2 + 1 attempt, and the rest of the data for training.
# user_(1-10): Test i-th person's clip, and the rest of the data for training.
# shuffle_(0-∞): Randomly choose 20% of whole clips into test set, and the rest of the data for training. You can shuffle infinite time.
strategy="normal_0" # normal_(0-4)/user_(1-10)/shuffle_(0-∞)
# The clip's length.
seq_len=224

python ./data_process/sensor_data_preprocess.py --input ${datasource_path} --output ${output_path} \
--method ${method} --strategy ${strategy} --length ${seq_len}