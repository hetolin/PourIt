#!/bin/bash

port=29501
crop_size=512

file=scripts/train.py
config=configs/pourit_seen_ours.yaml
echo python -m torch.distributed.launch --nproc_per_node=6 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size
python -m torch.distributed.launch --nproc_per_node=6 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size
