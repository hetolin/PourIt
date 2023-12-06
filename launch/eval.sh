#!/bin/bash

file=scripts/eval.py
config=configs/pourit_seen_ours.yaml
echo $file --config $config
python $file --config $config

file=scripts/eval.py
config=configs/pourit_unseen_ours.yaml
echo $file --config $config
python $file --config $config


