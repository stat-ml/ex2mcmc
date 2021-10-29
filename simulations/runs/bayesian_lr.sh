#!/bin/bash

for dataset_config in  covertype eeg digits pima swiss breast
do
    echo "Dataset ${dataset_config}"
    python experiments/bayesian_lr.py \
        configs/bayesian_log_reg.yaml \
        --dataset_config configs/datasets/${dataset_config}.yaml
done
