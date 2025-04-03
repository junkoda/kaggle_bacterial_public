#!/bin/sh

set -x

# Preprocess training data
python3 src/prepare.py baseline.yml

# Train model
date +"%Y-%m-%d %H:%M" >> log.txt
python3 -u src/train.py baseline.yml | tee -a log.txt

# Compute Local CV
python3 src/predict.py baseline.yml
