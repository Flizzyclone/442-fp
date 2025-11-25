#!/bin/bash
#SBATCH -output=example_job.log
echo "hello world"
python train.py
