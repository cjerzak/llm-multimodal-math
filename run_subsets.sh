#!/bin/bash
# run all fingerprinting / nudging skipping lora training (use cached results)
./run_all.sh M4 all 0 --skip-lora-training
./run_all.sh M4 all 0 --skip-lora-training --model-size 30b  # Combine flags
./run_all.sh M4 all 0 --skip-lora-training --model-size 235b  # Combine flags

# just run figures 
python Scripts/analysis/GenerateResultsFigures.py
