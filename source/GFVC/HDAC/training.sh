#!/bin/bash
MODEL="${1:-hdac}" 
#Other parameters are configured from the yaml files in <config/*> folder
#######
python run.py --config config/hdac.yaml --log_dir train_logs/  --debug
