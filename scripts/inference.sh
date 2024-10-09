#!/bin/bash
#SBATCH  --output=../logs/inference_logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G

# Activate the Conda environment
source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate anom
python -u ../src/inference/inference_test.py --config_path /usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/config/config_inference.yaml "$@"
