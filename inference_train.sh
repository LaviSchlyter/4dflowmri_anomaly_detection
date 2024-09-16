#!/bin/bash
#SBATCH  --output=sbatch_logs/sbatch_inference_logs/inference_train/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G

source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate anom
python -u inference_train.py "$@"