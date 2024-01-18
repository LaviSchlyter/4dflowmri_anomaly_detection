#!/bin/bash
#SBATCH  --output=sbatch_log_inference/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G

source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate vtk_wrap
python -u inference_test.py "$@"