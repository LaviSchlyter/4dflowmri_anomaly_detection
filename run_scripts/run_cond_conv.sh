#!/bin/bash
#SBATCH  --output=/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/sbatch_logs/sbatch_train_logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G

# --> CONFIGURE BEFORE RUNNING JOB
CONFIG_PATH="/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/config/config_cond_vae.yaml"
PREPROCESS_METHOD="masked_slice"
MODEL="cond_conv"
# --> CONFIGURE BEFORE RUNNING JOB

source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate anom
python -u ../train.py --model $MODEL --config_path $CONFIG_PATH --preprocess_method $PREPROCESS_METHOD