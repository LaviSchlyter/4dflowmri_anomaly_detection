#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

# --> CONFIGURE BEFORE RUNNING JOB
CONFIG_PATH="/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/config/config_cnn.yaml"
PREPROCESS_METHOD="masked_slice"
MODEL="simple_conv"
# --> CONFIGURE BEFORE RUNNING JOB

# Define a timestamp for the log filename
TIMESTAMP=$(date +'%Y%m%d_%H%M')

# Dynamically set the log output path using the model and preprocess method
LOG_DIR="../logs/train_logs"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_${MODEL}_${PREPROCESS_METHOD}.out"

# Ensure the log directory exists
mkdir -p $LOG_DIR

# Activate the conda environment
source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate anom


# Redirect stdout and stderr to your dynamically generated log file
exec > >(tee -i $LOG_FILE)
exec 2>&1

# Cleanup: Remove all Slurm log files in the current directory
find . -name "slurm-*.out" -type f -exec rm {} \;

# Run the training script with model and preprocess method as arguments
exec python -u ../src/training/train.py --model $MODEL --config_path $CONFIG_PATH --preprocess_method $PREPROCESS_METHOD
