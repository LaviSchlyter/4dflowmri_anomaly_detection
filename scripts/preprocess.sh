#!/bin/bash
#SBATCH --output=../logs/preprocess_logs/%j_preprocess.log
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

CONFIG_PATH="/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/config/config_preprocessing.yaml"


source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate anom

# Step 1: Preprocess data
python -u ../src/helpers/data_bern_numpy_to_preprocessed_hdf5.py --config_path $CONFIG_PATH "$@"

# Step 2: Create quadrant masks
python -u ../src/helpers/create_quadrant_masks.py 

# Step 3: Introduce synthetic anomalies into validation data
python -u ../src/helpers/synthetic_anomalies.py --config_path $CONFIG_PATH "$@"
