#!/bin/bash
#SBATCH  --output=preprocess_logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate anom
python -u data_bern_numpy_to_preprocessed_hdf5.py "$@"