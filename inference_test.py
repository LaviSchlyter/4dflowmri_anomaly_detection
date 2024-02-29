# In this we evaluate the model on the test set and save the results
import matplotlib.pyplot as plt
import os
import h5py
import sys
import glob
import numpy as np
import random
from config import system_eval as config_sys
from helpers.data_loader import load_data
from models.vae import VAE_convT, ConvWithAux, ConvWithEncDecAux

import torch

import logging
logging.basicConfig(level=logging.INFO)

from config import system_eval as eval_config
from sklearn.metrics import roc_auc_score, average_precision_score
import config.system as sys_config


seed = 42  # you can set to your preferred seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Adjusting the plot size
plt.rcParams['figure.figsize'] = [10, 6]

# Adjusting the font size
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Adjusting line width
plt.rcParams['lines.linewidth'] = 1.3
# Adjusting the title size
plt.rcParams['axes.titlesize'] = 16

# Setting a colorblind-friendly color palette
#plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set3.colors)
# Setting a custom color palette
color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)


idx_start_ts = 0
idx_end_ts = 34

models_dir = "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/logs"
data_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/final_segmentations'

def filter_subjects(data_path, experiment_name):
    # Paths to compressed sensing data directories
    cs_controls_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/preprocessed/controls/dicom_compressed_sensing'
    cs_patients_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/preprocessed/patients/dicom_compressed_sensing'

    # Get all subject names and remove the .npy extension
    test_names = [os.path.splitext(name)[0].split('seg_')[1] for name in os.listdir(data_path + '/test')]
    test_names.sort()

    # Get the list of compressed sensing subjects and remove the .npy extension
    cs_subjects_controls = set(os.path.splitext(name)[0] for name in os.listdir(cs_controls_path))
    cs_subjects_patients = set(os.path.splitext(name)[0] for name in os.listdir(cs_patients_path))
    cs_subjects = cs_subjects_controls.union(cs_subjects_patients)

    # Filter subjects based on experiment requirements
    if 'without_cs' in experiment_name:
        filtered_names = [name for name in test_names if name not in cs_subjects]
    elif 'only_cs' in experiment_name:
        filtered_names = [name for name in test_names if name in cs_subjects]
    else:
        filtered_names = test_names

    return filtered_names



############################################## INCORECT VALIDATION SET ##############################################

## WITHOUT UPDATED ASCENDING AORTA ! We should really do it becsaue the centerlines don't look great
## The validation indices are not correct. We removed only without mixing from validation but used the mixed ones as well for training.

"""
list_of_experiments_with_rotation = [
                                    'vae_convT/masked_slice/20231202-1247_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__with_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231130-1151_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__with_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231130-1148_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__with_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231129-2127_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__with_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231129-2131_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__with_rotation_without_cs_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']

list_of_experiments_without_rotation = [
                                        'vae_convT/masked_slice/20231202-1244_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_without_cs_decreased_interpolation_factor_cube_3', 
                                        'vae_convT/masked_slice/20231130-1311_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231130-1153_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231129-2125_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231129-2121_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                        ]
"""

## WITH UPDATED ASCENDING AORTA, WITHOUT ALL SUBJECTS (SET TO ZERO)
# We don't have all the subjects because angle 3/2 did not work for some subjects. Those two subjects are zero basically.
# We ONLY remove the first 2 slices (and use skip)
## Also the validation indices are not correct. We removed only without mixing from validation but used the mixed ones as well for training.

"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20231209-1643_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231209-1615_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231209-1607_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231209-1621_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    ]


name_pre_extension = ['_without_some_data_redid_other_version_']

# Still exists, this is where we have one patient in trainigna nd validation which are zero. 

list_of_experiments_without_rotation = ['vae_convT/masked_slice/20231209-1613_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231209-1624_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

"""


## WITH UPDATED ASCENDING AORTA, WITH ALL SUBJECTS (ANGLE CHANGE), ORDERING WITH ASCENDING AORTA
## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We also reomve alway the first and last 2 slices
## We then order the points in ascending order. 
## --> This did not work well ... But also not tested with the correct test set.... annoying to get the correct test set back
## Also the validation indices are not correct. We removed only without mixing from validation but used the mixed ones as well for training.
"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20231210-1659_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231210-1702_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231210-1706_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231210-1708_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


#name_pre_extension = ['']
# Not available anymore
list_of_experiments_without_rotation = ['vae_convT/masked_slice/20231210-1713_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231210-1712_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231210-1710_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
                                        ]
"""



## WITH UPDATED ASCENDING AORTA, WITH ALL SUBJECTS (ANGLE CHANGE), NO ORDERING WITH ASCENDING AORTA
## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We do not remove the first and last 2 slices - but rather remove the two first only 
## We DO NOT order the points in ascending order. 
## Also the validation indices are not correct. We removed only without mixing from validation but used the mixed ones as well for training.

"""
# We add the purpose fail to make sure it matches with seed 20 -- Spoiler: IT DOESNT FUCKING 
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20231211-2134_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231211-2133_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231211-2131_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231215-1105_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_purpose_false_indices_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231211-2129_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
name_pre_extension = ['']# This is the true thing


list_of_experiments_without_rotation = ['vae_convT/masked_slice/20231211-2124_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231211-2126_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231216-0900_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_purpose_false_indices_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',    
                                        ]
"""

############################################## INCORECT VALIDATION SET ##############################################






############################################## CORRECT VALIDATION SET ##############################################

## 1. WITHOUT UPDATED ASCENDING AORTA ! We should really do it becsaue the centerlines don't look great ----------------------------------------------------------------------- BLUE TABLE

## The validation indices are not correct. We removed only without mixing from validation but used the mixed ones as well for training.
# -----------> HERE we have corrected the indices we remove on the validation set

"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20231217-1536_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__with_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231217-1532_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__with_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231215-1433_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__with_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231217-1542_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__with_rotation_without_cs_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']

list_of_experiments_without_rotation = ['vae_convT/masked_slice/20231217-1538_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231216-0857_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231218-1017_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_without_cs_decreased_interpolation_factor_cube_3']
                                        
"""

## 2. WITH UPDATED ASCENDING AORTA, WITHOUT ALL SUBJECTS (SET TO ZERO) -------------------------------------------------------------------------------------------------------- PURPLE TABLE

# We don't have all the subjects because angle 3/2 did not work for some subjects. Those two subjects are zero basically.
# We ONLY remove the first 2 slices (and use skip)
## Also the validation indices are not correct. We removed only without mixing from validation but used the mixed ones as well for training.
# -----------> HERE we have corrected the indices we remove on the validation set

# Little note: In the first case, I hasn't yet renamed the dataset that's why there is no zero in the name.
"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20231215-1216_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231217-1605_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_some_data_redid_other_version_with_zeros_with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

# True one
#name_pre_extension = ['_without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10', '_without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10',
#                      '_without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10', '_without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10']

# For testing on another test case
#name_pre_extension = ['_with_rotation_without_cs', '_with_rotation_without_cs', '_without_rotation_without_cs', '_without_rotation_without_cs']

name_pre_extension = ['_with_rotation_without_cs_skip_updated_ao_S10', '_with_rotation_without_cs_skip_updated_ao_S10', '_without_rotation_without_cs_skip_updated_ao_S10', '_without_rotation_without_cs_skip_updated_ao_S10']

list_of_experiments_without_rotation = ['vae_convT/masked_slice/20231218-1010_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_some_data_redid_other_version_with_zeros_without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231217-1609_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_some_data_redid_other_version_with_zeros_without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
"""


## 3. WITH UPDATED ASCENDING AORTA, WITH ALL SUBJECTS (ANGLE CHANGE), NO ORDERING WITH ASCENDING AORTA ------------------------------------------------------------------------ YELLOW TABLE

## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We do not remove the first and last 2 slices - but rather remove the two first only 
## We DO NOT order the points in ascending order
# -----------> HERE we have corrected the indices we remove on the validation set

"""
list_of_experiments_with_rotation = [
                                    'vae_convT/masked_slice/20231215-1034_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231214-1948_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231213-2247_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231214-1945_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
name_pre_extension = ['']# This is the true thing


name_pre_extension = ['_without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10', '_without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10', '_without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10','_without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10', 
                      '_without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10', '_without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10', '_without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10', '_without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10']

list_of_experiments_without_rotation = [
                                        'vae_convT/masked_slice/20231215-1032_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231214-1951_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231213-2252_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231214-1923_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',]

"""


## 4. WITH UPDATED ASCENDING AORTA, WITHOUT ALL SUBJECTS (WE REMOVED THEM) ------------------------------------------------------------------------------------------------------ RED TABLE

## Also the validation indices are not correct. We removed only without mixing from validation but used the mixed ones as well for training.

"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20231217-1709_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231213-2224_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231213-2230_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     ]
#name_pre_extension = ['_with_rotation_without_cs_skip_updated_ao_S10',
#                      '_with_rotation_without_cs_skip_updated_ao_S10',
#                      '_without_rotation_without_cs_skip_updated_ao_S10',
#                      '_without_rotation_without_cs_skip_updated_ao_S10'] # When you want to test on other test case.
name_pre_extension = ['']
# No need to add extension, it is present in the name of the experiment.    

list_of_experiments_without_rotation = ['vae_convT/masked_slice/20231217-1702_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231213-2227_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231213-2229_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


"""
"""

## 5. WITH UPDATED ASCENDING AORTA, WITH POISSON MIX FOR TRAINING,  WITH ALL SUBJECTS (ANGLE CHANGE), NO ORDERING WITH ASCENDING AORTA ------------------------------------------ GRAY TABLE
## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We do not remove the first and last 2 slices - but rather remove the two first only 
## We DO NOT order the points in ascending order
# -----------> HERE we have corrected the indices we remove on the validation set
# POisson mix for training 

list_of_experiments_with_rotation = [
                                    'vae_convT/masked_slice/20231220-2300_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231220-2256_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231215-1144_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231215-1109_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231219-1926_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231219-1930_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
                                    
name_pre_extension = [''] # This is the true thing

list_of_experiments_without_rotation = [
                                        'vae_convT/masked_slice/20231220-2302_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231220-2305_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231215-1150_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231215-1139_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231218-1031_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        ]

"""
## 6. WITH UPDATED ASCENDING AORTA, WITH AND WITHOUT GRADIENT MIX POISSON FOR TRAINING BUT ALSO VALIDATION,  WITH ALL SUBJECTS (ANGLE CHANGE), NO ORDERING WITH ASCENDING AORTA ------------------------------- PINK TABLE
## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We do not remove the first and last 2 slices - but rather remove the two first only 
## We DO NOT order the points in ascending order
# -----------> HERE we have either ways no indices
# Training with poisson mix adn wihtout mix and validation too - no removing indices


"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20231221-1840_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_all_poisson_tr_val_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231221-1811_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_all_poisson_tr_val_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231221-1757_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_all_poisson_tr_val_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231221-1752_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_all_poisson_tr_val_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231221-1750_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_all_poisson_tr_val_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
                                    
name_pre_extension = [''] # This is the true thing

list_of_experiments_without_rotation = ['vae_convT/masked_slice/20231220-2317_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_all_poisson_tr_val_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231220-2321_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_all_poisson_tr_val_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231220-2325_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_all_poisson_tr_val_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231220-2329_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_all_poisson_tr_val_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231228-1203_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_all_poisson_tr_val_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
                                        ]
"""



## 7. RECONSTRUCTION BASEDWITH UPDATED ASCENDING AORTA - WITH ALL SUBJECTS (ANGLE CHANGE), NO ORDERING WITH ASCENDING AORTA ------------------------------------------ GRAY TABLE
## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We do not remove the first and last 2 slices - but rather remove the two first only 
## We DO NOT order the points in ascending order


"""

list_of_experiments_with_rotation = [
                                        'vae_convT/masked_slice/20240108-1859_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_5_with_rotation',
                                        'vae_convT/masked_slice/20240108-1853_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_10_with_rotation',
                                        'vae_convT/masked_slice/20240108-1857_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_15_with_rotation',
                                        'vae_convT/masked_slice/20240108-1600_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_20_with_rotation',
                                        'vae_convT/masked_slice/20240108-1558_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_25_with_rotation',
                                        ]



name_pre_extension = ['_with_rotation_without_cs_skip_updated_ao_S10','_with_rotation_without_cs_skip_updated_ao_S10','_with_rotation_without_cs_skip_updated_ao_S10','_with_rotation_without_cs_skip_updated_ao_S10','_with_rotation_without_cs_skip_updated_ao_S10',
                    '_without_rotation_without_cs_skip_updated_ao_S10','_without_rotation_without_cs_skip_updated_ao_S10','_without_rotation_without_cs_skip_updated_ao_S10','_without_rotation_without_cs_skip_updated_ao_S10','_without_rotation_without_cs_skip_updated_ao_S10'] # This is the true thing

list_of_experiments_without_rotation = [
                                    'vae_convT/masked_slice/20240108-1118_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_5_without_rotation',
                                    'vae_convT/masked_slice/20240108-1120_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_10_without_rotation',
                                    'vae_convT/masked_slice/20240108-1123_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_15_without_rotation',
                                    'vae_convT/masked_slice/20240108-1521_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_20_without_rotation',
                                    'vae_convT/masked_slice/20240108-1523_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_25_without_rotation'
                                    ]
                                    
"""
idx_start_ts = 0
idx_end_ts = 34


############################################## ONLY COMPRESSED SENSING ##############################################
## 7. ONLY COMPRESSED SENSING - WITH UPDATED ASCENDING AORTA - WITH ALL SUBJECTS (ANGLE CHANGE), NO ORDERING WITH ASCENDING AORTA ------------------------------------------ 
## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We do not remove the first and last 2 slices - but rather remove the two first only 
## We DO NOT order the points in ascending order - ONLY COMPRESSED SENSING

"""
list_of_experiments_with_rotation = ['conv_with_aux/masked_slice/20240212-1036_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240212-1040_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240212-1119_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240212-1558_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240212-1613_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


idx_start_ts = 0
idx_end_ts = 17

name_pre_extension = ['']


list_of_experiments_without_rotation = ['conv_with_aux/masked_slice/20240212-1033_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240212-1039_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240212-1053_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240212-1557_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240212-1600_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']



"""

"""
list_of_experiments_with_rotation = ['conv_with_aux/masked_slice/20240213-1552_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240213-1433_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240213-1017_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240213-0947_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240212-1718_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


idx_start_ts = 0
idx_end_ts = 17

name_pre_extension = ['']


list_of_experiments_without_rotation = ['conv_with_aux/masked_slice/20240213-1538_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240213-1454_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240213-1018_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240213-1011_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240212-1720_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


"""

"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20240115-1928_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240115-1939_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240115-1945_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240115-1949_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240115-1952_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = ['vae_convT/masked_slice/20240116-1100_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240116-1058_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240116-1055_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240116-1053_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240116-1050_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


"""


#############

# THE SEEDS HAVE BEEN FIXED


"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20240214-1610_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240214-1613_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240214-1615_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240214-1617_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240214-1622_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = ['vae_convT/masked_slice/20240215-0858_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240215-0901_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240215-0906_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240215-0910_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240215-0914_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

"""

"""
list_of_experiments_with_rotation =    ['conv_with_aux/masked_slice/20240214-1119_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1120_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1123_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1125_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1128_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


idx_start_ts = 0
idx_end_ts = 17

name_pre_extension = ['']


list_of_experiments_without_rotation = ['conv_with_aux/masked_slice/20240214-1214_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1216_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1217_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1219_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1433_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']



"""
"""

list_of_experiments_with_rotation =    ['conv_enc_dec_aux/masked_slice/20240215-0930_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240215-0957_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240218-1218_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240218-1221_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240218-1223_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


idx_start_ts = 0
idx_end_ts = 17

name_pre_extension = ['']


list_of_experiments_without_rotation = ['conv_enc_dec_aux/masked_slice/20240218-1225_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240218-1226_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240227-1428_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240227-1429_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240227-1433_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


"""


#############

######################################### ALL DATA (includes compressed sensing) ##########################################
## 8. ALL DATA - WITH UPDATED ASCENDING AORTA - WITH ALL SUBJECTS (ANGLE CHANGE), NO ORDERING WITH ASCENDING AORTA ------------------------------------------
## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We do not remove the first and last 2 slices - but rather remove the two first only
## We DO NOT order the points in ascending order
"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20240116-1803_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240116-1808_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240116-1830_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240117-0835_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240118-1021_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 54


list_of_experiments_without_rotation = ['vae_convT/masked_slice/20240118-1028_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240118-1040_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240118-1108_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240119-1757_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240119-1812_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


"""

#############

# THE SEEDS HAVE BEEN FIXED - ALL DATA
"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20240218-1233_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240218-1236_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240218-1237_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240218-1240_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240218-1242_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 54


list_of_experiments_without_rotation = ['vae_convT/masked_slice/20240225-0800_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240225-0803_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240225-0808_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240225-0811_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240225-0828_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
"""


"""
list_of_experiments_with_rotation = ['conv_with_aux/masked_slice/20240225-0833_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_with_aux/masked_slice/20240225-0838_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_with_aux/masked_slice/20240225-2241_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_with_aux/masked_slice/20240225-2243_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_with_aux/masked_slice/20240225-2300_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 54


list_of_experiments_without_rotation = ['conv_with_aux/masked_slice/20240225-2310_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240225-2316_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240225-2318_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240225-2320_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240225-2322_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

"""

list_of_experiments_with_rotation = ['conv_enc_dec_aux/masked_slice/20240226-1739_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_enc_dec_aux/masked_slice/20240226-1741_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_enc_dec_aux/masked_slice/20240226-1746_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_enc_dec_aux/masked_slice/20240226-1749_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_enc_dec_aux/masked_slice/20240226-1834_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 54


list_of_experiments_without_rotation = ['conv_enc_dec_aux/masked_slice/20240226-1837_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240226-1845_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240226-1846_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240227-1414_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240227-1417_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']



##############

############################################## CORRECT VALIDATION SET ##############################################
list_of_experiments_paths = list_of_experiments_with_rotation + list_of_experiments_without_rotation

logging.info('list_of_experiments_paths: {}'.format(list_of_experiments_paths))

if __name__ == '__main__':
    adjacent_batch_slices = None
    batch_size = 32 
    
    
    for i, exp_rel_path in enumerate(list_of_experiments_paths):
        logging.info('============================================================')
        logging.info('Processing model: {}'.format(exp_rel_path))
        logging.info('============================================================')
        exp_path = os.path.join(models_dir, exp_rel_path)
        

        pattern = os.path.join(exp_path, "*best*")
        
        best_exp_path = glob.glob(pattern)[0]
        

        model_str = exp_rel_path.split("/")[0]
        
        preprocess_method = exp_rel_path.split("/")[1]
        exp_name = exp_rel_path.split("/")[-1]
        logging.info('name pre extension: {}'.format(name_pre_extension))


        # Check if self-supervised or reconstruction based
        if exp_rel_path.__contains__('SSL'):
            model_type = 'self-supervised'
            in_channels = 4
            out_channels =1 
        else:
            model_type = 'reconstruction-based'
            in_channels = 4
            out_channels = 4

        

        # Check if name_pre_extension is non empty
        if len(name_pre_extension) > 1:
            name_extension = name_pre_extension[i]
        else:
            name_pre = name_pre_extension[0]
            name_extension = name_pre + exp_name.split('2Dslice_')[1].split('_decreased_interpolation_factor_cube_3')[0]
        
        
        
        # Excepetionally else one above
        #name_extension = name_pre_extension[i]
        logging.info('name_extension: {}'.format(name_extension))

        if exp_name.__contains__('vae_convT'):
            model = VAE_convT(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
        elif exp_name.__contains__('conv_with_aux'):
            model = ConvWithAux(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
        elif exp_name.__contains__('conv_enc_dec'):
            model = ConvWithEncDecAux(in_channels=in_channels, out_channels=out_channels, gf_dim=8)

        else:
            raise ValueError('Exp name {} has no model recognized'.format(exp_name))
        # Load the model onto device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(best_exp_path, map_location=device))
        model.to(device)
        model.eval()
        

        healthy_scores = []
        anomalous_scores = []
        healthy_idx = 0
        anomalous_idx = 0
        spatial_size_z = 64
        preprocess_method = exp_rel_path.split("/")[1]
        config = {'preprocess_method': preprocess_method}

        
        data_dict = load_data(sys_config=config_sys, config=config, idx_start_tr=0, idx_end_tr=1, idx_start_vl=0, idx_end_vl=1,idx_start_ts=idx_start_ts, idx_end_ts=idx_end_ts, with_test_labels= True, suffix = name_extension)
        images_test = data_dict['images_test']
        labels_test = data_dict['labels_test']
        rotation_matrix = data_dict.get('rotation_test', None)
        
        # Get the names of the test subjects
        test_names = filter_subjects(data_path, exp_rel_path)

        logging.info('============================================================')
        # Print test names
        logging.info('Test subjects: {}'.format(test_names))
        logging.info('============================================================')



        subject_indexes = range(np.int16(images_test.shape[0]/spatial_size_z))

        for subject_idx in subject_indexes:
            logging.info('Processing subject {}'.format(test_names[subject_idx]))
            start_idx = 0
            end_idx = batch_size
            
            
            
            subject_anomaly_score = []
            subject_reconstruction = []
            subject_sliced = images_test[subject_idx*spatial_size_z:(subject_idx+1)*spatial_size_z]
            subject_labels = labels_test[subject_idx*spatial_size_z:(subject_idx+1)*spatial_size_z]
            if rotation_matrix is not None:
                rotation_matrix_subject = rotation_matrix[subject_idx*spatial_size_z:(subject_idx+1)*spatial_size_z]
            while end_idx <= spatial_size_z:
                batch = subject_sliced[start_idx:end_idx]
                labels = subject_labels[start_idx:end_idx]
                batch_z_slice = torch.from_numpy(np.arange(start_idx, end_idx)).float().to(device)
                batch = torch.from_numpy(batch).transpose(1,4).transpose(2,4).transpose(3,4).float().to(device)
                rotation_matrix_batch = None
                if rotation_matrix is not None:
                    rotation_matrix_batch = rotation_matrix_subject[start_idx:end_idx]
                    rotation_matrix_batch = torch.from_numpy(rotation_matrix_batch).to(device).float()
                with torch.no_grad():
                    model.eval()
                    
                    input_dict = {'input_images': batch, 'batch_z_slice':batch_z_slice, 'adjacent_batch_slices':adjacent_batch_slices,'rotation_matrix': rotation_matrix_batch}
                    output_dict = model(input_dict)

                    output_images = torch.sigmoid(output_dict['decoder_output'])
                    # Compute anomaly score
                    subject_anomaly_score.append(output_images.cpu().detach().numpy())
                    # Check if all labels are anomalous
                    if np.all(labels == 0):
                        legend = "healthy"
                    elif np.all(labels == 1):
                        legend = "anomalous"
                    else:
                        raise ValueError("Labels are not all healthy or all anomalous, change batch size")
                start_idx += batch_size
                end_idx += batch_size
            if legend == "healthy":
                healthy_scores.append(np.concatenate(subject_anomaly_score))
                healthy_idx += 1
            else:
                anomalous_scores.append(np.concatenate(subject_anomaly_score))
                anomalous_idx += 1

            logging.info('{}_subject {} anomaly_score: {:.4e} +/- {:.4e}'.format(legend, subject_idx, np.mean(subject_anomaly_score), np.std(subject_anomaly_score)))

        healthy_scores = np.array(healthy_scores)
        healthy_mean_anomaly_score = np.mean(healthy_scores)
        healthy_std_anomaly_score = np.std(healthy_scores)

        anomalous_scores = np.array(anomalous_scores)
        anomalous_mean_anomaly_score = np.mean(anomalous_scores)
        anomalous_std_anomaly_score = np.std(anomalous_scores)

        logging.info('============================================================')
        logging.info('Control subjects anomaly_score: {} +/- {:.4e}'.format(healthy_mean_anomaly_score, healthy_std_anomaly_score))
        logging.info('Anomalous subjects anomaly_score: {} +/- {:.4e}'.format(anomalous_mean_anomaly_score, anomalous_std_anomaly_score))
        logging.info('============================================================')


        # Compute AUC-ROC
        anomalous_scores_patient = np.mean(anomalous_scores, axis=(1,2,3,4,5))
        healthy_scores_patient = np.mean(healthy_scores, axis=(1,2,3,4,5))
        
        y_true = np.concatenate((np.zeros(len(healthy_scores_patient.flatten())), np.ones(len(anomalous_scores_patient.flatten()))))
        y_scores = np.concatenate((healthy_scores_patient.flatten(), anomalous_scores_patient.flatten()))
        auc_roc = roc_auc_score(y_true, y_scores)
        logging.info('AUC-ROC: {:.2f}'.format(auc_roc))

        # After AUC-ROC is computed, compute the average precision score
        # Compute AUC-PR
        auc_pr = average_precision_score(y_true, y_scores)
        logging.info('AUC-PR: {:.2f}'.format(auc_pr))
        

        



