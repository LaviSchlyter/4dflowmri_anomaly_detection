# In this file we evaluate the best model on the test set and visualize the results
import numpy as np

import matplotlib.pyplot as plt


import os
import re
import sys
import torch
import h5py
import glob
import random
import logging
import seaborn as sns
import pandas as pd
import datetime
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore", message="All-NaN axis encountered")
import random

from helpers.utils import make_dir_safely
from config import system_eval as config_sys
from helpers.data_loader import load_data, load_syntetic_data



# Import models
from models.vae import VAE, VAE_convT, VAE_linear
from models.condconv import CondVAE, CondConv
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/helpers')
from batches import plot_batch_3d_complete, plot_batch_3d_complete_1_chan


# Download the methods for generating synthetic data
from helpers.synthetic_anomalies import generate_noise, create_cube_mask,\
                         generate_deformation_chan, create_hollow_noise


from helpers.utils_inference import plot_scores, return_indexes_to_remove, get_random_and_neighbour_indices, \
    statistical_tests, plot_slices, compute_average_precision, compute_auc, validation_metrics, get_combined_images
# For the patch blending we import from another directory
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/git_repos/many-tasks-make-light-work')
from multitask_method.tasks.patch_blending_task import TestPatchInterpolationBlender, \
    TestPoissonImageEditingMixedGradBlender, TestPoissonImageEditingSourceGradBlender

from multitask_method.tasks.labelling import FlippedGaussianLabeller


labeller = FlippedGaussianLabeller(0.2)


from helpers.visualization import plot_batches_SSL, plot_batches_SSL_in_out



seed = 0  # you can set to your preferred seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False





if __name__ == '__main__':

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
    
    models_dir = "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/logs"
    list_of_experiments_run1 = [
    'cond_vae/masked_slice/20230722-0851_cond_vae_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-n_experts3_2Dslice_without_noise_cube_3',
    'cond_vae/masked_slice/20230721-1938_cond_vae_masked_slice_SSL_lr1.800e-03_scheduler-e1500-bs8-gf_dim8-daFalse-n_experts3_with_interpolation_training_2Dslice_without_noise_cube_3',
    'cond_vae/masked_slice/20230721-1021_cond_vae_masked_slice_SSL_lr1.800e-03_scheduler-e1500-bs8-gf_dim8-daFalse-n_experts3_with_interpolation_training_2Dslice_decreased_interpolation_factor_cube_3',
    'cond_vae/masked_slice/20230721-1018_cond_vae_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-n_experts3_with_interpolation_training_2Dslice_decreased_interpolation_factor_cube_3',
    'cond_vae/masked_slice/20230721-1011_cond_vae_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-n_experts3_reproducibility_2Dslice_decreased_interpolation_factor_cube_3',
    'cond_vae/masked_slice/20230721-1008_cond_vae_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-n_experts3_2Dslice_decreased_interpolation_factor_cube_3',
    'cond_vae/masked_slice/20230721-1002_cond_vae_masked_slice_SSL_lr1.800e-03_scheduler-e1500-bs8-gf_dim8-daFalse-n_experts3_2Dslice_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20230721-0835_vae_convT_masked_slice_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse-f100',
    'vae_convT/masked_slice/20230720-1837_vae_convT_masked_slice_SSL_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse_2Dslice_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20230720-1729_vae_convT_masked_slice_SSL_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse_with_interpolation_training_2Dslice_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20230720-1726_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse_with_interpolation_training_2Dslice_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20230720-0851_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse_poisson_mix_training_2Dslice_without_noise_cube_3',
    'vae_convT/masked_slice/20230720-0847_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse_2Dslice_without_noise_cube_3',
    'vae_convT/masked_slice/20230720-0841_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse_2Dslice',
    'vae_convT/masked_slice/20230720-0834_vae_convT_masked_slice_SSL_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse_poisson_mix_training_2Dslice_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20230719-1850_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse_poisson_mix_training_2Dslice_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20230719-1840_vae_convT_masked_slice_SSL_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse_2Dslice_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20230719-1833_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse_2Dslice_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20230719-1830_vae_convT_masked_slice_lr1.000e-03-e3000-bs8-gf_dim8-daFalse-f100',
    'vae_convT/masked_slice/20230719-1828_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100_2Dslice_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20230719-1826_vae_convT_masked_slice_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse-f100_2Dslice_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20230719-1823_vae_convT_masked_slice_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse-f100',
    'vae_convT/masked_slice/20230719-1821_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100'
]

    list_of_best_experiments_run1 = ["vae_convT/masked_slice/20230719-1823_vae_convT_masked_slice_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse-f100",
                                     "vae_convT/masked_slice/20230719-1840_vae_convT_masked_slice_SSL_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse_2Dslice_decreased_interpolation_factor_cube_3",
                                     "cond_vae/masked_slice/20230721-1002_cond_vae_masked_slice_SSL_lr1.800e-03_scheduler-e1500-bs8-gf_dim8-daFalse-n_experts3_2Dslice_decreased_interpolation_factor_cube_3"]
    

    list_of_experiments_seeds = [
    'vae_convT/masked_slice/20230729-1647_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice_decreased_interpolation_factor_cube_3',
    'cond_vae/masked_slice/20230729-1010_cond_vae_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-n_experts3__SEED_25_2Dslice_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20230729-0959_vae_convT_masked_slice_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse-f100__SEED_25',
    'vae_convT/masked_slice/20230729-0153_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20230728-2259_vae_convT_masked_slice_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse-f100__SEED_20',
    'cond_vae/masked_slice/20230728-1239_cond_vae_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-n_experts3__SEED_20_2Dslice_decreased_interpolation_factor_cube_3',
    'cond_vae/masked_slice/20230728-1236_cond_vae_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100-n_experts3__SEED_25',
    'cond_vae/masked_slice/20230728-1220_cond_vae_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100-n_experts3__SEED_20',
    'cond_vae/masked_slice/20230728-1217_cond_vae_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100-n_experts3__SEED_15',
    'cond_vae/masked_slice/20230728-1214_cond_vae_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100-n_experts3__SEED_10',
    'cond_vae/masked_slice/20230728-1211_cond_vae_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100-n_experts3__SEED_5',
    'vae_convT/masked_slice/20230727-0855_vae_convT_masked_slice_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse-f100__SEED_15',
    'vae_convT/masked_slice/20230727-0846_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice_decreased_interpolation_factor_cube_3',
    'cond_vae/masked_slice/20230727-0153_cond_vae_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-n_experts3__SEED_15_2Dslice_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20230726-0849_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice_decreased_interpolation_factor_cube_3',
    'cond_vae/masked_slice/20230726-0825_cond_vae_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-n_experts3__SEED_10_2Dslice_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20230726-0825_vae_convT_masked_slice_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse-f100__SEED_10',
    'vae_convT/masked_slice/20230726-0818_vae_convT_masked_slice_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse-f100__SEED_5',
    'vae_convT/masked_slice/20230725-2010_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice_decreased_interpolation_factor_cube_3',
    'cond_vae/masked_slice/20230725-2010_cond_vae_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-n_experts3__SEED_5_2Dslice_decreased_interpolation_factor_cube_3'
]
    one_run = ["vae_convT/masked_slice/20230719-1823_vae_convT_masked_slice_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse-f100"]

    list_of_experiments_paths = one_run

    

    # ========================================================  
    # ==================== LOGGING CONFIG ====================
    # ========================================================

    project_data_root = config_sys.project_data_root
    project_code_root = config_sys.project_code_root

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    
    # Pick same random images
    keep_same_indices = True
    # Batch size
    batch_size = 16
    adjacent_batch_slices = None
    actions = ["visualize"] # ["visualize", ]

    data_to_visualize = ["test"]  # ["validation", "test"] 

    # Load the data
    data_dir = '/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/data'

    # We received new batch in July 2023, we added 14 subjects to the test data 9 control and 5 cases
    # if we select idx_end_ts =20 then we look at the first batch recevied with 7 cases and 13 controls\
    #idx_end_ts = 34
    idx_end_ts = 34
    
    # Please note the test labels are just a mask of ones for sick patients and zero for healthy
    
    
    deformation_list = ['None', 'noisy', 'deformation', 'hollow_circle', 'patch_interpolation', 'poisson_with_mixing', 'poisson_without_mixing'] 
    # Create dictionary to map the current test subjects to their ID
    test_subjects_dict = {0: "MACDAVD_201_", 1: "MACDAVD_202_", 2: "MACDAVD_203_", 3: "MACDAVD_301_", 4: "MACDAVD_302_", 5: "MACDAVD_303_", 6: "MACDAVD_304_",
                          7: "MACDAVD_125_", 8: "MACDAVD_127_", 9: "MACDAVD_129_", 10: "MACDAVD_131_", 11: "MACDAVD_133_", 12: "MACDAVD_139_", 13: "MACDAVD_141_",
                          14: "MACDAVD_143_", 15: "MACDAVD_145_", 16: "MACDAVD_121_", 17:"MACDAVD_137_", 18: "MACDAVD_123_", 19: "MACDAVD_135_", 20: "MACDAVD_206_",
                          21: "MACDAVD_208_", 22: "MACDAVD_209_", 23: "MACDAVD_311_", 24:"MACDAVD_404_", 25:"MACDAVD_156_", 26: "MACDAVD_157_", 27: "MACDAVD_158_",
                          28: "MACDAVD_159_", 29: "MACDAVD_160_", 30: "MACDAVD_161_",31: "MACDAVD_163_", 32: "MACDAVD_164_", 33: "MACDAVD_165_"}

    #noise_generator = generate_noise(batch_size = batch_size,mean_range = (0, 0.1), std_range = (0, 0.2), n_samples= images_healthy_unseen.shape[0]) 

    for model_rel_path in list_of_experiments_paths:

        model_path = os.path.join(models_dir, model_rel_path)
        pattern = os.path.join(model_path, "*best*")
        best_model_path = glob.glob(pattern)[0]

        model_str = model_rel_path.split("/")[0]
        preprocess_method = model_rel_path.split("/")[1]
        model_name = model_rel_path.split("/")[-1]
        # Several things
        #1. Load the correct synthetic dataset
        #2. Get the appropriate deformation list
        #3. Remove indices which were used for training

        # We need to adapt the synthetic validation data based on preprocess_method as well as the spatial z and the model used
        if model_name.__contains__('without_noise_cube_3'):
            synthetic_data_note = 'without_noise_cube_3'
            data_vl = load_syntetic_data(preprocess_method =preprocess_method, idx_start=35, idx_end=42, sys_config = config_sys, note = synthetic_data_note)
            images_vl  = data_vl['images']
            labels_vl = data_vl['masks']
            deformation_list_validation_set = ['None','deformation', 'patch_interpolation', 'poisson_with_mixing', 'poisson_without_mixing']
            validation_indexes, deformation_list_validation_set = return_indexes_to_remove(model_name, deformation_list_validation_set, images_vl, with_noise = False)
        elif model_name.__contains__('without_noise'):
            synthetic_data_note = 'without_noise'
            data_vl = load_syntetic_data(preprocess_method =preprocess_method, idx_start=35, idx_end=42, sys_config = config_sys, note = synthetic_data_note)
            images_vl  = data_vl['images']
            labels_vl = data_vl['masks']
            deformation_list_validation_set = ['None','deformation', 'patch_interpolation', 'poisson_with_mixing', 'poisson_without_mixing']
            validation_indexes, deformation_list_validation_set = return_indexes_to_remove(model_name, deformation_list_validation_set, images_vl, with_noise = False)
        elif model_name.__contains__('decreased_interpolation_factor_cube_3'):
            synthetic_data_note = 'decreased_interpolation_factor_cube_3'
            deformation_list_validation_set = ['None', 'noisy', 'deformation', 'hollow_circle', 'patch_interpolation', 'poisson_with_mixing', 'poisson_without_mixing']
            data_vl = load_syntetic_data(preprocess_method =preprocess_method, idx_start=35, idx_end=42, sys_config = config_sys, note = synthetic_data_note)
            images_vl  = data_vl['images']
            labels_vl = data_vl['masks']
            validation_indexes, deformation_list_validation_set = return_indexes_to_remove(model_name, deformation_list_validation_set, images_vl, with_noise = False)
        elif model_name.__contains__('decreased_interpolation_factor'):
            synthetic_data_note = 'decreased_interpolation_factor'
            deformation_list_validation_set = ['None', 'noisy', 'deformation', 'hollow_circle', 'patch_interpolation', 'poisson_with_mixing', 'poisson_without_mixing']
            data_vl = load_syntetic_data(preprocess_method =preprocess_method, idx_start=35, idx_end=42, sys_config = config_sys, note = synthetic_data_note)
            images_vl  = data_vl['images']
            labels_vl = data_vl['masks']
            validation_indexes, deformation_list_validation_set = return_indexes_to_remove(model_name, deformation_list_validation_set, images_vl, with_noise = False)
        else:
            synthetic_data_note = ''
            deformation_list_validation_set = ['None', 'noisy', 'deformation', 'hollow_circle', 'patch_interpolation', 'poisson_with_mixing', 'poisson_without_mixing']
            
            data_vl = load_syntetic_data(preprocess_method =preprocess_method, idx_start=35, idx_end=42, sys_config = config_sys, note = synthetic_data_note)
            images_vl  = data_vl['images']
            labels_vl = data_vl['masks']
            validation_indexes, deformation_list_validation_set = return_indexes_to_remove(model_name, deformation_list_validation_set, images_vl, with_noise = False)
        config = {'preprocess_method': preprocess_method}
        if preprocess_method.__contains__('full_aorta'):
            spatial_size_z = 256
        else:
            spatial_size_z = 64


        
        if idx_end_ts == 20:
            note = 'initial_batch'
            logging.info(f"----------------------- USING INITIAL BATCH -----------------------")

        # Please note the test labels are just a mask of ones for sick patients and zero for healthy
        _, _, images_test, labels_test = load_data(sys_config=config_sys, config=config, idx_start_vl=35, idx_end_vl=42,idx_start_ts=0, idx_end_ts=idx_end_ts, with_test_labels= True)

        

        # ========================================================
        # Logging the shapes
        logging.info(f"Validation images shape: {images_vl.shape}")
        logging.info(f"Validation labels shape: {labels_vl.shape}")
        logging.info(f"Test images shape: {images_test.shape}")
        logging.info(f"Test labels shape: {labels_test.shape}")
                
        

        self_supervised = True if "SSL" in model_name else False
        config['self_supervised'] = self_supervised

        if model_str == "vae":
            if self_supervised:
                model = VAE(in_channels=4, out_channels=1, gf_dim=8)
            else:
                model = VAE(in_channels=4, out_channels=4, gf_dim=8)
        elif model_str == "vae_convT":
            if self_supervised:
                model = VAE_convT(in_channels=4, out_channels=1, gf_dim=8)
            else:
                model = VAE_convT(in_channels=4, out_channels=4, gf_dim=8)
        elif model_str == "cond_vae":
            match = re.search(r'n_experts(\d+)', model_rel_path)
            num_experts = int(match.group(1))
            if self_supervised:
                model = CondVAE(in_channels=4, out_channels=1, gf_dim=8, num_experts=num_experts)
            else:
                model = CondVAE(in_channels=4, out_channels=4, gf_dim=8, num_experts=num_experts)
        elif model_str == "cond_conv":
            match = re.search(r'n_experts(\d+)', model_rel_path)
            num_experts = int(match.group(1))
            if self_supervised:
                model = CondConv(in_channels=4, out_channels=1, gf_dim=8, num_experts=num_experts)
            else:
                model = CondConv(in_channels=4, out_channels=4, gf_dim=8, num_experts=num_experts)
        else:
            raise ValueError("Model not recognized")
        
        # Load the model onto device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        model.eval()
        
        results_dir = os.path.join(project_code_root,'Results/Evaluation/' + model_str + '/' + preprocess_method + '/' + model_name)
        
        make_dir_safely(results_dir)
        log_file = os.path.join(results_dir, 'log.txt')
        open(log_file, 'w').close()

        # Clear the root loggers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w')
        logger = logging.getLogger()
        #logging.basicConfig(filename=log_file, level=logging.INFO)
        #logging.getLogger().addHandler(logging.FileHandler(log_file))
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # add console handler to logger
        logger.addHandler(ch)
        logging.info('============================================================')
        logging.info(f'Logging results for {model_name}')
        logging.info(f'Logging results in {log_file}')
        logging.info('============================================================')
        logging.info(f"device used: {device}")
        

        # You'll need to load data here since it depends on the preprocessing method
        
        # ========================================================
        # ==================== LOAD DATA ========================
        # ========================================================
        

        # Write summary of important parameters to log file
        logging.info('============================================================')
        logging.info('SUMMARY OF IMPORTANT PARAMETERS')
        logging.info('============================================================')
        logging.info(f"Model: {model_str}")
        logging.info(f"Preprocessing method: {preprocess_method}")
        logging.info(f"Model name: {model_name}")
        logging.info(f"Model loaded from {best_model_path}")
        


        if "validation" in data_to_visualize:
            # We have 7 patients in the validation set with 7 (or less) deformations and 64 slices each
            
            n_def = len(deformation_list_validation_set)

            n_val_patients = 7
            results_dir_val = results_dir + '/' + 'validation'
            make_dir_safely(results_dir_val)
            all_anomaly_scores = []
            all_masks = []
            batch_size = 32

            for i in range(n_def*n_val_patients):
                # Current deformation
                if i%n_val_patients == 0:
                    logging.info(f"******Deformation: {deformation_list_validation_set[i//n_val_patients]} ******")
                results_dir_def = os.path.join(results_dir_val, deformation_list_validation_set[i//n_val_patients])
                make_dir_safely(results_dir_def)
                
                #images_def = images_vl[i*spatial_size_z:(i+1)*spatial_size_z]
                #labels_def = labels_vl[i*spatial_size_z:(i+1)*spatial_size_z]
                images_def = images_vl[validation_indexes[i*spatial_size_z:(i+1)*spatial_size_z]]
                labels_def = labels_vl[validation_indexes[i*spatial_size_z:(i+1)*spatial_size_z]]
                start_idx = 0
                end_idx = batch_size
                subject_anomaly_score = []
                subject_masks = [] # Not necessary but easier to reuse metrics functions
                subject_idx = i%n_val_patients
                subject_batch = []
                while end_idx <= spatial_size_z:
                    # If get neighbour slices
                    if model_str.__contains__('cond_conv'):
                        # Then we are using the neighbour slices as well for the routing function
                        neighbour_indices = get_random_and_neighbour_indices(np.arange(start_idx, end_idx), subject_length=spatial_size_z)
                        adjacent_images = get_combined_images(images_def, neighbour_indices)
                        # The adjacent images are of size [b,3,32,32,24,4] for ascending aorta
                        # We want to have the channel in the second dimension and add the new slices dimenison into the channel dimension
                        adjacent_batch_slices = torch.from_numpy(adjacent_images).to(device).transpose(1,5).transpose(2,5).transpose(3,5).transpose(4,5).float()
                        adjacent_batch_slices = adjacent_batch_slices.reshape(adjacent_batch_slices.shape[0], -1, adjacent_batch_slices.shape[3], adjacent_batch_slices.shape[4], adjacent_batch_slices.shape[5])
                        # size [b,c*3, 32,32,24]
                    batch = images_def[start_idx:end_idx]
                    labels = labels_def[start_idx:end_idx]
                    

                    # Select indices, convert to torch and send to device
                    batch = torch.from_numpy(batch).transpose(1,4).transpose(2,4).transpose(3,4).float().to(device)
                    labels = torch.from_numpy(labels).transpose(1,4).transpose(2,4).transpose(3,4).float().to(device)
                    batch_z_slice = torch.from_numpy(np.arange(start_idx, end_idx)).float().to(device)
                    with torch.no_grad():
                        model.eval()
                        
                        input_dict = {'input_images': batch, 'batch_z_slice':batch_z_slice, "adjacent_batch_slices": adjacent_batch_slices}
                        output_dict = model(input_dict)
                        if self_supervised:
                            output_images = torch.sigmoid(output_dict['decoder_output'])
                        else:
                            # Reconstruction based
                            output_images = torch.abs(output_dict['decoder_output'] - batch)
                        # Compute anomaly score
                        subject_anomaly_score.append(output_images.cpu().detach().numpy())
                        subject_masks.append(labels.cpu().detach().numpy())
                    output_images = output_images.transpose(1,2).transpose(2,3).transpose(3,4).cpu().detach().numpy()
                    labels = labels.transpose(1,2).transpose(2,3).transpose(3,4).cpu().detach().numpy()
                    batch = batch.transpose(1,2).transpose(2,3).transpose(3,4).cpu().detach().numpy()
                    subject_batch.append(batch)

                    # Check if actions include visualization
                    if "visualize" in actions:
                        plot_batches_SSL(batch, output_images, labels, channel_to_show = 1,every_x_time_step=1, out_path=os.path.join(results_dir_def, 'subject_{}_batch_{}_to_{}.png'.format(subject_idx,start_idx, end_idx)))
                    start_idx += batch_size
                    end_idx += batch_size
                    # End of subject 
                    # Log subject mean and std anomaly score
                    
                logging.info('Subject {} anomaly_score: {:.4e} +/- {:.4e}'.format(subject_idx,np.mean(np.concatenate(subject_anomaly_score)), np.std(np.concatenate(subject_anomaly_score))))


                # Save the input and output images for the subject
                # Save the anomaly scores for the subject
                # Save the masks for the subject
                results_dir_def_array = results_dir_def + '/' + 'arrays'
                make_dir_safely(results_dir_def_array)
                np.save(os.path.join(results_dir_def_array, 'subject_{}_anomaly_scores.npy'.format(subject_idx)), np.concatenate(subject_anomaly_score))    
                np.save(os.path.join(results_dir_def_array, 'subject_{}_masks.npy'.format(subject_idx)), np.concatenate(subject_masks))
                np.save(os.path.join(results_dir_def_array, 'subject_{}_input_images.npy'.format(subject_idx)), np.concatenate(subject_batch))
                
                all_anomaly_scores.append(np.concatenate(subject_anomaly_score))
                all_masks.append(np.concatenate(subject_masks))
            # End of all data
            # Evaluate anomalies
            
            all_anomaly_scores = np.array(all_anomaly_scores)
            all_masks = np.array(all_masks)


            # Compute validation metrics for whole validation set
            # Careful if self supervised, we have one channel and not 4, so extract channel from masks
            if self_supervised:
                # [#patients, #slices, # channel, 32, 32, 24]
                all_masks = all_masks[:,:,0:1,:,:,:]
            validation_metrics(all_anomaly_scores, all_masks)

            

            # Now we do it for each deformation, we can reshape the anomaly scores and the masks
            # to be of shape (n_def, n_patients, n_slices, 32, 32, 24, channel)
            # Careful again if self supervised, we have one channel and not 4
            if self_supervised:
                all_anomaly_scores = all_anomaly_scores.reshape(n_def, n_val_patients, spatial_size_z,1, 32, 32, 24)    
                all_masks = all_masks.reshape(n_def, n_val_patients, spatial_size_z, 1, 32, 32, 24)
            else:
                all_anomaly_scores = all_anomaly_scores.reshape(n_def, n_val_patients, spatial_size_z, 4, 32, 32, 24)
                all_masks = all_masks.reshape(n_def, n_val_patients, spatial_size_z, 4, 32, 32, 24)
            for i, deformation in enumerate(deformation_list_validation_set):
                all_anomaly_scores_def = all_anomaly_scores[i]
                all_masks_def = all_masks[i]
                logging.info('***********************************************************************************************************')
                logging.info(f"Deformation: {deformation}")
                logging.info('***********************************************************************************************************')
                validation_metrics(all_anomaly_scores_def, all_masks_def, deformation)


        # Visualize the predictions on the test set (no artificial deformations)
        # Some are healthy and some are anomalous
        if "test" in data_to_visualize:
            # Create another log file specifically for test
            log_file = os.path.join(results_dir, 'log_test.txt')
            condition_handler = logging.FileHandler(log_file)
            condition_handler.setLevel(logging.INFO)
            logger.addHandler(condition_handler)
            healthy_scores = []
            healthy_idx = 0
            anomalous_scores = []
            anomalous_idx = 0
            highest_anomaly_score_healthy = 0
            lowest_anomaly_score_anomalous = 1e10
            highest_anomaly_score_anomalous = 0
            lowest_anomaly_score_healthy = 1e10
            
            results_dir_test = results_dir + '/' + 'test'
            make_dir_safely(results_dir_test)
            # We want to separate by subject
            
            subject_indexes = range(np.int16(images_test.shape[0]/spatial_size_z))
            for subject_idx in subject_indexes:
                start_idx = 0
                end_idx = batch_size
                subject_anomaly_score = []
                subject_reconstruction = []
                subject_sliced = images_test[subject_idx*spatial_size_z:(subject_idx+1)*spatial_size_z]
                subject_labels = labels_test[subject_idx*spatial_size_z:(subject_idx+1)*spatial_size_z]
                while end_idx <= spatial_size_z:
                    # Select slices for batch and run model
                    
                    # If get neighbour slices
                    if model_str.__contains__('cond_conv'):
                        # Then we are using the neighbour slices as well for the routing function
                        neighbour_indices = get_random_and_neighbour_indices(np.arange(start_idx, end_idx), subject_length=spatial_size_z)
                        adjacent_images = get_combined_images(subject_sliced, neighbour_indices)
                        # The adjacent images are of size [b,3,32,32,24,4] for ascending aorta
                        # We want to have the channel in the second dimension and add the new slices dimenison into the channel dimension
                        adjacent_batch_slices = torch.from_numpy(adjacent_images).to(device).transpose(1,5).transpose(2,5).transpose(3,5).transpose(4,5).float()
                        adjacent_batch_slices = adjacent_batch_slices.reshape(adjacent_batch_slices.shape[0], -1, adjacent_batch_slices.shape[3], adjacent_batch_slices.shape[4], adjacent_batch_slices.shape[5])
                        # size [b,c*3, 32,32,24]
                    batch = subject_sliced[start_idx:end_idx]
                    labels = subject_labels[start_idx:end_idx]
                    batch_z_slice = torch.from_numpy(np.arange(start_idx, end_idx)).float().to(device)
                    batch = torch.from_numpy(batch).transpose(1,4).transpose(2,4).transpose(3,4).float().to(device)
                    with torch.no_grad():
                        model.eval()
                        
                        input_dict = {'input_images': batch, 'batch_z_slice':batch_z_slice, 'adjacent_batch_slices':adjacent_batch_slices}
                        output_dict = model(input_dict)
                        if self_supervised:
                            output_images = torch.sigmoid(output_dict['decoder_output'])
                        else:
                            # Reconstruction based
                            model_output = output_dict['decoder_output']
                            output_images = torch.abs(output_dict['decoder_output'] - batch)
                            subject_reconstruction.append(model_output.cpu().detach().numpy())
                        # Compute anomaly score
                        subject_anomaly_score.append(output_images.cpu().detach().numpy())
                        # Check if all labels are anomalous
                        if np.all(labels == 0):
                            legend = "healthy"
                        elif np.all(labels == 1):
                            legend = "anomalous"
                        else:
                            raise ValueError("Labels are not all healthy or all anomalous, change batch size")


                        # If visualizing, save the images
                        if "visualize" in actions:
                            output_images = output_images.transpose(1,2).transpose(2,3).transpose(3,4).cpu().detach().numpy()
                            #labels = labels*np.ones_like(output_images)
                            batch = batch.transpose(1,2).transpose(2,3).transpose(3,4).cpu().detach().numpy()

                            plot_batches_SSL_in_out(batch, output_images, channel_to_show = 1,every_x_time_step=1, out_path=os.path.join(results_dir_test, '{}_subject_{}_batch_{}_to_{}.png'.format(legend, subject_idx, start_idx, end_idx)))
                    start_idx += batch_size
                    end_idx += batch_size
                if legend == "healthy":
                    healthy_scores.append(np.concatenate(subject_anomaly_score))
                    # Save the healthy subject with highest anomaly score
                    if np.mean(subject_anomaly_score) > highest_anomaly_score_healthy:
                        highest_anomaly_score_healthy = np.mean(subject_anomaly_score)
                        highest_anomaly_score_healthy_subject = healthy_idx
                        highest_anomaly_score_healthy_subject_dataset = subject_idx
                        output_highest_anomaly_score_healthy_subject = np.concatenate(subject_anomaly_score)

                    # Save the healthy subject with lowest anomaly score
                    if np.mean(subject_anomaly_score) < lowest_anomaly_score_healthy:
                        lowest_anomaly_score_healthy = np.mean(subject_anomaly_score)
                        lowest_anomaly_score_healthy_subject = healthy_idx
                        lowest_anomaly_score_healthy_subject_dataset = subject_idx
                        output_lowest_anomaly_score_healthy_subject = np.concatenate(subject_anomaly_score)
                    healthy_idx += 1
                    

                else:
                    anomalous_scores.append(np.concatenate(subject_anomaly_score))
                    # Save the anomalous subject with highest anomaly score
                    if np.mean(subject_anomaly_score) > highest_anomaly_score_anomalous:
                        highest_anomaly_score_anomalous = np.mean(subject_anomaly_score)
                        highest_anomaly_score_anomalous_subject = anomalous_idx
                        highest_anomaly_score_anomalous_subject_dataset = subject_idx
                        output_highest_anomaly_score_anomalous_subject = np.concatenate(subject_anomaly_score)
                    # Save the anomalous subject with lowest anomaly score
                    if np.mean(subject_anomaly_score) < lowest_anomaly_score_anomalous:
                        lowest_anomaly_score_anomalous = np.mean(subject_anomaly_score)
                        lowest_anomaly_score_anomalous_subject = anomalous_idx
                        lowest_anomaly_score_anomalous_subject_dataset = subject_idx
                        output_lowest_anomaly_score_anomalous_subject = np.concatenate(subject_anomaly_score)
                    anomalous_idx += 1
                
                logging.info('{}_subject {} anomaly_score: {:.4e} +/- {:.4e}'.format(legend, subject_idx, np.mean(subject_anomaly_score), np.std(subject_anomaly_score)))
                # Save the anomaly scores for the subject if it does not exist
                results_dir_subject = os.path.join(results_dir_test, "outputs")
                inputs_dir_subject = os.path.join(results_dir_test, "inputs")
                make_dir_safely(inputs_dir_subject)
                make_dir_safely(results_dir_subject)
                # Check if the subject_reconstruction list is empty else save it
                if len(subject_reconstruction) > 0:
                    subject_reconstruction = np.concatenate(subject_reconstruction)
                    reconstruction_dir_subject = os.path.join(results_dir_test, "reconstruction")
                    make_dir_safely(reconstruction_dir_subject)
                    file_path_reconstruction = os.path.join(reconstruction_dir_subject, f'{test_subjects_dict[subject_idx]}_reconstruction.npy')
                    np.save(file_path_reconstruction, subject_reconstruction)
                    ## Check if the file_path exists else save it
                    #if not os.path.exists(file_path_reconstruction):
                    np.save(file_path_reconstruction, subject_reconstruction)
                file_path_input = os.path.join(inputs_dir_subject, f'{test_subjects_dict[subject_idx]}_inputs.npy')
                file_path = os.path.join(results_dir_subject, f'{test_subjects_dict[subject_idx]}_anomaly_scores.npy')
                ## Check if the file_path exists else save it
                #if not os.path.exists(file_path):
                np.save(file_path, np.concatenate(subject_anomaly_score))
                np.save(file_path_input, subject_sliced)
                # End of subject
            # End of data set
            healthy_scores = np.array(healthy_scores)
            anomalous_scores = np.array(anomalous_scores)
            
            healthy_mean_anomaly_score = np.mean(healthy_scores)
            healthy_std_anomaly_score = np.std(healthy_scores)
            anomalous_mean_anomaly_score = np.mean(anomalous_scores)
            anomalous_std_anomaly_score = np.std(anomalous_scores)
            logging.info('============================================================')
            logging.info('Control subjects anomaly_score: {} +/- {:.4e}'.format(healthy_mean_anomaly_score, healthy_std_anomaly_score))
            logging.info('Anomalous subjects anomaly_score: {} +/- {:.4e}'.format(anomalous_mean_anomaly_score, anomalous_std_anomaly_score))
            logging.info('============================================================')
            # Compute AUC-ROC score
            logging.info('============================================================')
            logging.info('Computing AUC-ROC score...')
            logging.info('============================================================')
            compute_auc(healthy_scores, anomalous_scores, results_dir, format_wise= "patient_wise", agg_function=np.mean)
            compute_auc(healthy_scores, anomalous_scores, results_dir, format_wise= "imagewise", agg_function=np.mean)
            compute_auc(healthy_scores, anomalous_scores, results_dir, format_wise= "2Dslice", agg_function=np.mean)
            logging.info('============================================================')
            # Compute average Precision
            logging.info('============================================================')
            logging.info('Computing average precision...')
            logging.info('============================================================')
            compute_average_precision(healthy_scores, anomalous_scores, format_wise= "patient_wise", agg_function= np.mean)
            compute_average_precision(healthy_scores, anomalous_scores, format_wise= "imagewise", agg_function= np.mean)
            compute_average_precision(healthy_scores, anomalous_scores, format_wise= "2Dslice", agg_function= np.mean)

            
            if "visualize" in actions:
                logging.info('============================================================')
                # Plot the scores
                logging.info('============================================================')
                logging.info('Plotting scores...')
                logging.info('============================================================')
                plot_scores(healthy_scores, anomalous_scores, results_dir, level = 'patient')
                plot_scores(healthy_scores, anomalous_scores, results_dir, level = 'imagewise')
                # Plotting scores with the highest and lowest anomaly scores
                logging.info('============================================================')
                logging.info('Plotting scores with the highest and lowest anomaly scores...')
                logging.info('============================================================')
                plot_scores(healthy_scores[lowest_anomaly_score_healthy_subject:lowest_anomaly_score_healthy_subject+1], anomalous_scores[highest_anomaly_score_anomalous_subject:highest_anomaly_score_anomalous_subject+1], results_dir, level = 'patient', note= "healthiest_healthy_vs_most_anomalous_anomalous")
                most_separable_patients_z_slices = plot_scores(healthy_scores[lowest_anomaly_score_healthy_subject:lowest_anomaly_score_healthy_subject+1], anomalous_scores[highest_anomaly_score_anomalous_subject:highest_anomaly_score_anomalous_subject+1], results_dir, level = 'imagewise', note= "healthiest_healthy_vs_most_anomalous_anomalous")
                plot_scores(healthy_scores[highest_anomaly_score_healthy_subject:highest_anomaly_score_healthy_subject+1], anomalous_scores[lowest_anomaly_score_anomalous_subject:lowest_anomaly_score_anomalous_subject+1], results_dir, level = 'patient', note= "most_anomalous_healthy_vs_healthiest_anomalous")
                least_separable_patients_z_slices = plot_scores(healthy_scores[highest_anomaly_score_healthy_subject:highest_anomaly_score_healthy_subject+1], anomalous_scores[lowest_anomaly_score_anomalous_subject:lowest_anomaly_score_anomalous_subject+1], results_dir, level = 'imagewise', note= "most_anomalous_healthy_vs_healthiest_anomalous")
                # Using the mosst separable patients, plot the z slices from the the subject index saved in highest_anomaly_score_anoamalous_subject_dataset and lowest_anomaly_score_healthy_subject_dataset
                logging.info('============================================================')
                logging.info('Plotting z slices from the most separable patients...')
                logging.info('============================================================')
                image_highest_anomaly_score_anomalous_subject = images_test[highest_anomaly_score_anomalous_subject_dataset*spatial_size_z:(highest_anomaly_score_anomalous_subject_dataset+1)*spatial_size_z]
                image_lowest_anomaly_score_healthy_subject = images_test[lowest_anomaly_score_healthy_subject_dataset*spatial_size_z:(lowest_anomaly_score_healthy_subject_dataset+1)*spatial_size_z]
                image_lowest_anomaly_score_anomalous_subject = images_test[lowest_anomaly_score_anomalous_subject_dataset*spatial_size_z:(lowest_anomaly_score_anomalous_subject_dataset+1)*spatial_size_z]
                image_highest_anomaly_score_healthy_subject = images_test[highest_anomaly_score_healthy_subject_dataset*spatial_size_z:(highest_anomaly_score_healthy_subject_dataset+1)*spatial_size_z]
                # Put everything in a a dictionary
                images_dict = {'image_highest_anomaly_score_anomalous_subject': image_highest_anomaly_score_anomalous_subject,
                                'image_lowest_anomaly_score_healthy_subject': image_lowest_anomaly_score_healthy_subject,
                                'image_lowest_anomaly_score_anomalous_subject': image_lowest_anomaly_score_anomalous_subject,
                                'image_highest_anomaly_score_healthy_subject': image_highest_anomaly_score_healthy_subject,
                                'highest_anomaly_score_anomalous_subject': highest_anomaly_score_anomalous_subject,
                                'lowest_anomaly_score_healthy_subject': lowest_anomaly_score_healthy_subject,
                                'lowest_anomaly_score_anomalous_subject': lowest_anomaly_score_anomalous_subject,
                                'highest_anomaly_score_healthy_subject': highest_anomaly_score_healthy_subject,
                                'output_highest_anomaly_score_anomalous_subject': output_highest_anomaly_score_anomalous_subject,
                                'output_lowest_anomaly_score_healthy_subject': output_lowest_anomaly_score_healthy_subject,
                                'output_lowest_anomaly_score_anomalous_subject': output_lowest_anomaly_score_anomalous_subject,
                                'output_highest_anomaly_score_healthy_subject': output_highest_anomaly_score_healthy_subject}
                
                
                # Function to plot
                save_dir_test_images = os.path.join(project_code_root, results_dir, 'thesis_plots')
                plot_slices(images_dict = images_dict, most_separable_patients_z_slices = most_separable_patients_z_slices, least_separable_patients_z_slices = least_separable_patients_z_slices, save_dir_test_images = save_dir_test_images, time_steps = 1)
                plot_slices(images_dict = images_dict, most_separable_patients_z_slices = most_separable_patients_z_slices, least_separable_patients_z_slices = least_separable_patients_z_slices, save_dir_test_images = save_dir_test_images, time_steps = 15)

                # Do statistical tests 
                logging.info('============================================================')
                logging.info('Doing statistical tests...')
                logging.info('============================================================')
                # With subject means
                logging.info('Patient wise \n')
                statistical_tests(healthy_scores.mean(axis=(1,2,3,4,5)), anomalous_scores.mean(axis=(1,2,3,4,5)))
                logging.info('============================================================')
                # Finish logging
                logging.info('============================================================')
                logging.info('Logging finished')
            # remove the handler when you're done logging
            logger.removeHandler(condition_handler)
        logger.removeHandler(handler)
            
            
