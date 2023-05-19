# =================================================================================
# ============== GENERAL PACKAGE IMPORTS ==========================================
# =================================================================================
import os

import torch 
import yaml
import argparse
import logging
import numpy as np
import datetime
from config import system as config_sys
import wandb


# =================================================================================
# ============== IMPORT HELPER FUNCTIONS ===========================================
# =================================================================================

from helpers import data_bern_numpy_to_preprocessed_hdf5
from helpers import data_bern_numpy_to_hdf5
from helpers.utils import make_dir_safely

# =================================================================================
# ============== IMPORT MODELS ==================================================
# =================================================================================

from models.vae import VAE

# =================================================================================
# ============== Train function ==================================================
# =================================================================================




# =================================================================================
# ============== MAIN FUNCTION ==================================================
# =================================================================================

if __name__ ==  "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Train a VAE model.')
    parser.add_argument('--model', type=str, default="vae", help='Model to train.')
    parser.add_argument('--model_name', type=str, default=0, help='Name of the model.')
    parser.add_argument('--config', type=str, required= True, help='Path to the config file.')
    parser.add_argument('--checkpoint', type=str, default="logs", help='Path to the checkpoint file to restore.')
    parser.add_argument('--continue_training', type=bool, default=False, help='Continue training from checkpoint.')
    #parser.add_argument('--train', type=str)
    #parser.add_argument('--val', type=str)
    parser.add_argument('--preprocess', type=str)

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    
    # Combine arguments with config file
    for arg, value in vars(args).items():
        setattr(config, arg, value)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Generate a timestamp string
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")

    with wandb.init(project="4dflowmri_anomaly_detection", config=config, tags= ['debug']):
        config = wandb.config
    

        # ======= MODEL CONFIGURATION =======
        model = config['model']

        
        model_name = config['model_name']

        preprocess_method = config['preprocess']
        continue_training = config['continue_training']

        savepath = config_sys['project_code_root'] + "data"

        # ======= TRAINING PARAMETERS CONFIGURATION =======

        epochs = config['epochs']
        batch_size = config['batch_size']
        learning_rate = config['learning_rate']
        z_dim = config['z_dim']
        
        # ================================================
        # ======= LOGGING CONFIGURATION ==================
        # ================================================

        project_data_root = config_sys['project_data_root']
        project_code_root = config_sys['project_code_root']
        
        # The log directory is namedafter the model and the experiment run name
        log_dir = os.path.join(config_sys['log_root'], model, model_name +'_'+ timestamp)

        # Create the log directory if it does not exist
        make_dir_safely(log_dir)
        logging.info('=============================================================================')
        logging.info(f"Logging to {log_dir}")
        logging.info('=============================================================================')

        # ================================================
        # ======= DATA CONFIGURATION =====================
        # ================================================

        if preprocess_method == 'none':

            logging.info('=============================================================================')
            logging.info(f"Preprocessing method: {preprocess_method}")
            logging.info('Loading training data from: {}'.format(project_data_root))

            data_tr = data_bern_numpy_to_hdf5.load_data(basepath=project_data_root, 
                                                        idx_start=0, 
                                                        idx_end=5, 
                                                        mode='train',
                                                        savepath = savepath)
            

            images_tr = data_tr['images_train']            
            labels_tr = data_tr['labels_train']    
            logging.info(type(images_tr))    
            logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
            logging.info('Shape of training labels: %s' %str(labels_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t]

            logging.info('=============================================================================')
            logging.info('Loading validation data from: {}'.format(project_data_root))

            data_vl = data_bern_numpy_to_hdf5.load_data(basepath=project_data_root,
                                                        idx_start=5,
                                                        idx_end=10,
                                                        mode='val',
                                                        savepath=savepath)
            
            images_vl = data_vl['images_val']
            labels_vl = data_vl['labels_val']
            logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
            logging.info('Shape of validation labels: %s' %str(labels_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t]

        # ================================================
        # === If mask preprocessing is selected ==========
        # ================================================
        elif preprocess_method == 'mask':

            logging.info('=============================================================================')
            logging.info(f"Preprocessing method: {preprocess_method}")
            logging.info('Loading training data from: {}'.format(project_data_root))

            data_tr = data_bern_numpy_to_preprocessed_hdf5.load_masked_data(basepath=project_data_root,
                                                                            idx_start=0,
                                                                            idx_end=5,
                                                                            mode='train',
                                                                            savepath=savepath)
            
            images_tr = data_tr['masked_images_train']
            logging.info(type(images_tr))
            logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]

            logging.info('=============================================================================')
            logging.info('Loading validation data from: {}'.format(project_data_root))

            data_vl = data_bern_numpy_to_preprocessed_hdf5.load_masked_data(basepath=project_data_root,
                                                                            idx_start=5,
                                                                            idx_end=10,
                                                                            mode='val',
                                                                            savepath=savepath)
            
            images_vl = data_vl['masked_images_val']
            logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
            logging.info('=============================================================================')
        
        # ================================================
        # === If slicing preprocessing is selected ==========

        elif preprocess_method == 'slice':

            logging.info('=============================================================================')
            logging.info(f"Preprocessing method: {preprocess_method}")
            logging.info('Loading training data from: {}'.format(project_data_root))

            data_tr = data_bern_numpy_to_preprocessed_hdf5.load_cropped_data_sliced(basepath=project_data_root,
                                                                                    idx_start=0,
                                                                                    idx_end=5,
                                                                                    mode='train',
                                                                                    savepath=savepath)
            
            images_tr = data_tr['sliced_images_train']
            logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
            logging.info('=============================================================================')

            logging.info('=============================================================================')
            logging.info('Loading validation data from: {}'.format(project_data_root))

            data_vl = data_bern_numpy_to_preprocessed_hdf5.load_cropped_data_sliced(basepath=project_data_root,
                                                                                    idx_start=5,
                                                                                    idx_end=10,
                                                                                    mode='val',
                                                                                    savepath=savepath)
            
            images_vl = data_vl['sliced_images_val']
            logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
            logging.info('=============================================================================')

        # ================================================
        # ==== If masked slicing preprocessing is selected
        # ================================================

        elif preprocess_method == 'masked_slice':

            logging.info('=============================================================================')
            logging.info(f"Preprocessing method: {preprocess_method}")
            logging.info('Loading training data from: {}'.format(project_data_root))

            data_tr = data_bern_numpy_to_preprocessed_hdf5.load_masked_data_sliced(basepath=project_data_root,
                                                                                    idx_start=0,
                                                                                    idx_end=5,
                                                                                    mode='train',
                                                                                    savepath=savepath)
            
            images_tr = data_tr['sliced_images_train']
            logging.info(type(images_tr))
            logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
            
            logging.info('=============================================================================')
            logging.info('Loading validation data from: {}'.format(project_data_root))

            data_vl = data_bern_numpy_to_preprocessed_hdf5.load_masked_data_sliced(basepath=project_data_root,
                                                                                    idx_start=5,
                                                                                    idx_end=10,
                                                                                    mode='val',
                                                                                    savepath=savepath)
            
            images_vl = data_vl['sliced_images_val']
            logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
            logging.info('=============================================================================')

        
        # ================================================
        # ==== if sliced full aorta preprocessing is selected
        # ================================================
        elif preprocess_method == 'sliced_full_aorta':

            logging.info('=============================================================================')
            logging.info(f"Preprocessing method: {preprocess_method}")
            logging.info('Loading training data from: {}'.format(project_data_root))

            data_tr = data_bern_numpy_to_preprocessed_hdf5.load_cropped_data_sliced_full_aorta(basepath=project_data_root,
                                                                                                idx_start=0,
                                                                                                idx_end=5,
                                                                                                mode='train',
                                                                                                savepath=savepath)
            
            images_tr = data_tr['sliced_images_train']
            logging.info(type(images_tr))
            logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]

            logging.info('=============================================================================')
            logging.info('Loading validation data from: {}'.format(project_data_root))

            data_vl = data_bern_numpy_to_preprocessed_hdf5.load_cropped_data_sliced_full_aorta(basepath=project_data_root,
                                                                                                idx_start=5,
                                                                                                idx_end=10,
                                                                                                mode='val',
                                                                                                savepath=savepath)
            
            images_vl = data_vl['sliced_images_val']
            logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
            logging.info('=============================================================================')

        # ================================================
        # ==== if masked sliced full aorta preprocessing is selected
        # ================================================

        elif preprocess_method == 'masked_sliced_full_aorta':

            logging.info('=============================================================================')
            logging.info(f"Preprocessing method: {preprocess_method}")
            logging.info('Loading training data from: {}'.format(project_data_root))

            data_tr = data_bern_numpy_to_preprocessed_hdf5.load_masked_data_sliced_full_aorta(basepath=project_data_root,
                                                                                                idx_start=0,
                                                                                                idx_end=5,
                                                                                                mode='train',
                                                                                                savepath=savepath)
            
            images_tr = data_tr['sliced_images_train']
            logging.info(type(images_tr))
            logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]

            logging.info('=============================================================================')
            logging.info('Loading validation data from: {}'.format(project_data_root))

            data_vl = data_bern_numpy_to_preprocessed_hdf5.load_masked_data_sliced_full_aorta(basepath=project_data_root,
                                                                                                idx_start=5,
                                                                                                idx_end=10,
                                                                                                mode='val',
                                                                                                savepath=savepath)
            
            images_vl = data_vl['sliced_images_val']
            logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
            logging.info('=============================================================================')
        else:
            raise ValueError(f"Preprocessing method {preprocess_method} not implemented.")



        # ================================================
        # Initialize the model, training parameters, model name and logging
        # ================================================

        # Initialize the model
        if model == 'vae':
            model = VAE()
        else:
            raise ValueError(f"Unknown model: {model}")
        
        if continue_training:
            pass
        
