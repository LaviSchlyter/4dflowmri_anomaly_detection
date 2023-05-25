# =================================================================================
# ============== GENERAL PACKAGE IMPORTS ==========================================
# =================================================================================
import os

import torch 
import tqdm
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
from helpers.run import train, load_model, evaluate
from helpers.data_loader import load_data



# =================================================================================
# ============== IMPORT MODELS ==================================================
# =================================================================================

from models.vae import VAE

# =================================================================================
# ============== MAIN FUNCTION ==================================================
# =================================================================================

if __name__ ==  "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Train a VAE model.')
    parser.add_argument('--model', type=str, default="vae", help='Model to train.')
    parser.add_argument('--model_name', type=str, default='0', help='Name of the model.')
    parser.add_argument('--config_path', type=str, required= True, help='Path to the config file.')
    parser.add_argument('--checkpoint', type=str, default="logs", help='Path to the checkpoint file to restore.')
    parser.add_argument('--continue_training', type=bool, default=False, help='Continue training from checkpoint.')
    #parser.add_argument('--train', type=str)
    #parser.add_argument('--val', type=str)
    parser.add_argument('--preprocess_method', type=str)
    # Adding the next ones to enable parameter sweeps but they are defined in the yaml file
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--do_data_augmentation', type=bool)
    parser.add_argument('--gen_loss_factor', type=float)
    parser.add_argument('--z_dim', type=int)

    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    
    # Combine arguments with config file
    for arg, value in vars(args).items():
        if value is not None:
            config[arg] = value
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    if str(config['model_name']) != str(0):
        
        # Costume name given
        pass
    else:
        config['model_name'] =  f"{timestamp}_{config['preprocess_method']}_lr{'{:.3e}'.format(config['lr'])}-e{config['epochs']}-bs{config['batch_size']}-zdim{config['z_dim']}-da{config['do_data_augmentation']}"
    
    wandb_mode = "online" # online/ disabled

    with wandb.init(project="4dflowmri_anomaly_detection", name=config['model_name'], config=config, tags= ['debug']):
        config = wandb.config
        print('after_init', config['model_name'])

        
        # Check if it is a sweep
        sweep_id = os.environ.get("WANDB_SWEEP_ID")
        if sweep_id:
            # We add a level for the sweep name 
            config['exp_path'] = os.path.join(config_sys.log_root, config['model'],config['preprocess_method'], sweep_id,config['model_name'])
        else:
            config['exp_path'] = os.path.join(config_sys.log_root, config['model'],config['preprocess_method'], config['model_name'])
        log_dir = config['exp_path']
        # ================================================
        # ======= LOGGING CONFIGURATION ==================
        # ================================================
        project_data_root = config_sys.project_data_root
        project_code_root = config_sys.project_code_root
        # Create the log directory if it does not exist
        make_dir_safely(log_dir)
        logging.info('=============================================================================')
        logging.info(f"Logging to {log_dir}")
        logging.info('=============================================================================')
        # ================================================
        # ======= DATA CONFIGURATION LOADING =====================
        # ================================================
        # Load the data
        images_tr, images_vl = load_data(config, config_sys, idx_start_tr=0, idx_end_tr=5, idx_start_vl=5, idx_end_vl=8)
        
        # ================================================
        # Initialize the model, training parameters, model name and logging
        # ================================================
        if config['model'] == 'vae':
            model = VAE(z_dim=config['z_dim'], in_channels=4, gf_dim=8).to(device)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        
        if config['continue_training']:
            continue_train_path = os.path.join(project_code_root, config["model_directory"])
            model = load_model(model, continue_train_path, config["latest_model_epoch"])
            already_completed_epochs = config['latest_model_epoch']
        else:
            already_completed_epochs = 0

        # Train 
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        train(model, images_tr, images_vl, log_dir, already_completed_epochs, config, device, optimizer)

