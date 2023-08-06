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
import math
from pytorch_model_summary import summary

# =================================================================================
# ============== IMPORT HELPER FUNCTIONS ===========================================
# =================================================================================

from helpers import data_bern_numpy_to_preprocessed_hdf5
from helpers import data_bern_numpy_to_hdf5
from helpers.utils import make_dir_safely, verify_leakage, kld_min
from helpers.run import train, load_model, evaluate
from helpers.data_loader import load_data, load_syntetic_data


SEED = 25
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =================================================================================
# ============== IMPORT MODELS ==================================================
# =================================================================================

from models.vae import VAE, VAE_linear, VAE_convT
from models.condconv import CondVAE, CondConv

# =================================================================================
# ============== MAIN FUNCTION ==================================================
# =================================================================================

if __name__ ==  "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Train a VAE model.')
    parser.add_argument('--model', type=str, default="vae_convT", help='Model to train.')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the model.')
    parser.add_argument('--config_path', type=str, required= True, help='Path to the config file.')
    parser.add_argument('--checkpoint', type=str, default="logs", help='Path to the checkpoint file to restore.')
    parser.add_argument('--continue_training', type=bool, help='Continue training from checkpoint.')
    parser.add_argument('--preprocess_method', type=str)
    # Adding the next ones to enable parameter sweeps but they are defined in the yaml file
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--do_data_augmentation', type=bool)
    parser.add_argument('--gen_loss_factor', type=float)
    parser.add_argument('--z_dim', type=int)
    parser.add_argument('--tilt', type=int)

    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    
    # Combine arguments with config file
    for arg, value in vars(args).items():
        if value is not None:
            config[arg] = value

    # Get the slurm job id
    config['AAslurm_job_id'] = os.environ.get("SLURM_JOB_ID")
  
    # ================================================
    # Check that if self supervised is True, then use_synthetic_validation is also True
    # ================================================
    if config['self_supervised']:
        config['use_synthetic_validation'] = True
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    if str(config['model_name']) != str(0) and str(config['model_name']) != str(None):
        # Costume name given
        pass
    elif config['model'] == 'vae':
        config['model_name'] =  f"{timestamp}_{config['model']}_{config['preprocess_method'] + '_SSL' if config['self_supervised'] else config['preprocess_method']}_lr{'{:.3e}'.format(config['lr'])}{'_scheduler' + '-e' + str(config['epochs']) if config['use_scheduler'] else '-e' + str(config['epochs'])}-bs{config['batch_size']}-gf_dim{config['gf_dim']}-da{str(config['do_data_augmentation']) if config['self_supervised'] else str(config['do_data_augmentation']) + '-f' +str(config['gen_loss_factor'])}"
    elif config['model'] == 'vae_linear':
        config['model_name'] =  f"{timestamp}_{config['model']}_{config['preprocess_method'] + '_SSL' if config['self_supervised'] else config['preprocess_method']}_lr{'{:.3e}'.format(config['lr'])}{'_scheduler' + '-e' + str(config['epochs']) if config['use_scheduler'] else '-e' + str(config['epochs'])}-bs{config['batch_size']}-gf_dim{config['gf_dim']}-zdim{config['z_dim']}-da{str(config['do_data_augmentation']) if config['self_supervised'] else str(config['do_data_augmentation']) + '-f' +str(config['gen_loss_factor'])}"
    elif config['model'] == 'vae_convT':
        config['model_name'] =  f"{timestamp}_{config['model']}_{config['preprocess_method'] + '_SSL' if config['self_supervised'] else config['preprocess_method']}_lr{'{:.3e}'.format(config['lr'])}{'_scheduler' + '-e' + str(config['epochs']) if config['use_scheduler'] else '-e' + str(config['epochs'])}-bs{config['batch_size']}-gf_dim{config['gf_dim']}-da{str(config['do_data_augmentation']) if config['self_supervised'] else str(config['do_data_augmentation']) + '-f' +str(config['gen_loss_factor'])}"
    elif config['model'] == 'cond_vae':
        config['model_name'] =  f"{timestamp}_{config['model']}_{config['preprocess_method'] + '_SSL' if config['self_supervised'] else config['preprocess_method']}_lr{'{:.3e}'.format(config['lr'])}{'_scheduler' + '-e' + str(config['epochs']) if config['use_scheduler'] else '-e' + str(config['epochs'])}-bs{config['batch_size']}-gf_dim{config['gf_dim']}-da{str(config['do_data_augmentation']) if config['self_supervised'] else str(config['do_data_augmentation']) + '-f' +str(config['gen_loss_factor'])}-n_experts{config['n_experts']}"
    elif config['model'] == 'cond_conv':
        config['model_name'] =  f"{timestamp}_{config['model']}_{config['preprocess_method'] + '_SSL' if config['self_supervised'] else config['preprocess_method']}_lr{'{:.3e}'.format(config['lr'])}{'_scheduler' + '-e' + str(config['epochs']) if config['use_scheduler'] else '-e' + str(config['epochs'])}-bs{config['batch_size']}-gf_dim{config['gf_dim']}-da{str(config['do_data_augmentation']) if config['self_supervised'] else str(config['do_data_augmentation']) + '-f' +str(config['gen_loss_factor'])}-n_experts{config['n_experts']}"
    else:
        config['model_name'] =  f"{timestamp}_{config['model']}_{config['preprocess_method'] + '_SSL' if config['self_supervised'] else config['preprocess_method']}_lr{'{:.3e}'.format(config['lr'])}{'_scheduler' + '-e' + str(config['epochs']) if config['use_scheduler'] else '-e' + str(config['epochs'])}-bs{config['batch_size']}-gf_dim{config['gf_dim']}-da{config['do_data_augmentation']}-f{str(config['gen_loss_factor'])}"
    
    # Add extra note to name
    if len(config['note']) > 0:
        config['model_name'] = config['model_name'] + f"{'_' + config['note']}"
    if config['use_synthetic_validation']:
        if len(config['synthetic_data_note']) > 0:
            config['model_name'] = config['model_name'] + f"{'_' + config['validation_metric_format']}" + f"{'_' + config['synthetic_data_note']}"    
        else:
            config['model_name'] = config['model_name'] + f"{'_' + config['validation_metric_format']}"
    wandb_mode = "online" # online/ disabled
    tag = ''
    if config['self_supervised']:
        tag = 'self_supervised'
    else:
        tag = 'reconstruction'

    if config['use_synthetic_validation']:
        tags = [config['model'], 'synthetic_validation', 'various_SEED', config['validation_metric_format']]
        tags.append(tag)
    else:
        tags = [config['model'], 'various_SEED']
        tags.append(tag)
    with wandb.init(project="4dflowmri_anomaly_detection", name=config['model_name'], config=config, tags= tags):
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
        # ======= DATA CONFIGURATION LOADING =============
        # ================================================

        # Check if there is data leakage (some people in both train and test)
        verify_leakage()

        # Load the data
        images_tr, images_vl, _ = load_data(config, config_sys, idx_start_tr=config['idx_start_tr'], idx_end_tr=config['idx_end_tr'], idx_start_vl=config['idx_start_vl'], idx_end_vl=config['idx_end_vl'])

        # Load synthetic data for validation if needed
        if config['use_synthetic_validation']:
            images_vl = load_syntetic_data(preprocess_method = config['preprocess_method'], idx_start=config['idx_start_vl'], idx_end=config['idx_end_vl'], sys_config = config_sys, note = config['synthetic_data_note'])
            logging.info(f"Using synthetic validation data with shape: {images_vl['images'].shape}")
        
        # ================================================
        # Initialize the model, training parameters, model name and logging
        # ================================================
        if config['model'] == 'vae':
            if config['self_supervised']:
                # In this case we have a binary classification problem
                model = VAE(in_channels=4, gf_dim=config['gf_dim'], out_channels=1).to(device)
            else:
                model = VAE(in_channels=4, gf_dim=config['gf_dim'], out_channels=4).to(device)
        elif config['model'] == 'vae_convT':
            if config['self_supervised']:
                # In this case we have a binary classification problem
                model = VAE_convT(in_channels=4, gf_dim=config['gf_dim'], out_channels=1).to(device)
            else:
                model = VAE_convT(in_channels=4, gf_dim=config['gf_dim'], out_channels=4).to(device)
        elif config['model'] == 'cond_vae':
            if config['self_supervised']:
                # In this case we have a binary classification problem
                model = CondVAE(in_channels=4, gf_dim=config['gf_dim'], out_channels=1, num_experts=config['n_experts']).to(device)
            else:
                model = CondVAE(in_channels=4, gf_dim=config['gf_dim'], out_channels=4, num_experts=config['n_experts']).to(device)
        elif config['model'] == 'cond_conv':
            # Then we need the neighbouts for the routing function 
            config['get_neighbours'] = True
            if config['self_supervised']:
                # In this case we have a binary classification problem
                model = CondConv(in_channels=4, gf_dim=config['gf_dim'], out_channels=1, num_experts=config['n_experts']).to(device)
            else:
                model = CondConv(in_channels=4, gf_dim=config['gf_dim'], out_channels=4, num_experts=config['n_experts']).to(device)
        else:
            raise ValueError(f"Unknown model: {config['model']}")
        
        
        if config['continue_training']:
            continue_train_path = os.path.join(project_code_root, config["model_directory"])
            model = load_model(model, continue_train_path, config, device=device)
            already_completed_epochs = config['latest_model_epoch']
        else:
            already_completed_epochs = 0
        # ================================================
        # Print summary of the model
        # ================================================
        logging.info('=======================================================')
        logging.info('Details of the model architecture')
        logging.info('=======================================================')
    
        input_dict= {'input_images': torch.zeros((1, 4, 32, 32, 24)).to(device).float(), 'batch_z_slice': torch.zeros((1,)).to(device).float(), 'adjacent_batch_slices':torch.zeros((1, 12, 32, 32, 24)).to(device)}
        logging.info(summary(model, input_dict, show_input=False))
        
        
        
        
        
        
        # Train 
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], betas=(config['beta1'], config['beta2']))
        train(model, images_tr, images_vl, log_dir, already_completed_epochs, config, device, optimizer)

