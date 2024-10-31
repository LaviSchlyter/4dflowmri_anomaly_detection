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
import sys
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection')
from config import system as config_sys
import wandb
from pytorch_model_summary import summary

# =================================================================================
# ============== IMPORT HELPER FUNCTIONS ===========================================
# =================================================================================
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/src')
from helpers.utils import make_dir_safely, verify_leakage, create_suffix
from helpers.run import train, load_model, evaluate
from helpers.data_loader import load_data, load_syntetic_data


# =================================================================================
# ============== IMPORT MODELS ==================================================
# =================================================================================

from models.model_zoo import SimpleConvNet, SimpleConvNetInterpolate, SimpleConvNet_linear, VAE_convT, ConvWithAux, ConvWithEncDecAux,ConvWithDeepAux, ConvWithDeepEncDecAux, ConvWithDeeperEncDecAux, ConvWithDeeperBNEncDecAux
    
from models.condconv import CondVAE, CondConv

# =================================================================================
# ============== HANDLING CANCELLATION ===========================================
# =================================================================================
import signal

def signal_handler(signum, frame):
    # Print the time at which the job was cancelled
    print("********JOB CANCELLED at ", datetime.datetime.now(), "********")
    # Insert any cleanup or final logging here
    sys.exit(0)

# Attach the signal handler
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Indices to remove based on training deformations
def get_exclusion_indices(config):
    deformations = config['deformation_list']
    num_slices_per_subject = config['spatial_size_z']
    training_blending_method = config['blending'].get('method', 'mixed_grad')

    deformation_to_exclude = {
        "mixed_grad": "poisson_with_mixing",
        "source_grad": "poisson_without_mixing",
        "interpolation": "patch_interpolation"
    }[training_blending_method]  # Map blending methods to their corresponding deformations

    start_val = config['idx_start_vl']
    end_val = config['idx_end_vl']
    num_subjects = end_val - start_val


    total_slices = num_subjects * num_slices_per_subject * len(deformations)  # Total number of slices across all deformations
    slices_per_deformation = total_slices // len(deformations)  # Slices dedicated to each deformation type

    

    exclusion_indices = []

    if deformation_to_exclude in deformations:
        index = deformations.index(deformation_to_exclude)
        start_index = index * slices_per_deformation
        end_index = start_index + slices_per_deformation
        exclusion_indices.append((start_index, end_index))
    config['indices_to_remove'] = [start_index, end_index]
    return config


# =================================================================================
# ============== MAIN FUNCTION ==================================================
# =================================================================================
def generate_model_name(config, timestamp):
    # Generate the dynamic suffix based on the current configuration
    suffix = create_suffix(config)

    base_name = f"{timestamp}_{config['model']}_{config['preprocess_method']}"

    # Append "_SSL" if self-supervised learning is applied.
    if config['self_supervised']:
        base_name += "_SSL"
    else:
        # Append loss factor if not self-supervised and factor is applicable.
        if 'gen_loss_factor' in config:
            base_name += f"-f{config['gen_loss_factor']}"

    # Compose the base string with learning rate, scheduler use, epochs, batch size, and gf_dim.
    base_name += f"_lr{'{:.3e}'.format(config['lr'])}"
    if config.get('use_scheduler'):
        base_name += "_scheduler"
    base_name += f"-e{config['epochs']}-bs{config['batch_size']}-gf_dim{config['gf_dim']}"

    # Append z_dim if it exists and not using self-supervised learning.
    if 'z_dim' in config and not config['self_supervised']:
        base_name += f"-zdim{config['z_dim']}"

    # Handle the presence of experts in the configuration.
    if 'n_experts' in config:
        base_name += f"-n_experts{config['n_experts']}"

    # Append data augmentation status.
    base_name += f"-da{str(config['do_data_augmentation'])}"
    # Append the dynamic suffix and seed note
    base_name += f"__SEED_{config['seed']}_{config['validation_metric_format']}_{suffix}"
    
    # Add synthetic data note if present
    if config.get('synthetic_data_note'):
        base_name += f"{config['synthetic_data_note']}"
    return base_name

if __name__ ==  "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Train a CNN model.')
    parser.add_argument('--model', type=str, default="simple_conv", help='Model to train.')
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

    SEED = config['seed']
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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

    # ================================================
    
    if config['continue_training']:
        config['model_name'] = config['model_directory'].split('/')[-1]

    else:
        # Generate the model name
        config['model_name'] = generate_model_name(config, timestamp)


    
    
    
    tag = ''
    if config['self_supervised']:
        tag = 'self_supervised'
    else:
        tag = 'reconstruction'

    if config['use_synthetic_validation']:
        tags = [config['model'], 'synthetic_validation', f"{SEED}", config['validation_metric_format'], "fixed_anomaly_seed"]
        tags.append(tag)
    else:
        tags = [config['model'], f"{SEED}","fixed_anomaly_seed"]
        tags.append(tag)
        
    
    # ================================================
    if config['use_wandb']:
        wandb_mode = 'online'
    else:
        wandb_mode = 'disabled'

    with wandb.init(project="4dflowmri_anomaly_detection", name=config['model_name'], config=config, tags= tags, mode=wandb_mode):
        config = wandb.config
        print('after_init', config['model_name'])

        # Check if it is a sweep
        sweep_id = os.environ.get("WANDB_SWEEP_ID")
        if sweep_id:
            # We add a level for the sweep name 
            config['exp_path'] = os.path.join(config_sys.log_experiments_root, config['model'],config['preprocess_method'], sweep_id,config['model_name'])
        else:
            config['exp_path'] = os.path.join(config_sys.log_experiments_root, config['model'],config['preprocess_method'], config['model_name'])
        log_dir = config['exp_path']


    config['exp_path'] = os.path.join(config_sys.log_experiments_root, config['model'],config['preprocess_method'], config['model_name'])
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

    # Create the suffix    
    suffix = create_suffix(config)

    # Load the data
    data_dict = load_data(config, config_sys, idx_start_tr=config['idx_start_tr'], idx_end_tr=config['idx_end_tr'], idx_start_vl=config['idx_start_vl'], idx_end_vl=config['idx_end_vl'], suffix = suffix)
    images_tr = data_dict['images_tr']
    images_vl = data_dict['images_vl']

    # Load s data for validation if needed
    if config['use_synthetic_validation']:
        images_vl = load_syntetic_data(preprocess_method = config['preprocess_method'], idx_start=config['idx_start_vl'], idx_end=config['idx_end_vl'], sys_config = config_sys, note = suffix + config['synthetic_data_note'])
        logging.info(f"Using synthetic validation data with shape: {images_vl['images'].shape}")
    
    # ================================================
    # Initialize the model, training parameters, model name and logging
    # ================================================
    # Check if self-supervised or not
    if config['self_supervised']:
        config['out_channels'] = 1
    else:
        config['out_channels'] = 4
    config['in_channels'] = 4   

    model_mapping = {
        'simple_conv': SimpleConvNet,
        'conv_interpolate': SimpleConvNetInterpolate,
        'vae_convT': VAE_convT,
        'cond_vae': CondVAE,
        'cond_conv': CondConv,
        'conv_with_aux': ConvWithAux,
        'conv_enc_dec_aux': ConvWithEncDecAux,
        'deep_conv_with_aux': ConvWithDeepAux,
        'deep_conv_enc_dec_aux': ConvWithDeepEncDecAux,
        'deeper_conv_enc_dec_aux': ConvWithDeeperEncDecAux,
        'deeper_bn_conv_enc_dec_aux': ConvWithDeeperBNEncDecAux
    }

    if config['model'] in model_mapping:
        model_class = model_mapping[config['model']]
        model = model_class(in_channels=config['in_channels'], gf_dim=config['gf_dim'], out_channels=config['out_channels']).to(device)
    else:
        raise ValueError(f"Unknown model: {config['model']}")
    
    
    if config['continue_training']:
        continue_train_path = os.path.join(project_code_root, config["model_directory"])
        model = load_model(model, continue_train_path, config, device=device)
        already_completed_epochs = config['latest_model_epoch']
        
    else:
        already_completed_epochs = 0

    
    # Get the indices to exclude based on the training deformations
    config = get_exclusion_indices(config)
                
    # ================================================
    # Print summary of the model
    # ================================================
    logging.info('=======================================================')
    logging.info('Details of the model architecture')
    logging.info('=======================================================')

    input_dict= {'input_images': torch.zeros((1, 4, 32, 32, 24)).to(device).float(), 'batch_z_slice': torch.zeros((1,)).to(device).float(), 
                    'adjacent_batch_slices':torch.zeros((1, 12, 32, 32, 24)).to(device), 'rotation_matrix': torch.zeros((1, 3, 3)).to(device).float()}
    logging.info(summary(model, input_dict, show_input=False))

    
    # Train 
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], betas=(config['beta1'], config['beta2']))
    train(model, data_dict, images_vl, log_dir, already_completed_epochs, config, device, optimizer)

