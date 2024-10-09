import torch
import numpy as np
import logging
import os
from tqdm import tqdm
import wandb
import matplotlib
from matplotlib import pyplot as plt

import sys
sys.path.append("/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_score/src")

from torch.optim.lr_scheduler import CosineAnnealingLR


# =================================================================================
# ============== HELPER FUNCTIONS =============================================

from helpers.utils import compute_losses, make_dir_safely, apply_blending, compute_euler_angles_xyz_and_trace, save_inputs_outputs
from helpers.utils_inference import get_combined_images
from helpers.batches import iterate_minibatches
from helpers.metrics import compute_auc_roc_score, compute_average_precision_score
from helpers.synthetic_anomalies import create_cube_mask, create_cube_mask_4D


# Set colormap for consistency
cmapper_gray = matplotlib.cm.get_cmap("gray")

# List of models that do not have auxiliary encoders and decoders
model_list_without_enc_dec_aux = ['simple_conv', 'vae_convT', 'cond_vae', 'conv_with_aux', 'deep_conv_with_aux']
model_list_with_enc_dec_aux = ['deep_conv_enc_dec_aux', 'deeper_bn_conv_enc_dec_aux', 'deeper_conv_enc_dec_aux']

def get_random_and_neighbour_indices_train(total_length, num_random, subject_length):
    random_indices = np.random.choice(total_length, size=num_random, replace=False)
    neighbour_indices = []
    for idx in random_indices:
        subject_num = idx // subject_length
        slice_num = idx % subject_length
        prev_idx = subject_num * subject_length + max(0, slice_num - 1)
        next_idx = subject_num * subject_length + min(subject_length - 1, slice_num + 1)
        neighbour_indices.append((prev_idx, idx, next_idx))
    
    return neighbour_indices

# =================================================================================
# ============== Evaluation function =============================================
# =================================================================================
def evaluate(model,epoch, images_vl, best_val_score, log_dir,config, device, val_table_watch, data_dict=None):
    # Set the model to eval mode
    model.eval()
    
    
    val_gen_losses = 0
    val_res_losses = 0
    val_lat_losses = 0
    val_losses = 0
    val_euler_losses = 0
    val_trace_losses = 0

    number_of_batches = 0
    detected_errors = []
    val_masks = []

    # set y to None
    y = None
    # Set adjacent_batch_slices to None
    adjacent_batch_slices = None
    # Loop over the batches
    # No data augmentation for validation set
    for nv, batch_dict in enumerate(iterate_minibatches(images_vl, config, data_augmentation=False, with_labels= config['use_synthetic_validation'], remove_indices= config['self_supervised'], indices_to_remove = config['indices_to_remove'], data_dict=data_dict, train_data = False)):

        with torch.no_grad():
            input_images = batch_dict['X']
            batch_z_slice = batch_dict['batch_z_slice']
            y = batch_dict['Y']
            if y is not None:
                y = torch.from_numpy(y).transpose(1,4).transpose(2,4).transpose(3,4).to(device)
            rotation_matrix = batch_dict['rotation_matrix']
            
            batch_z_slice = torch.from_numpy(batch_z_slice).to(device)

            # If we are looking at neighbouring slices we need to adapt the inputs
            if config.get('get_neighbours', False):
                adjacent_images = np.copy(input_images)
                adjacent_batch_slices = torch.from_numpy(adjacent_images).to(device).transpose(1,5).transpose(2,5).transpose(3,5).transpose(4,5).float()
                adjacent_batch_slices = adjacent_batch_slices.reshape(adjacent_batch_slices.shape[0], -1, adjacent_batch_slices.shape[3], adjacent_batch_slices.shape[4], adjacent_batch_slices.shape[5])


                input_images = input_images[:,1,...]
        

            # Transfer the input_images to "device"
            input_images = torch.from_numpy(input_images).transpose(1,4).transpose(2,4).transpose(3,4).to(device)

            # Transfer the rotation matrix to "device"
            rotation_matrix = torch.from_numpy(rotation_matrix).to(device).float()
            
            # Forward pass
            input_dict = {'input_images': input_images.float(), 'batch_z_slice': batch_z_slice.float(), 'adjacent_batch_slices': adjacent_batch_slices,
                          'rotation_matrix': rotation_matrix}
            output_dict = model(input_dict)
            # Visualization
            if (epoch%(config['validation_viz_frequency']) == 0) or (epoch ==1):
                if nv%200 == 0:
                    if config['use_wandb']:
                        visualize_intermediate_results(epoch, input_images, output_dict['decoder_output'], val_table_watch, labels=y)
                    save_inputs_outputs(nv, epoch, input_images, output_dict['decoder_output'], config,labels=y, training= False)
            
            if config['use_synthetic_validation'] and not config['self_supervised']:
                # Take the difference between input and output (for the unsupervised way)
                detected_errors.append(torch.abs(input_images - output_dict['decoder_output']).cpu().detach().numpy())
                val_masks.append(y)
            

            # Check keys in output_dict to adapt loss computation
            # Compute the loss
            """
            When evaluating the model, we want to look at the losses without the factors
            """
            if config['self_supervised']:
                # For the self-supervised the output of network is already the error
                detected_errors.append(output_dict['decoder_output'].cpu().detach().numpy())
                val_masks.append(y.cpu().detach().numpy())
            else:
                
                if config['model'] in model_list_without_enc_dec_aux:
                    dict_loss = compute_losses(input_images, output_dict, config)
                    val_gen_losses += dict_loss['gen_loss'].mean().item()
                    val_res_losses += dict_loss['res_loss'].mean().item()
                    val_lat_losses += dict_loss['lat_loss'].mean().item()
                elif config['model'] in model_list_with_enc_dec_aux:
                    # Uses the euler angles and trace of the rotation matrix
                    with torch.no_grad():
                        # Compute the Euler angles and the trace of the rotation matrix
                        euler_trace_results= compute_euler_angles_xyz_and_trace(rotation_matrix)
                        euler_angles, trace = euler_trace_results['euler_angles'].detach(), euler_trace_results['trace'].detach()

                        # Add the euler angles and trace to the input dictionary
                        input_dict['euler_angles'] = euler_angles
                        input_dict['trace'] = trace

                    dict_loss = compute_losses(input_images, output_dict, config, input_dict)
                    
                    
                    val_gen_losses += dict_loss['gen_loss'].mean().item()
                    val_res_losses += dict_loss['res_loss'].mean().item()
                    val_lat_losses += dict_loss['lat_loss'].mean().item()
                    val_euler_losses += dict_loss['euler_loss'].mean().item()
                    val_trace_losses += dict_loss['trace_loss'].mean().item()
                else:
                    raise ValueError('output_dict does not contain the correct keys')
            
                # Val loss does not have the factors - this enables better comparaison

                val_losses += dict_loss['val_loss'].item()
            number_of_batches += 1
    if config['use_synthetic_validation']:
        logging.info('Computing validation metrics')
        # Compute ROC AUC score
        auc_roc = compute_auc_roc_score(detected_errors, val_masks, config)
        logging.info('Epoch: {}, val_auc_roc: {:.5f}'.format(epoch, auc_roc))
        # Compute average precision score
        ap = compute_average_precision_score(detected_errors, val_masks, config)
        logging.info('Epoch: {}, val_ap: {:.5f}'.format(epoch, ap))

        if config['use_wandb']:
            wandb.log({'val_auc_roc': round(auc_roc, 5)})
            wandb.log({'val_ap': round(ap, 5)})


        if config['val_metric'] == 'auc_roc':
            current_score = auc_roc
            best_val_name = 'best_val_auc_roc'
        elif config['val_metric'] == 'AP':
            current_score = ap
            best_val_name = 'best_val_ap'
        else:
            raise ValueError('val_metric not recognized', config['val_metric'])
            
        # Save the model if the validation score is the best we've seen so far.
        if current_score > best_val_score:
            logging.info('Saving best model at epoch {}'.format(epoch))
            best_val_score = current_score
            if config['use_wandb']:
                wandb.run.summary['best_val_epoch'] = epoch
                wandb.run.summary[best_val_name] = best_val_score
                wandb.run.summary['best_validation'] = best_val_score
                checkpoint(model, os.path.join(log_dir, f"{config['model_name']}.ckpt-best"))
    else:
        if val_losses < best_val_score:
            logging.info('Saving best model at epoch {}'.format(epoch))
            best_val_score = val_losses
            if config['use_wandb']:
                wandb.run.summary['best_val_epoch'] = epoch
                wandb.run.summary['best_validation'] = best_val_score
            checkpoint(model, os.path.join(log_dir, f"{config['model_name']}.ckpt-best"))
    if config['self_supervised']:
        # We don't have the same losses 
        pass
    else:

        
        if config['model'] in model_list_without_enc_dec_aux:
            # VAE model
            # Logging the validation losses
            logging.info('Epoch: {}, val_gen_losses: {:.5f}, val_lat_losses: {:.5f}, val_res_losses: {:.5f}, val_losses: {:.5f}'.format(epoch,  val_gen_losses/number_of_batches,val_lat_losses/number_of_batches, val_res_losses/number_of_batches, val_losses/number_of_batches))

            if config['use_wandb']:
                wandb.log({ 'val_gen_losses': round(val_gen_losses/number_of_batches, 5) , 'val_lat_losses': round(val_lat_losses/number_of_batches, 5), 'val_res_losses': round(val_res_losses/number_of_batches, 5), 'val_losses': round(val_losses/number_of_batches, 5), 'best_val_score': best_val_score/number_of_batches})
        elif config['model'] in model_list_with_enc_dec_aux:
            logging.info('Epoch: {}, val_gen_losses: {:.5f}, val_lat_losses: {:.5f}, val_res_losses: {:.5f}, val_euler_losses: {:.5f}, val_trace_losses: {:.5f}, val_losses: {:.5f}'.format(epoch,  val_gen_losses/number_of_batches,val_lat_losses/number_of_batches, val_res_losses/number_of_batches, val_euler_losses/number_of_batches, val_trace_losses/number_of_batches, val_losses/number_of_batches))
            if config['use_wandb']:
                wandb.log({ 'val_gen_losses': round(val_gen_losses/number_of_batches, 5) , 'val_lat_losses': round(val_lat_losses/number_of_batches, 5), 'val_res_losses': round(val_res_losses/number_of_batches, 5), 'val_euler_losses': round(val_euler_losses/number_of_batches, 5), 'val_trace_losses': round(val_trace_losses/number_of_batches, 5), 'val_losses': round(val_losses/number_of_batches, 5), 'best_val_score': best_val_score/number_of_batches})

        else:
            raise ValueError('Makes sure the model name is either in model_list_without_enc_dec_aux or model_list_with_enc_dec_aux')

    # Set the model back to train mode
    model.train()
    return best_val_score


# =================================================================================
# ============== Training function =============================================
# =================================================================================
def train(model: torch.nn.Module,
            data_dict: dict, 
            images_vl: np.ndarray,
            log_dir:str,
            already_completed_epochs:int,
            config,
            device: torch.device,
            optimizer: torch.optim.Optimizer,):
        
    # Set the model to train mode and send it to "device"
    model.train()
    logging.info('=============================================================================')
    logging.info('Training model')
    logging.info('=============================================================================')
    # ======= PRINT CONFIGURATION =======
    print(f"Model: {model}")
    print(f"Model name: {config['model_name']}")
    print(f"Preprocess method: {config['preprocess_method']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"do_data_agumentation", config['do_data_augmentation'])
    print(f"preprocess_method", config['preprocess_method'])
    print(f"use_synthetic_validation", config['use_synthetic_validation'])
    print(f"self_supervised", config['self_supervised'])
    print(f"use_scheduler", config['use_scheduler'])
    print(f'validation_metric_format', config['validation_metric_format'])
    print(f'validation_metric', config['val_metric'])
    print(f'List of indices of type of deformation to remove from validation since used in training: {config["indices_to_remove"]}')  
    # Auxiliary loss factor
    print(f'aux_loss_factor', config.get('aux_loss_factor', None))
    
    if config['use_wandb']:
        wandb.watch(model, log="all", log_freq=200)

    images_tr = data_dict['images_tr']

    if config['use_synthetic_validation']:
        # Set the best validation to zero
        best_val_score = 0
    else:
        # Set the best validation loss to infinity
        best_val_score = np.inf

    # If we continue training, then we load the best val score recorded so far
    if config['continue_training']:
        best_val_score = config['best_val_score']
        logging.info('Continuing training from epoch {} with best validation score {}'.format(already_completed_epochs, best_val_score))

    # If self supervised, generate a mask for the blending
    if config['self_supervised']:
        if config.get('get_neighbours', False):
            mask_shape = (60, images_tr.shape[1], images_tr.shape[2], images_tr.shape[3])
            mask_blending = create_cube_mask_4D(mask_shape, WH=30, depth=20, inside=True).astype(np.bool8) # (x,y,repeated_axis_z, t)
        else:

            mask_shape = images_tr.shape[1:-1] # (x,y,t)
            mask_blending = create_cube_mask(mask_shape, WH= 20, depth= 12,  inside=True).astype(np.bool8)
        criterion = torch.nn.BCEWithLogitsLoss()
        if config['use_wandb']:
            # Initiate wandb tables
            val_table_watch = wandb.Table(columns=["epoch", "input", "output", "mask"])
            tr_table_watch = wandb.Table(columns=["epoch", "input", "output", "mask"])
    else:
        if config['use_wandb']:
            # Initiate wandb tables
            val_table_watch = wandb.Table(columns=["epoch", "input", "output"])
            tr_table_watch = wandb.Table(columns=["epoch", "input", "output"])
        else:
            pass
    if config['use_scheduler']:
        scheduler = CosineAnnealingLR(optimizer,
                              T_max = config['epochs'] - already_completed_epochs, # Maximum number of iterations.
                             eta_min = 5e-4) # Minimum learning rate.
        logging.info('Scheduler:', str(scheduler.__class__.__name__))
        
    
    adjacent_batch_slices = None

    # Loop over the epochs
    for epoch in tqdm(range(already_completed_epochs, config['epochs'])):
        # Start at 1
        epoch += 1
        print('========================= Epoch {} / {} ========================='.format(epoch, config['epochs']))  
        

        # Loop over the batches
        
        mmd_factor_losses = 0
        gen_factor_losses = 0
        res_losses = 0
        lat_losses = 0
        losses = 0
        bce_losses = 0
        euler_losses = 0
        trace_losses = 0
        
        number_of_batches = 0
        # Set y to None
        y = None

        for nt, batch_dict in enumerate(iterate_minibatches(images_tr, config, data_augmentation=config['do_data_augmentation'], data_dict = data_dict)):


            input_images = batch_dict['X']
            batch_z_slice = batch_dict['batch_z_slice']
            y = batch_dict['Y']
            rotation_matrix = batch_dict['rotation_matrix']
            if y is not None:
                y = torch.from_numpy(y).transpose(1,4).transpose(2,4).transpose(3,4).to(device)
        
            batch_z_slice = torch.from_numpy(batch_z_slice).to(device)

        
            # If self supervised, apply blending - type of blending is defined within function.
            if config['self_supervised']:
                 
                if config.get('get_neighbours', False):
                    # We look for three indices from within a patient to blend into the three slices contained in batch element
                    indices_selected = get_random_and_neighbour_indices_train(images_tr.shape[0], config['batch_size'], config['spatial_size_z'])
                    # Get the images with neighbours
                    images_for_blend = get_combined_images(images_tr, indices_selected)

                    # The functions to blend when on 4D data does not work well with dimensions that are two low
                    # We thus repeat the the images on the z axis
                    # Then repeat on the z axis 
                    #images_for_blend = np.transpose(images_for_blend, (1,2,0,3,4))
                    images_for_blend = np.repeat(images_for_blend, 20, axis=1) # We'll have image size 60 (3*20)
                    # Same for inputs
                    #input_images = np.transpose(input_images, (1,2,0,3,4))
                    input_images = np.repeat(input_images, 20, axis=1) # We'll have image size 60 (3*20)
                    
                    adjacent_images, anomaly_masks = apply_blending(input_images, images_for_blend, mask_blending, config)
                    
                    # The adjacent_images has the prev, curr, next images 
                    # The input to the network is the middle for each batch element
                    input_images = adjacent_images[:, 1, ...]
                    # Same for anomaly masks
                    anomaly_masks = anomaly_masks[:, 1, ...]
                    # Transfer the anomaly_masks to "device" and add channel dim
                    anomaly_masks = torch.from_numpy(anomaly_masks).to(device).unsqueeze(dim =1)

                    # The adjacent images are of size [b,3,32,32,24,4] for ascending aorta
                    # We want to have the channel in the second dimension and add the new slices dimenison into the channel dimension
                    adjacent_batch_slices = torch.from_numpy(adjacent_images).to(device).transpose(1,5).transpose(2,5).transpose(3,5).transpose(4,5).float()
                    adjacent_batch_slices = adjacent_batch_slices.reshape(adjacent_batch_slices.shape[0], -1, adjacent_batch_slices.shape[3], adjacent_batch_slices.shape[4], adjacent_batch_slices.shape[5])
                    # size [b,c*3, 32,32,24]

                else:
                    # We only have one image per element in the batch and pick a random one from the training set
                
                    random_indices = np.random.choice(images_tr.shape[0], size=input_images.shape[0], replace=False)
                    sorted_indices = np.sort(random_indices)
                    images_for_blend =  images_tr[sorted_indices, ...]

                    # Apply blending
                    input_images, anomaly_masks = apply_blending(input_images, images_for_blend, mask_blending, config)
                    
                    # Transfer the anomaly_masks to "device" and add channel dim
                    anomaly_masks = torch.from_numpy(anomaly_masks).to(device).unsqueeze(dim =1)
            else:
                anomaly_masks = None
                
            # Transfer the input_images to "device"
            input_images = torch.from_numpy(input_images).transpose(1,4).transpose(2,4).transpose(3,4).to(device)
            # Transfer the rotation matrix to "device"
            rotation_matrix = torch.from_numpy(rotation_matrix).to(device).float()
            
            # Reset the gradients
            optimizer.zero_grad()
            
            input_dict = {'input_images': input_images.float(), 'batch_z_slice': batch_z_slice.float(), 'adjacent_batch_slices':adjacent_batch_slices,
                          'rotation_matrix': rotation_matrix}
            output_dict = model(input_dict) 
            if (epoch%(config['training_viz_frequency']) == 0) or (epoch ==1):
                # Visualization
                if nt%300 == 0:
                    if config['use_wandb']:
                        visualize_intermediate_results(epoch, input_images, output_dict['decoder_output'], tr_table_watch, labels=anomaly_masks)
                    save_inputs_outputs(nt, epoch, input_images, output_dict['decoder_output'], config, labels=anomaly_masks)

            # If doing a self-supervised task (anomaly detection)
            if config['self_supervised']:
                # Here we don't use VAE loss 
                dict_loss = {}
                # Compute the binary cross entropy loss
                dict_loss['bce_loss'] = criterion(output_dict['decoder_output'], anomaly_masks.float())
                dict_loss['loss'] = dict_loss['bce_loss']


                # Look into detaching theoutput because it is part of the graph 
                with torch.no_grad():
                # We add the contribution of the auxiliary loss
                # Compute the Euler angles and the trace of the rotation matrix
                    euler_trace_results= compute_euler_angles_xyz_and_trace(rotation_matrix)
                    euler_angles, trace = euler_trace_results['euler_angles'].detach(), euler_trace_results['trace'].detach()

                    # Add the euler angles and trace to the input dictionary
                    input_dict['euler_angles'] = euler_angles
                    input_dict['trace'] = trace

                dict_loss_aux = compute_losses(input_images, output_dict, config, input_dict)
                # Factor to adjust the importance of auxiliary losses
                
                dict_loss['loss'] += dict_loss_aux['total_aux_loss'].mean() # Average out the auxialiary loss over the batch
                
                euler_losses += dict_loss_aux['euler_loss'].mean().item()
                trace_losses += dict_loss_aux['trace_loss'].mean().item()
                bce_losses += dict_loss['bce_loss'].item() 


                
            else:

                # Check keys in output_dict to adapt loss computation
                # Compute the loss
                if config['model'] in model_list_without_enc_dec_aux:
                    dict_loss = compute_losses(input_images, output_dict, config)
                    gen_factor_losses += dict_loss['gen_factor_loss'].mean().item()
                    res_losses += dict_loss['res_loss'].mean().item()
                    lat_losses += dict_loss['lat_loss'].mean().item()

                elif config['model'] in model_list_with_enc_dec_aux:
                    
                    with torch.no_grad():
                        # Compute the Euler angles and the trace of the rotation matrix
                        euler_trace_results= compute_euler_angles_xyz_and_trace(rotation_matrix)
                        euler_angles, trace = euler_trace_results['euler_angles'].detach(), euler_trace_results['trace'].detach()

                        # Add the euler angles and trace to the input dictionary
                        input_dict['euler_angles'] = euler_angles
                        input_dict['trace'] = trace

                    dict_loss = compute_losses(input_images, output_dict, config, input_dict)
                    gen_factor_losses += dict_loss['gen_factor_loss'].mean().item()
                    res_losses += dict_loss['res_loss'].mean().item()
                    lat_losses += dict_loss['lat_loss'].mean().item()
                    euler_losses += dict_loss['euler_loss'].mean().item()
                    trace_losses += dict_loss['trace_loss'].mean().item()

                else:
                    raise ValueError('output_dict does not contain the correct keys')
            
            # Backward pass
            dict_loss['loss'].backward()

            # Clip gradients
            if config['max_grad_norm'] != 'None':
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])

            # Update the weights
            optimizer.step()
            # Compute the losses
            losses += dict_loss['loss'].item()
            number_of_batches += 1
        if config['use_scheduler']:
            if config['use_wandb']:
                # Log the learning rate
                wandb.log({ "lr": scheduler.get_last_lr()[0]})
            scheduler.step()
            
                        
        if (epoch%(config['training_frequency']) == 0) or (epoch ==1):
            # Check if self supervised
            if config['self_supervised']:

                if config['model'] in model_list_with_enc_dec_aux:

                    # Logging the training losses including the auxiliary losses
                    logging.info('Epoch: {}, train_loss: {:.5f}, train_bce_loss: {:.5f}, train_euler_loss: {:.5f}, train_trace_loss: {:.5f}'.format(epoch, losses/number_of_batches, bce_losses/number_of_batches, (euler_losses)/number_of_batches, (trace_losses)/number_of_batches))
                    if config['use_wandb']:
                        wandb.log({'train_loss': round(losses/number_of_batches, 5), 'train_euler_loss': round((euler_losses)/number_of_batches, 5), 'train_trace_loss': round((trace_losses)/number_of_batches, 5)})
                    
                else:
                    logging.info('Epoch: {}, train_loss: {:.5f}'.format(epoch, losses/number_of_batches))
                    if config['use_wandb']:
                        wandb.log({'train_loss': round(losses/number_of_batches, 5)})
            else:
                if config['model'] in model_list_without_enc_dec_aux:
                    logging.info('Epoch: {}, train_gen_factor_loss: {:.5f},train_lat_loss: {:.5f}, train_res_loss: {:.5f}, train_loss: {:.5f}'.format(epoch, gen_factor_losses/number_of_batches, lat_losses/number_of_batches, res_losses/number_of_batches, losses/number_of_batches))
                    if config['use_wandb']:
                        wandb.log({'train_gen_factor_loss': round(gen_factor_losses/number_of_batches, 5), 'train_lat_loss': round(lat_losses/number_of_batches, 5), 'train_res_loss': round(res_losses/number_of_batches, 5), 'train_loss': round(losses/number_of_batches, 5)})

                elif config['model'] in model_list_with_enc_dec_aux:
                    logging.info('Epoch: {}, train_gen_factor_loss: {:.5f},train_lat_loss: {:.5f}, train_res_loss: {:.5f}, train_euler_loss: {:.5f}, train_trace_loss: {:.5f}, train_loss: {:.5f}'.format(epoch, gen_factor_losses/number_of_batches, lat_losses/number_of_batches, res_losses/number_of_batches, (euler_losses)/number_of_batches, (trace_losses)/number_of_batches, losses/number_of_batches))
                    if config['use_wandb']:
                        wandb.log({'train_gen_factor_loss': round(gen_factor_losses/number_of_batches, 5), 'train_lat_loss': round(lat_losses/number_of_batches, 5), 'train_res_loss': round(res_losses/number_of_batches, 5), 'train_euler_loss': round((euler_losses)/number_of_batches, 5), 'train_trace_loss': round((trace_losses)/number_of_batches, 5), 'train_loss': round(losses/number_of_batches, 5)})


            # Save the model
            checkpoint(model, os.path.join(log_dir, f"{config['model_name']}.ckpt-{epoch}"))

        # Evaluate the model on the validation set
        if (epoch%(config['validation_frequency']) == 0) or (epoch ==1):
            if not config['use_wandb']:
                val_table_watch = None
            best_val_score = evaluate(model, epoch, images_vl, best_val_score, log_dir, config, device, val_table_watch, data_dict=data_dict)

        torch.cuda.empty_cache()
    if config['use_wandb']:
        wandb.log({"val_table": val_table_watch})
        wandb.log({"tr_table": tr_table_watch})

# Saving/loading the model
def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, checkpoint_path, config, device):
    model.load_state_dict(torch.load(checkpoint_path+'/{}.ckpt-{}'.format(config['model_name'], config['latest_model_epoch']), map_location=device))
    return model


# ==================================================================
# Helper functions for visualization during training
# ==================================================================


def visualize_intermediate_results(epoch, input_images, ouput_images, table_watch, labels=None, table_dict=None):
    # Make labels into numpy array if not already
    if labels is not None:
        # Apply sigmoid to output (we need to apply the simgoid to the output since we use the BCEWithLogitsLoss (numpy ))
        ouput_images = torch.sigmoid(ouput_images)
        labels = labels.cpu().detach().numpy()
    # Transfer the input_images to "cpu"
    input_cpu = input_images.cpu().detach().numpy()
    output_cpu = ouput_images.cpu().detach().numpy()
    # Check how many channels we have
    channels = ouput_images.shape[1]

    # Map the colors to have the same scale
    input_cmap = input_cpu
    output_cmap = output_cpu
    # Plot random images in wandb table

    if channels == 1:
        # Self supervised, then we also want to plot the labels mask and increase the number of images to log
        


        for i in range(4):
            random_index = np.random.randint(0, input_cpu.shape[0])
            random_time = np.random.randint(0, input_cpu.shape[4])
            fig0, ax = plt.subplots()
            im = ax.imshow(input_cmap[random_index, 0, :,:, random_time])
            cbar = fig0.colorbar(im)
            out_fig0, ax = plt.subplots()
            im = ax.imshow(output_cmap[random_index, 0, :,:, random_time])
            cbar = out_fig0.colorbar(im)
            # Plot mask as well
            mask_fig0, ax = plt.subplots()
            im = ax.imshow(labels[random_index, 0, :,:, random_time])
            cbar = mask_fig0.colorbar(im)
            table_watch.add_data(epoch, wandb.Image(fig0), wandb.Image(out_fig0), wandb.Image(mask_fig0))
            plt.close(fig0)
            plt.close(out_fig0)
            plt.close(mask_fig0)
    else:
            
        for i in range(2):
            random_index = np.random.randint(0, input_cpu.shape[0])
            random_time = np.random.randint(0, input_cpu.shape[4])
            # Manually add a colorbar using Matplotlib
            fig0, ax = plt.subplots()
            im = ax.imshow(input_cmap[random_index, 0, :,:, random_time])
            cbar = fig0.colorbar(im)
            
            out_fig0, ax = plt.subplots()
            im = ax.imshow(output_cmap[random_index, 0, :,:, random_time])
            cbar = out_fig0.colorbar(im)
            table_watch.add_data(epoch, wandb.Image(fig0), wandb.Image(out_fig0))
            plt.close(fig0)
            plt.close(out_fig0)
            fig1, ax = plt.subplots()
            im = ax.imshow(input_cmap[random_index, 1, :,:, random_time])
            cbar = fig1.colorbar(im)
            out_fig1, ax = plt.subplots()
            im = ax.imshow(output_cmap[random_index, 1, :,:, random_time])
            cbar = out_fig1.colorbar(im)
            table_watch.add_data(epoch, wandb.Image(fig1), wandb.Image(out_fig1))
            plt.close(fig1)
            plt.close(out_fig1)

            fig2, ax = plt.subplots()
            im = ax.imshow(input_cmap[random_index, 2, :,:, random_time])
            cbar = fig2.colorbar(im)
            out_fig2, ax = plt.subplots()
            im = ax.imshow(output_cmap[random_index, 2, :,:, random_time])
            cbar = out_fig2.colorbar(im)
            table_watch.add_data(epoch, wandb.Image(fig2), wandb.Image(out_fig2))
            plt.close(fig2)
            plt.close(out_fig2)
  
