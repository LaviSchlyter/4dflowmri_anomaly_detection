import torch
import numpy as np
import logging
import os
from tqdm import tqdm
import wandb
import matplotlib
from matplotlib import pyplot as plt

# =================================================================================
# ============== HELPER FUNCTIONS =============================================
from batches import iterate_minibatches
from utils import compute_losses_VAE, make_dir_safely, compute_losses_TVAE, compute_losses_MMDVAE \
    , apply_poisson_blending
from metrics import compute_auc_roc_score, compute_average_precision_score
from synthetic_anomalies import create_cube_mask


# Set colormap for consistency
cmapper_gray = matplotlib.cm.get_cmap("gray")
# =================================================================================
# ============== Evaluation function =============================================
# =================================================================================
def evaluate(model,epoch, images_vl, best_val_score, log_dir,config, device, val_table_watch):
    # Set the model to eval mode
    model.eval()
    
    val_kld_losses = 0
    val_mmd_losses = 0
    val_gen_losses = 0
    val_res_losses = 0
    val_lat_losses = 0
    val_losses = 0
    number_of_batches = 0
    images_error = []
    val_masks = []

    # set y to None
    y = None
    # Loop over the batches
    # No data augmentation for validation set
    for nv, batch in enumerate(iterate_minibatches(images_vl, config['batch_size'], data_augmentation=False, with_labels= config['use_synthetic_validation'], remove_indices= config['self_supervised'])):

        with torch.no_grad():
            if isinstance(batch, tuple):
                # We have the labels
                input_images, y = batch  
                y = torch.from_numpy(y).transpose(1,4).transpose(2,4).transpose(3,4).to(device)
            else:
                input_images = batch

            # Transfer the input_images to "device"
            input_images = torch.from_numpy(input_images).transpose(1,4).transpose(2,4).transpose(3,4).to(device)
            # Forward pass
            output_dict = model(input_images.float())
            # Visualization
            if nv%100 == 0:
                visualize(epoch, input_images, output_dict['decoder_output'], val_table_watch, labels=y)
                save_inputs_outputs(nv, epoch, input_images, output_dict['decoder_output'], config,labels=y)
            
            if config['use_synthetic_validation'] and not config['self_supervised']:
                # Take the difference between input and output (for the unsupervised way)
                images_error.append(torch.abs(input_images - output_dict['decoder_output']).cpu().detach().numpy())
                val_masks.append(y)
            

            # Check keys in output_dict to adapt loss computation
            # Compute the loss
            """
            When evaluating the model, we want to look at the losses without the factors
            """
            if config['self_supervised']:
                # For the self-supervised the output of network is already the error
                images_error.append(output_dict['decoder_output'].cpu().detach().numpy())
                val_masks.append(y.cpu().detach().numpy())
            else:

                if config['model'] == 'vae':
                    dict_loss = compute_losses_VAE(input_images, output_dict, config)
                    val_gen_losses += dict_loss['gen_loss'].mean().item()
                    val_res_losses += dict_loss['res_loss'].mean().item()
                    val_lat_losses += dict_loss['lat_loss'].mean().item()
                elif config['model'] == 'tvae':
                    dict_loss = compute_losses_TVAE(input_images, output_dict, config)
                    val_gen_losses += dict_loss['gen_loss'].mean().item()
                    val_kld_losses += dict_loss['kld_loss'].mean().item()
                elif config['model'] == 'mmd_vae':
                    dict_loss = compute_losses_MMDVAE(input_images, output_dict, config, device)
                    val_gen_losses += dict_loss['gen_loss'].mean().item()
                    val_mmd_losses += dict_loss['mmd_loss'].mean().item()
                else:
                    raise ValueError('output_dict does not contain the correct keys')
            
                # Val loss does not have the factors - this enables better comparaison

                val_losses += dict_loss['val_loss'].item()
            number_of_batches += 1
    if config['use_synthetic_validation']:
        logging.info('Computing validation metrics')
        # Compute ROC AUC score
        auc_roc = compute_auc_roc_score(images_error, val_masks, config)
        logging.info('Epoch: {}, val_auc_roc: {:.5f}'.format(epoch, auc_roc))
        wandb.log({'val_auc_roc': round(auc_roc, 5)})
        # Compute average precision score
        ap = compute_average_precision_score(images_error, val_masks, config)
        logging.info('Epoch: {}, val_ap: {:.5f}'.format(epoch, ap))
        wandb.log({'val_ap': round(ap, 5)})


        if config['val_metric'] == 'auc_roc':
            current_score = auc_roc
        elif config['val_metric'] == 'AP':
            current_score = ap
        else:
            raise ValueError('val_metric not recognized', config['val_metric'])
            
        # Save the model if the validation score is the best we've seen so far.
        if current_score > best_val_score:
            logging.info('Saving best model at epoch {}'.format(epoch))
            wandb.run.summary['best_val_epoch'] = epoch
            best_val_score = current_score
            checkpoint(model, os.path.join(log_dir, f"{config['model_name']}.ckpt-best"))
    else:
        if val_losses < best_val_score:
            logging.info('Saving best model at epoch {}'.format(epoch))
            wandb.run.summary['best_val_epoch'] = epoch
            best_val_score = val_losses
            checkpoint(model, os.path.join(log_dir, f"{config['model_name']}.ckpt-best"))
    if config['self_supervised']:
        # We don't have the same losses 
        pass
    else:

        
        if config['model'] == 'vae':
            # VAE model
            # Logging the validation losses
            logging.info('Epoch: {}, val_gen_losses: {:.5f}, val_lat_losses: {:.5f}, val_res_losses: {:.5f}, val_losses: {:.5f}'.format(epoch,  val_gen_losses/number_of_batches,val_lat_losses/number_of_batches, val_res_losses/number_of_batches, val_losses/number_of_batches))
            wandb.log({ 'val_gen_losses': round(val_gen_losses/number_of_batches, 5) , 'val_lat_losses': round(val_lat_losses/number_of_batches, 5), 'val_res_losses': round(val_res_losses/number_of_batches, 5), 'val_losses': round(val_losses/number_of_batches, 5), 'best_val_score': best_val_score/number_of_batches})
        elif config['model'] == 'tvae':
            # TVAE model
            # Logging the validation losses
            logging.info('Epoch: {}, val_gen_losses: {:.5f}, val_kld_losses: {:.5f}, val_losses: {:.5f}'.format(epoch,  val_gen_losses/number_of_batches,val_kld_losses/number_of_batches, val_losses/number_of_batches))
            wandb.log({ 'val_gen_losses': round(val_gen_losses/number_of_batches, 5) , 'val_kld_losses': round(val_kld_losses/number_of_batches, 5), 'val_losses': round(val_losses/number_of_batches, 5), 'best_val_score': best_val_score/number_of_batches})
        elif config['model'] == 'mmd_vae':
            # MMDVAE model
            # Logging the validation losses
            logging.info('Epoch: {}, val_gen_losses: {:.5f}, val_mmd_losses: {:.5f}, val_losses: {:.5f}'.format(epoch,  val_gen_losses/number_of_batches,val_mmd_losses/number_of_batches, val_losses/number_of_batches))
            wandb.log({ 'val_gen_losses': round(val_gen_losses/number_of_batches, 5) , 'val_mmd_losses': round(val_mmd_losses/number_of_batches, 5), 'val_losses': round(val_losses/number_of_batches, 5), 'best_val_score': best_val_score/number_of_batches})

    # Set the model back to train mode
    model.train()
    return best_val_score


# =================================================================================
# ============== Training function =============================================
# =================================================================================
def train(model: torch.nn.Module,
            images_tr: np.ndarray, 
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
    


    if config['use_synthetic_validation']:
        # Set the best validation to zero
        best_val_score = 0
    else:
        # Set the best validation loss to infinity
        best_val_score = np.inf
    # If self supervised, generate a mask for the blending
    if config['self_supervised']:
        mask_shape = images_tr.shape[1:-1] # (x,y,t)
        mask_blending = create_cube_mask(mask_shape, WH= 20, depth= 12,  inside=True).astype(np.bool8)
        criterion = torch.nn.BCEWithLogitsLoss()
        # Initiate wandb tables
        val_table_watch = wandb.Table(columns=["epoch", "input", "output", "mask"])
        tr_table_watch = wandb.Table(columns=["epoch", "input", "output", "mask"])
        val_table = []
        tr_table = []
    else:
        # Initiate wandb tables
        val_table_watch = wandb.Table(columns=["epoch", "input", "output"])
        tr_table_watch = wandb.Table(columns=["epoch", "input", "output"])
        
    # Loop over the epochs
    for epoch in tqdm(range(already_completed_epochs, config['epochs'])):
        print('========================= Epoch {} / {} ========================='.format(epoch + 1, config['epochs']))  


        # Loop over the batches
        kld_losses = 0
        mmd_factor_losses = 0
        gen_factor_losses = 0
        res_losses = 0
        lat_losses = 0
        losses = 0

        
        number_of_batches = 0
        # Set y to None
        y = None
        

        for nt, batch in enumerate(iterate_minibatches(images_tr, config['batch_size'], data_augmentation=config['do_data_augmentation'])):
            if isinstance(batch, tuple):
                # We have the labels
                input_images, y = batch  
                #y = np.transpose(y, (0,4,1,2,3))
                y = torch.from_numpy(y).transpose(1,4).transpose(2,4).transpose(3,4).to(device)
            else:
                input_images = batch


            # If self supervised, apply poisson blending
            if config['self_supervised']:
                # Batch size
                random_indices = np.arange(input_images.shape[0])
                np.random.shuffle(random_indices)
                sorted_indices = np.sort(random_indices)
                images_for_blend =  images_tr[sorted_indices, ...]
                # Apply poisson blending
                input_images, anomaly_masks = apply_poisson_blending(input_images, images_for_blend, mask_blending)
                # Transfer the anomaly_masks to "device" and add channel dim
                anomaly_masks = torch.from_numpy(anomaly_masks).to(device).unsqueeze(dim =1)
                
            # Transfer the input_images to "device"
            input_images = torch.from_numpy(input_images).transpose(1,4).transpose(2,4).transpose(3,4).to(device)
            
            # Reset the gradients
            optimizer.zero_grad()
            # Forward pass
            #ouput_images, z_mean, z_std, res = model(input_images.float()) #VAE
            output_dict = model(input_images.float()) 
            # Visualization
            if nt%100 == 0:
                visualize(epoch, input_images, output_dict['decoder_output'], tr_table_watch, labels=anomaly_masks)
                save_inputs_outputs(nt, epoch, input_images, output_dict['decoder_output'], config, labels=anomaly_masks)

            # If doing a self-supervised task (anomaly detection)
            if config['self_supervised']:
                dict_loss = {}
                # Compute the binary cross entropy loss
                dict_loss['loss'] = criterion(output_dict['decoder_output'], anomaly_masks.float())
                
            else:

                # Check keys in output_dict to adapt loss computation
                # Compute the loss
                if config['model'] == 'vae':
                    dict_loss = compute_losses_VAE(input_images, output_dict, config)
                    gen_factor_losses += dict_loss['gen_factor_loss'].mean().item()
                    res_losses += dict_loss['res_loss'].mean().item()
                    lat_losses += dict_loss['lat_loss'].mean().item()
                elif config['model'] == 'tvae':
                    dict_loss = compute_losses_TVAE(input_images, output_dict, config)
                    gen_factor_losses += dict_loss['gen_factor_loss'].mean().item()
                    kld_losses += dict_loss['kld_loss'].mean().item()
                elif config['model'] == 'mmd_vae':
                    dict_loss = compute_losses_MMDVAE(input_images, output_dict, config, device)
                    gen_factor_losses += dict_loss['gen_factor_loss'].mean().item()
                    mmd_factor_losses += dict_loss['mmd_factor_loss'].mean().item()
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
            
                        
        if epoch%(config['training_frequency']-1) == 0:
            # Check if self supervised
            if config['self_supervised']:
                logging.info('Epoch: {}, train_loss: {:.5f}'.format(epoch, losses/number_of_batches))
                wandb.log({'train_loss': round(losses/number_of_batches, 5)})
            else:
                if config['model'] == 'vae':
                    logging.info('Epoch: {}, train_gen_factor_loss: {:.5f},train_lat_loss: {:.5f}, train_res_loss: {:.5f}, train_loss: {:.5f}'.format(epoch, gen_factor_losses/number_of_batches, lat_losses/number_of_batches, res_losses/number_of_batches, losses/number_of_batches))

                    wandb.log({'train_gen_factor_loss': round(gen_factor_losses/number_of_batches, 5), 'train_lat_loss': round(lat_losses/number_of_batches, 5), 'train_res_loss': round(res_losses/number_of_batches, 5), 'train_loss': round(losses/number_of_batches, 5)})

                elif config['model'] == 'tvae':
                    logging.info('Epoch: {}, train_gen_factor_loss: {:.5f},train_kld_loss: {:.5f}, train_loss: {:.5f}'.format(epoch, gen_factor_losses/number_of_batches, kld_losses/number_of_batches, losses/number_of_batches))

                    wandb.log({'train_gen_factor_loss': round(gen_factor_losses/number_of_batches, 5), 'train_kld_loss': round(kld_losses/number_of_batches, 5), 'train_loss': round(losses/number_of_batches, 5)})
                elif config['model'] == 'mmd_vae':
                    logging.info('Epoch: {}, train_gen_factor_loss: {:.5f},train_mmd_factor_loss: {:.5f}, train_loss: {:.5f}'.format(epoch, gen_factor_losses/number_of_batches, mmd_factor_losses/number_of_batches, losses/number_of_batches))

                    wandb.log({'train_gen_factor_loss': round(gen_factor_losses/number_of_batches, 5), 'train_mmd_factor_loss': round(mmd_factor_losses/number_of_batches, 5), 'train_loss': round(losses/number_of_batches, 5)})
                                # Save the model
            checkpoint(model, os.path.join(log_dir, f"{config['model_name']}.ckpt-{epoch}"))

        # Evaluate the model on the validation set
        if epoch%(config['validation_frequency']-1) == 0:
            best_val_score = evaluate(model, epoch, images_vl, best_val_score, log_dir, config, device, val_table_watch)
            # TODO: implement visualization of the latent space ? 
            #wandb.log({"val_table": val_table_watch})
            #wandb.log({"tr_table": tr_table_watch})

        torch.cuda.empty_cache()
            
    wandb.log({"val_table": val_table_watch})
    wandb.log({"tr_table": tr_table_watch})

# Saving/loading the model
def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, checkpoint_path, config, device):
    model.load_state_dict(torch.load(checkpoint_path+'/{}.ckpt-{}'.format(config['model_name'], config['latest_model_epoch']), map_location=device))
    return model

def visualize(epoch, input_images, ouput_images, table_watch, labels=None):
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
    input_cmap =input_cpu
    output_cmap =output_cpu
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
  

def save_inputs_outputs(n_image, epoch, input_, ouput_, config, labels=None):
    path_inter_inputs = os.path.join(config['exp_path'], 'intermediate_results/inputs')
    path_inter_outputs = os.path.join(config['exp_path'], 'intermediate_results/outputs')
    
    make_dir_safely(path_inter_inputs)
    make_dir_safely(path_inter_outputs)
    if config['self_supervised']:
        # Apply sigmoid to output
        ouput_ = torch.sigmoid(ouput_)
        path_inter_masks = os.path.join(config['exp_path'], 'intermediate_results/masks')
        make_dir_safely(path_inter_masks)
        labels = labels.cpu().detach().numpy()
        np.save(os.path.join(path_inter_masks,f"mask_image_{n_image}_epoch_{epoch}.npy"), labels)


    input_cpu = input_.cpu().detach().numpy()
    output_cpu = ouput_.cpu().detach().numpy()
    np.save(os.path.join(path_inter_inputs,f"input_image_{n_image}_epoch_{epoch}.npy"), input_cpu)
    np.save(os.path.join(path_inter_outputs,f"output_image_{n_image}_epoch_{epoch}.npy"), output_cpu)
    
