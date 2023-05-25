import torch
import numpy as np
import logging
import os
from tqdm import tqdm
import wandb
import matplotlib


# =================================================================================
# ============== HELPER FUNCTIONS =============================================
from batches import iterate_minibatches
from utils import compute_losses, make_dir_safely

# Set colormap for consistency
cmapper_gray = matplotlib.cm.get_cmap("gray")
# =================================================================================
# ============== Evaluation function =============================================
# =================================================================================
def evaluate(model,epoch, images_vl, best_val_loss, log_dir,config, device, val_table_watch):
    # Set the model to eval mode
    model.eval()
    val_gen_losses = 0
    val_gen_factor_losses = 0
    val_res_losses = 0
    val_lat_losses = 0
    val_losses = 0
    number_of_batches = 0
    # Loop over the batches
    for nv, batch in enumerate(iterate_minibatches(images_vl, config['batch_size'], data_augmentation=config['do_data_augmentation'])):
        with torch.no_grad():
            input_images = batch
            # Transfer the input_images to "device"
            input_images = torch.from_numpy(input_images).transpose(1,4).transpose(2,4).transpose(3,4).to(device)
            # Forward pass
            ouput_images, z_mean, z_std, res = model(input_images.float())
            # Visualization
            if nv%50 == 0:
                visualize(epoch, input_images, ouput_images, val_table_watch)
                save_inputs_outputs(nv, epoch, input_images, ouput_images, config)
            # Compute the loss
            gen_loss, res_loss, lat_loss = compute_losses(input_images, ouput_images, z_mean, z_std, res)
            gen_factor_loss = config['gen_loss_factor']*gen_loss
            loss = torch.mean(gen_factor_loss + lat_loss)
            val_gen_losses += gen_loss.mean()
            val_gen_factor_losses += gen_factor_loss.mean()
            val_res_losses += res_loss.mean()
            val_lat_losses += lat_loss.mean()
            val_losses += loss
            number_of_batches += 1

    if val_losses < best_val_loss:
        logging.info('Saving best model at epoch {}'.format(epoch))
        best_val_loss = val_losses
        checkpoint(model, os.path.join(log_dir, f"{config['model_name']}.ckpt-best"))

        
    

    # Logging the validation losses
    logging.info('Epoch: {}, val_gen_losses: {:.3f}, val_gen_factor_losses: {:.3f}, val_lat_losses: {:.3f}, val_res_losses: {:.3f}, val_losses: {:.3f}'.format(epoch, val_gen_losses/number_of_batches, val_gen_factor_losses/number_of_batches,val_lat_losses/number_of_batches, val_res_losses/number_of_batches, val_losses/number_of_batches))
    wandb.log({'val_gen_losses': torch.round(val_gen_losses/number_of_batches, decimals = 3), 'val_gen_factor_losses': torch.round(val_gen_factor_losses/number_of_batches, decimals=3), 'val_lat_losses': torch.round(val_lat_losses/number_of_batches, decimals = 3), 'val_res_losses': torch.round(val_res_losses/number_of_batches, decimals = 3), 'val_losses': torch.round(val_losses/number_of_batches, decimals = 3), 'best_val_loss': best_val_loss/number_of_batches})
    
    # Set the model back to train mode
    model.train()


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

    # Initiate wandb tables
    val_table_watch = wandb.Table(columns=["epoch", "input", "output"])
    tr_table_watch = wandb.Table(columns=["epoch", "input", "output"])

    # Set the best validation loss to infinity
    best_val_loss = np.inf
    # Loop over the epochs
    for epoch in tqdm(range(already_completed_epochs, config['epochs'])):
        print('========================= Epoch {} / {} ========================='.format(epoch + 1, config['epochs']))  


        # Loop over the batches
        gen_losses = 0
        gen_factor_losses = 0
        res_losses = 0
        lat_losses = 0
        losses = 0
        
        number_of_batches = 0
        for nt, batch in enumerate(iterate_minibatches(images_tr, config['batch_size'], data_augmentation=config['do_data_augmentation'])):
            input_images = batch

            # Transfer the input_images to "device"
            input_images = torch.from_numpy(input_images).transpose(1,4).transpose(2,4).transpose(3,4).to(device)
            # Reset the gradients
            optimizer.zero_grad()
            # Forward pass
            ouput_images, z_mean, z_std, res = model(input_images.float())
            # Visualization
            if nt%50 == 0:
                visualize(epoch, input_images, ouput_images, tr_table_watch)
                save_inputs_outputs(nt, epoch, input_images, ouput_images, config)
            # Compute the loss
            gen_loss, res_loss, lat_loss = compute_losses(input_images, ouput_images, z_mean, z_std, res)
            gen_factor_loss = config['gen_loss_factor']*gen_loss
            loss = torch.mean(gen_factor_loss + lat_loss)
            # Backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
            # Compute the loss
            gen_losses += gen_loss.mean()
            gen_factor_losses += gen_factor_loss.mean()
            res_losses += res_loss.mean()
            lat_losses += lat_loss.mean()
            losses += loss
            number_of_batches += 1
            

        if epoch%10 == 0:
            
            logging.info('Epoch: {}, train_gen_loss: {:.3f}, train_gen_factor_loss: {:.3f},train_lat_loss: {:.3f}, train_res_loss: {:.3f}, train_loss: {:.3f}'.format(epoch, gen_losses/number_of_batches,gen_factor_losses/number_of_batches, lat_losses/number_of_batches, res_losses/number_of_batches, losses/number_of_batches))

            wandb.log({'train_gen_loss': torch.round(gen_losses/number_of_batches, decimals = 3),'train_gen_factor_loss': torch.round(gen_factor_losses/number_of_batches, decimals = 3), 'train_lat_loss': torch.round(lat_losses/number_of_batches, decimals = 3), 'train_res_loss': torch.round(res_losses/number_of_batches, decimals = 3), 'train_loss': torch.round(losses/number_of_batches, decimals = 3)})
            # Save the model
            checkpoint(model, os.path.join(log_dir, f"{config['model_name']}.ckpt-{epoch}"))

        # Evaluate the model on the validation set
        if epoch%config['validation_frequency'] == 0:
            evaluate(model, epoch, images_vl, best_val_loss, log_dir, config, device, val_table_watch)
            # TODO: implement visualization of the latent space ? 
            
    wandb.log({"val_table": val_table_watch})
    wandb.log({"tr_table": tr_table_watch})

# Saving/loading the model
def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, checkpoint_path, epoch, config):
    model.load_state_dict(torch.load(checkpoint_path+'/{}.ckpt-{}'.format(config['model_name'], epoch)))
    return model

def visualize(epoch, input_images, ouput_images, table_watch):
    input_cpu = input_images.cpu().detach().numpy()
    output_cpu = ouput_images.cpu().detach().numpy()

    # Map the colors to have the same scale
    input_cmap = cmapper_gray(input_cpu)
    output_cmap = cmapper_gray(output_cpu)
    input_cmap =input_cpu
    output_cmap =output_cpu
    # Plot random images in wandb table
    for i in range(2):
        random_index = np.random.randint(0, input_cpu.shape[0])
        random_time = np.random.randint(0, input_cpu.shape[4])
        table_watch.add_data(epoch, wandb.Image(input_cmap[random_index, 0, :,:, random_time]), wandb.Image(output_cmap[random_index, 0, :,:, random_time]))
        table_watch.add_data(epoch, wandb.Image(input_cmap[random_index, 1, :,:, random_time]), wandb.Image(output_cmap[random_index, 1, :,:, random_time]))
        table_watch.add_data(epoch, wandb.Image(input_cmap[random_index, 2, :,:, random_time]), wandb.Image(output_cmap[random_index, 2, :,:, random_time]))

        
        
        

def save_inputs_outputs(n_image, epoch, input_, ouput_, config):
    path_inter_inputs = os.path.join(config['exp_path'], 'intermediate_results/inputs')
    path_inter_outputs = os.path.join(config['exp_path'], 'intermediate_results/outputs')
    make_dir_safely(path_inter_inputs)
    make_dir_safely(path_inter_outputs)

    input_cpu = input_.cpu().detach().numpy()
    output_cpu = ouput_.cpu().detach().numpy()
    np.save(os.path.join(path_inter_inputs,f"input_image_{n_image}_epoch_{epoch}.npy"), input_cpu)
    np.save(os.path.join(path_inter_outputs,f"output_image_{n_image}_epoch_{epoch}.npy"), output_cpu)