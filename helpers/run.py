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
            if nv%100 == 0:
                visualize(epoch, input_images, ouput_images, val_table_watch)
                save_inputs_outputs(nv, epoch, input_images, ouput_images, config)
            # Compute the loss
            gen_loss, res_loss, lat_loss = compute_losses(input_images, ouput_images, z_mean, z_std, res)
            gen_factor_loss = config['gen_loss_factor']*gen_loss
            loss = torch.mean(gen_factor_loss + lat_loss)
            val_gen_losses += gen_loss.mean().item()
            val_gen_factor_losses += gen_factor_loss.mean().item()
            val_res_losses += res_loss.mean().item()
            val_lat_losses += lat_loss.mean().item()
            val_losses += loss.item()
            number_of_batches += 1

    if val_losses < best_val_loss:
        logging.info('Saving best model at epoch {}'.format(epoch))
        best_val_loss = val_losses
        checkpoint(model, os.path.join(log_dir, f"{config['model_name']}.ckpt-best"))

        
    

    # Logging the validation losses
    logging.info('Epoch: {}, val_gen_losses: {:.5f}, val_gen_factor_losses: {:.5f}, val_lat_losses: {:.5f}, val_res_losses: {:.5f}, val_losses: {:.5f}'.format(epoch, val_gen_losses/number_of_batches, val_gen_factor_losses/number_of_batches,val_lat_losses/number_of_batches, val_res_losses/number_of_batches, val_losses/number_of_batches))
    wandb.log({'val_gen_losses': round(val_gen_losses/number_of_batches, 5), 'val_gen_factor_losses': round(val_gen_factor_losses/number_of_batches, 5) , 'val_lat_losses': round(val_lat_losses/number_of_batches, 5), 'val_res_losses': round(val_res_losses/number_of_batches, 5), 'val_losses': round(val_losses/number_of_batches, 5), 'best_val_loss': best_val_loss/number_of_batches})
    
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
            if nt%500 == 0:
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
            gen_losses += gen_loss.mean().item()
            gen_factor_losses += gen_factor_loss.mean().item()
            res_losses += res_loss.mean().item()
            lat_losses += lat_loss.mean().item()
            losses += loss.item()
            number_of_batches += 1
                        
        if epoch%20 == 0:

            logging.info('Epoch: {}, train_gen_loss: {:.5f}, train_gen_factor_loss: {:.5f},train_lat_loss: {:.5f}, train_res_loss: {:.5f}, train_loss: {:.5f}'.format(epoch, gen_losses/number_of_batches,gen_factor_losses/number_of_batches, lat_losses/number_of_batches, res_losses/number_of_batches, losses/number_of_batches))

            wandb.log({'train_gen_loss': round(gen_losses/number_of_batches, 5),'train_gen_factor_loss': round(gen_factor_losses/number_of_batches, 5), 'train_lat_loss': round(lat_losses/number_of_batches, 5), 'train_res_loss': round(res_losses/number_of_batches, 5), 'train_loss': round(losses/number_of_batches, 5)})
            # Save the model
            checkpoint(model, os.path.join(log_dir, f"{config['model_name']}.ckpt-{epoch}"))

        # Evaluate the model on the validation set
        if epoch%config['validation_frequency'] == 0:
            evaluate(model, epoch, images_vl, best_val_loss, log_dir, config, device, val_table_watch)
            # TODO: implement visualization of the latent space ? 

        torch.cuda.empty_cache()
            
    wandb.log({"val_table": val_table_watch})
    wandb.log({"tr_table": tr_table_watch})

# Saving/loading the model
def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, checkpoint_path, config):
    model.load_state_dict(torch.load(checkpoint_path+'/{}.ckpt-{}'.format(config['model_name'], config['latest_model_epoch'])))
    return model

def visualize(epoch, input_images, ouput_images, table_watch):
    input_cpu = input_images.cpu().detach().numpy()
    output_cpu = ouput_images.cpu().detach().numpy()

    # Map the colors to have the same scale
    input_cmap =input_cpu
    output_cmap =output_cpu
    # Plot random images in wandb table
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


        # Log the figure with the colorbar using wandb.Image
        #wandb.log({"Input Image": wandb.Image(fig0, caption="Intensity channel"), "Output Image": wandb.Image(out_fig0, caption="Intensity channel"), "Input Image": wandb.Image(fig1, caption="vx channel"), "Output Image": wandb.Image(out_fig1, caption="vx channel")})
        

        #table_watch.add_data(epoch, wandb.Image(input_cmap[random_index, 0, :,:, random_time]), wandb.Image(output_cmap[random_index, 0, :,:, random_time]))
        
        #table_watch.add_data(epoch, wandb.Image(input_cmap[random_index, 1, :,:, random_time]), wandb.Image(output_cmap[random_index, 1, :,:, random_time]))
  
        #table_watch.add_data(epoch, wandb.Image(input_cmap[random_index, 2, :,:, random_time]), wandb.Image(output_cmap[random_index, 2, :,:, random_time]))
  
  


    

def save_inputs_outputs(n_image, epoch, input_, ouput_, config):
    path_inter_inputs = os.path.join(config['exp_path'], 'intermediate_results/inputs')
    path_inter_outputs = os.path.join(config['exp_path'], 'intermediate_results/outputs')
    make_dir_safely(path_inter_inputs)
    make_dir_safely(path_inter_outputs)

    input_cpu = input_.cpu().detach().numpy()
    output_cpu = ouput_.cpu().detach().numpy()
    np.save(os.path.join(path_inter_inputs,f"input_image_{n_image}_epoch_{epoch}.npy"), input_cpu)
    np.save(os.path.join(path_inter_outputs,f"output_image_{n_image}_epoch_{epoch}.npy"), output_cpu)