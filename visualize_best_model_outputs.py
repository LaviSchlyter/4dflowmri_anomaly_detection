# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import sys
import torch
import h5py
import glob
import random
import logging
logging.basicConfig(level=logging.INFO)
# %%

from helpers.utils import make_dir_safely
from config import system as config_sys
from helpers.data_loader import load_data

# Import models
from models.vae import VAE, VAE_convT, VAE_linear
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/helpers')
from batches import plot_batch_3d_complete, plot_batch_3d_complete_1_chan 


# Download the methods for generating synthetic data
from helpers.synthetic_anomalies import generate_noise, create_cube_mask,\
                         generate_deformation_chan, create_hollow_noise



# For the patch blending we import from another directory
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/git_repos/many-tasks-make-light-work')
from multitask_method.tasks.patch_blending_task import TestPatchInterpolationBlender, \
    TestPoissonImageEditingMixedGradBlender, TestPoissonImageEditingSourceGradBlender

from multitask_method.tasks.labelling import FlippedGaussianLabeller


labeller = FlippedGaussianLabeller(0.2)


from helpers.visualization import plot_batches_SSL
#%%

def apply_patch_deformation(def_function, all_images, batch, mask_blending):
    random_indices = random.sample(range(len(all_images)), batch_size)
    random_indices = np.sort(random_indices)
    images_for_blending = all_images[random_indices]
    batch = np.transpose(batch, (0,4,1,2,3))
    images_for_blending = np.transpose(images_for_blending, (0,4,1,2,3))
    
    blended_images = []
    anomaly_masks = []
    for input_, blender in zip(batch, images_for_blending):
        
        blending_function = def_function(labeller, blender, mask_blending)
        blended_image, anomaly_mask = blending_function(input_, mask_blending)
        # Expand dims to add batch dimension
        blended_image = np.expand_dims(blended_image, axis=0)
        anomaly_mask = np.expand_dims(anomaly_mask, axis=0)
        blended_images.append(blended_image)
        anomaly_masks.append(anomaly_mask)
    batch = np.concatenate(blended_images, axis = 0)    
    labels = np.concatenate(anomaly_masks, axis = 0)
    batch = np.transpose(batch, (0,2,3,4,1))
    labels = np.expand_dims(labels, axis = -1)
    return batch, labels

def apply_deformation(deformation_list, data, save_dir, actions):
    start_idx = 0
    end_idx = batch_size
    mask_shape = [32, 32, 24]
    mask_blending = create_cube_mask(mask_shape, WH= 20, depth= 12,  inside=True).astype(np.bool_)
    for deformation in deformation_list:
        print(deformation)
        while end_idx <= data.shape[0]:
            
            batch = data[start_idx:end_idx]
            
            if deformation == 'None':
                labels = np.zeros(batch.shape)

            elif deformation == 'noisy':
                mean, std, noise = next(noise_generator)
                noise = noise/[10,1,1,1]
                batch = batch + noise
                labels = noise

            elif deformation == 'deformation':
                batch, labels = generate_deformation_chan(batch)

            elif deformation == 'hollow_circle':
                labels = create_hollow_noise(batch, mean=mean, std=std)
                
                batch = batch + labels
            elif deformation == 'patch_interpolation':
                batch, labels = apply_patch_deformation(TestPatchInterpolationBlender, data, batch, mask_blending)
            elif deformation == 'poisson_with_mixing':
                batch, labels = apply_patch_deformation(TestPoissonImageEditingMixedGradBlender, data, batch, mask_blending)
            elif deformation == 'poisson_without_mixing':
                batch, labels = apply_patch_deformation(TestPoissonImageEditingSourceGradBlender, data, batch, mask_blending)
                
            else:
                raise NotImplementedError
            
            if labels.shape[-1] == 1:
                labels = np.repeat(labels, 4, axis=-1)
            batch = torch.from_numpy(batch).transpose(1,4).transpose(2,4).transpose(3,4).float().to(device)
            

            with torch.no_grad():
                output_dict = model(batch)
                output_images = torch.sigmoid(output_dict['decoder_output'])
            output_images = output_images.transpose(1,2).transpose(2,3).transpose(3,4).cpu().detach().numpy()
            labels = labels#.transpose(1,2).transpose(2,3).transpose(3,4).cpu().detach().numpy()
            batch = batch.transpose(1,2).transpose(2,3).transpose(3,4).cpu().detach().numpy()
            channel_to_show = 1
            if "visualize" in actions:

                plot_batches_SSL(batch, output_images, labels, channel_to_show = channel_to_show, every_x_time_step=1, out_path=os.path.join(save_dir, 'batch_{}_to_{}_{}_c_{}.png'.format(start_idx, end_idx, deformation, channel_to_show)))
            start_idx += batch_size
            end_idx += batch_size
        start_idx = 0
        end_idx = batch_size

#%%


if __name__ == '__main__':
    #%%
    models_dir = "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/logs"

    list_of_experiments_paths = ["vae_convT/masked_slice/20230622-1538_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e800-bs8-gf_dim8-daFalse_interpolation_training",]
                                 #"vae_convT/masked_slice/20230622-1535_vae_convT_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse_interpolation_training",
                                 #"vae_convT/masked_slice/20230622-1521_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e800-bs8-gf_dim8-daFalse",
                                 #"vae_convT/masked_slice/20230622-1517_vae_convT_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse",
                                 #"vae/masked_slice/20230622-1542_vae_masked_slice_SSL_lr1.000e-03_scheduler-e800-bs8-gf_dim8-daFalse_interpolation_training",
                                 #"vae/masked_slice/20230622-1545_vae_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse_interpolation_training",
                                 #"vae/masked_slice/20230622-1558_vae_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse"]
    
    # Pick same random images
    keep_same_indices = True
    # Number of slices to visualize
    n_indices = 320
    # Batch size
    batch_size = 32
    actions = ["visualize", "evaluate"]

    data_to_visualize = ["test"] # ["validation", "healthy_unseen"] 

    # Load the data
    data_dir = '/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/data'
    data_vl = h5py.File(os.path.join(data_dir, 'masked_slice_anomalies_images_from_35_to_42.hdf5'), 'r')
    images_vl  = data_vl['images']
    labels_vl = data_vl['masks']
    # Load the two healthy subjects not seen during training/validation
    # Now the healthy unseen is on the test set

    data_healthy_unseen = h5py.File(os.path.join(data_dir, 'val_masked_sliced_images_from_42_to_44.hdf5'), 'r')
    images_healthy_unseen = data_healthy_unseen['sliced_images_val']

    # Load the test set
    # You need to create a config dict containing the preprocessing method
    config = {'preprocess_method': 'masked_slice'}
    _, _, images_test, labels_test = load_data(sys_config=config_sys, config=config, idx_start_vl=35, idx_end_vl=42,idx_start_ts=0, idx_end_ts=20, with_test_labels= True)
    #h5py.File(os.path.join(data_dir, 'test_masked_sliced_images_from_0_to_20.hdf5'), 'r')
    
    

    #%%
    # Logging the shapes
    logging.info(f"Validation images shape: {images_vl.shape}")
    logging.info(f"Validation labels shape: {labels_vl.shape}")
    logging.info(f"Healthy unseen images shape: {images_healthy_unseen.shape}")
    logging.info(f"Test images shape: {images_test.shape}")
    logging.info(f"Test labels shape: {labels_test.shape}")



    deformation_list = ['None', 'noisy', 'deformation', 'hollow_circle', 'patch_interpolation', 'poisson_with_mixing', 'poisson_without_mixing']
    noise_generator = generate_noise(batch_size = batch_size,mean_range = (0, 0.1), std_range = (0, 0.2), n_samples= images_healthy_unseen.shape[0]) 
    for model_rel_path in list_of_experiments_paths:

        model_path = os.path.join(models_dir, model_rel_path)
        pattern = os.path.join(model_path, "*best*")
        best_model_path = glob.glob(pattern)[0]

        model_str = model_rel_path.split("/")[0]
        preprocess_method = model_rel_path.split("/")[1]
        model_name = model_rel_path.split("/")[-1]
        
        # You'll need to load data here since it depends on the preprocessing method

        self_supervised = True if "SSL" in model_name else False

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
        else:
            raise ValueError("Model not recognized")
        
        # Load the model onto device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logging.info(f"device used: {device}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        logging.info(f"Model loaded from {best_model_path}")
        


        # Visualize validation 
        if "validation" in data_to_visualize:
            logging.info("Visualizing validation data...")
            results_dir_val = 'Results/Evaluation/' + model_str + '/' + preprocess_method + '/' + model_name + '/' + 'validation'
            make_dir_safely(results_dir_val)
            # Should we keep the same random images for all models?
            if keep_same_indices:
                random.seed(42)
                indices = random.sample(range(len(images_vl)), n_indices)
                indices.sort()
            else:
                indices = random.sample(range(len(images_vl)), n_indices)
                indices.sort()
            
            # Select indices, convert to torch and send to device
            input_images_vl = torch.from_numpy(images_vl[indices]).transpose(1,4).transpose(2,4).transpose(3,4).to(device)
            input_labels_vl = torch.from_numpy(labels_vl[indices]).transpose(1,4).transpose(2,4).transpose(3,4).to(device)

            start_idx = 0
            end_idx = batch_size

            while end_idx <= n_indices:
                batch = input_images_vl[start_idx:end_idx]
                labels = input_labels_vl[start_idx:end_idx]
                with torch.no_grad():
                    output_dict = model(batch)
                    output_images = torch.sigmoid(output_dict['decoder_output'])
                output_images = output_images.transpose(1,2).transpose(2,3).transpose(3,4).cpu().detach().numpy()
                labels = labels.transpose(1,2).transpose(2,3).transpose(3,4).cpu().detach().numpy()
                batch = batch.transpose(1,2).transpose(2,3).transpose(3,4).cpu().detach().numpy()
                
                # Check if actions include visualization
                if "visualize" in actions:
                    plot_batches_SSL(batch, output_images, labels, channel_to_show = 1,every_x_time_step=1, out_path=os.path.join(results_dir_val, 'batch_{}_to_{}.png'.format(start_idx, end_idx)))
                start_idx += batch_size
                end_idx += batch_size

        # Apply deformations and visualize healthy unseen
        if "healthy_unseen" in data_to_visualize:
            logging.info("Visualizing healthy unseen data...")
            results_dir_val_unseen = 'Results/Evaluation/' + model_str + '/' + preprocess_method + '/' + model_name + '/' + 'healthy_unseen'
            make_dir_safely(results_dir_val_unseen)
            apply_deformation(deformation_list, images_healthy_unseen, save_dir = results_dir_val_unseen, actions = actions )

        # Visualize the predictions on the test set (no artificial deformations)
        # Some are healthy and some are anomalous
        if "test" in data_to_visualize:
            logging.info("Visualizing test data...")
            results_dir_test = 'Results/Evaluation/' + model_str + '/' + preprocess_method + '/' + model_name + '/' + 'test'
            make_dir_safely(results_dir_test)
            n_indices = images_test.shape[0]
            start_idx = 0
            # We want the batch_size to be divisible by the number of images for the test set
            batch_size = 32
            end_idx = batch_size
            while end_idx <= n_indices:
                logging.info(start_idx, end_idx)
                batch = images_test[start_idx:end_idx]
                labels = labels_test[start_idx:end_idx]
                batch = torch.from_numpy(batch).transpose(1,4).transpose(2,4).transpose(3,4).float().to(device)
                with torch.no_grad():
                    output_dict = model(batch)
                    output_images = torch.sigmoid(output_dict['decoder_output'])
                output_images = output_images.transpose(1,2).transpose(2,3).transpose(3,4).cpu().detach().numpy()
                labels = labels*np.ones_like(output_images)
                batch = batch.transpose(1,2).transpose(2,3).transpose(3,4).cpu().detach().numpy()

                # Check if all labels are anomalous
                if np.all(labels == 0):
                    legend = "healthy"
                elif np.all(labels == 1):
                    legend = "anomalous"


                if "visualize" in actions:
                
                    plot_batches_SSL(batch, output_images, labels, channel_to_show = 1,every_x_time_step=1, out_path=os.path.join(results_dir_test, '{}_batch_{}_to_{}.png'.format(legend, start_idx, end_idx)))
                start_idx += batch_size
                end_idx += batch_size
            
    
    logging.info("Done!")
    







            


# %%
