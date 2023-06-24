# This file is used to view the data (synthetic or not) that 
#are the inputs to the models.


# %%
import re
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import logging
# %%
# Helpers
from helpers.utils import make_dir_safely
logging.basicConfig(level=logging.INFO)
logging.info("Importing helpers...")


if __name__ == "__main__":
    
    # Set the experiment directory
    experiments_dir = os.path.join(os.getcwd(), "logs")
    logging.info(f"Experiments directory: {experiments_dir}")


    visualize_data = "training" # "training" or "validation"
    list_of_experiments_paths = ["vae_convT/masked_slice/20230622-1538_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e800-bs8-gf_dim8-daFalse_interpolation_training",
                                 "vae_convT/masked_slice/20230622-1535_vae_convT_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse_interpolation_training",
                                 "vae_convT/masked_slice/20230622-1521_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e800-bs8-gf_dim8-daFalse",
                                 "vae_convT/masked_slice/20230622-1517_vae_convT_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse",
                                 "vae/masked_slice/20230622-1542_vae_masked_slice_SSL_lr1.000e-03_scheduler-e800-bs8-gf_dim8-daFalse_interpolation_training",
                                 "vae/masked_slice/20230622-1545_vae_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse_interpolation_training",
                                 "vae/masked_slice/20230622-1558_vae_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse"]
    """
    "vae_convT/masked_slice/20230620-1427_vae_convT_interpolation_val_masked_slice_SSL_lr5.000e-03_scheduler-e150-bs8-gf_dim8-daFalse",
    "vae_convT/masked_slice/20230619-1947_vae_convT_masked_slice_SSL_lr1.000e-03-e350-bs8-gf_dim8-daFalse",
    "vae_convT/masked_slice/20230620-1420_vae_convT_interpolation_val_masked_slice_SSL_lr5.000e-03-e150-bs8-gf_dim8-daFalse"]
    """
    batch_size = 8
    # Pick n_batches random images
    n_batches = 10
    # Time step
    list_t_ = [0,3,10]
    for t_ in list_t_:
        for exp_path in list_of_experiments_paths:
            
            data_exp_dir = os.path.join(experiments_dir, exp_path) + '/intermediate_results' + '/' + visualize_data
            
            inputs = os.listdir(data_exp_dir)[0]
            outputs = os.listdir(data_exp_dir)[1]
            masks = os.listdir(data_exp_dir)[2]
            n_images = len(os.listdir(os.path.join(data_exp_dir, inputs)))
            random.seed(42)
            random_images = random.sample(range(n_images), n_batches)
            for i in range(n_batches):
                fig, axes = plt.subplots(nrows=3, ncols=batch_size, figsize=(15, 4))
                for j, ax in enumerate(axes.flatten()):
                    if j < batch_size:
                        
                        im = ax.imshow(np.load(os.path.join(data_exp_dir, inputs, os.listdir(os.path.join(data_exp_dir, inputs))[random_images[i]]))[j][0,:,:,t_])
                        # Make colorbar
                        cbar = fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
                    elif j >= batch_size and j < 2*batch_size:
                        
                        im = ax.imshow(np.load(os.path.join(data_exp_dir, outputs, os.listdir(os.path.join(data_exp_dir, outputs))[random_images[i]]))[j - batch_size][0,:,:,t_])
                        # Make colorbar
                        cbar = fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
                    else:
                        im = ax.imshow(np.load(os.path.join(data_exp_dir, masks, os.listdir(os.path.join(data_exp_dir, masks))[random_images[i]]))[j - 2*batch_size][0,:,:,t_])
                        # Make colorbar
                        cbar = fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
                    # Remove ticks
                    ax.set_xticks([])
                    ax.set_yticks([])
                # Give a title to the figure
                fig_name = 'batch' + re.split(r'input_image|.npy',os.listdir(os.path.join(data_exp_dir, inputs))[random_images[i]])[1] + f'_t{t_}'
                # Save figure
                plt.tight_layout()
                plt.savefig(os.path.join(data_exp_dir, fig_name))
                print(f"Saved figure {fig_name} to {data_exp_dir}")
                plt.close()
# %%
