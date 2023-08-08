# This file is used to visualize the data used for training and validation of the model
# %%
import re
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import logging

from config import system_eval as config_sys

# %%
# Helpers
from helpers.utils import make_dir_safely
logging.basicConfig(level=logging.INFO)
logging.info("Importing helpers...")


if __name__ == "__main__":
    
    # Set the experiment directory
    project_code_root = config_sys.project_code_root
    
    experiments_dir = os.path.join(project_code_root, "logs")
    logging.info(f"Experiments directory: {experiments_dir}")
    
    


    visualize_data = "training" # "training" or "validation"
    
    list_of_experiments_paths = ["vae_convT/masked_slice/20230704-2020_vae_convT_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse_with_interpolation_training_2Dslice"]
    
    batch_size = 8
    # Pick n_batches random images
    n_batches = 7
    # Time step
    list_t_ = [0,3,10]
    
    for exp_path in list_of_experiments_paths:
        print(f"------------- {exp_path} ---------------")
        
        
        data_exp_dir = os.path.join(experiments_dir, exp_path) + '/intermediate_results' + '/' + visualize_data
        save_dir = os.path.join(data_exp_dir, "data_viz")
        make_dir_safely(save_dir)
        
        inputs = "inputs"
        outputs ="outputs"
        if os.listdir(data_exp_dir).__contains__("masks"):
            masks = "masks"
        else:
            # We don't have a self supervised model so no masks, but let's take the absolute difference between input and output
            masks = None
            
        n_images = len(os.listdir(os.path.join(data_exp_dir, inputs)))
        random.seed(42)
        random_images = random.sample(range(n_images), n_batches)
        for t_ in list_t_:
                for i in range(n_batches):
                    
                    fig, axes = plt.subplots(nrows=3, ncols=batch_size, figsize=(15, 4))
                    for j, ax in enumerate(axes.flatten()):
                        if j < batch_size:
                            input_image = np.load(os.path.join(data_exp_dir, inputs, os.listdir(os.path.join(data_exp_dir, inputs))[random_images[i]]))[j][0,:,:,t_]
                            im = ax.imshow(input_image)
                            # Make colorbar
                            cbar = fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
                        elif j >= batch_size and j < 2*batch_size:
                            output_image = np.load(os.path.join(data_exp_dir, outputs, os.listdir(os.path.join(data_exp_dir, outputs))[random_images[i]]))[j - batch_size][0,:,:,t_]
                            im = ax.imshow(output_image)
                            # Make colorbar
                            cbar = fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
                        else:
                            if masks is None:
                                input_image = np.load(os.path.join(data_exp_dir, inputs, os.listdir(os.path.join(data_exp_dir, inputs))[random_images[i]]))[j - 2*batch_size][0,:,:,t_]
                                output_image = np.load(os.path.join(data_exp_dir, outputs, os.listdir(os.path.join(data_exp_dir, outputs))[random_images[i]]))[j - 2*batch_size][0,:,:,t_]
                                mask = np.abs(input_image - output_image)
                            else:
                                mask = np.load(os.path.join(data_exp_dir, masks, os.listdir(os.path.join(data_exp_dir, masks))[random_images[i]]))[j - 2*batch_size][0,:,:,t_]

                            im = ax.imshow(mask)
                            # Make colorbar
                            cbar = fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
                        # Remove ticks
                        ax.set_xticks([])
                        ax.set_yticks([])
                    # Give a title to the figure
                    fig_name = 'batch' + re.split(r'input_image|.npy',os.listdir(os.path.join(data_exp_dir, inputs))[random_images[i]])[1] + f'_t{t_}'
                    # Save figure
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(save_dir, fig_name))
                    plt.close()
        # %%
