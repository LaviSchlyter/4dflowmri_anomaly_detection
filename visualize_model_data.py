# This file is used to view the data (synthetic or not) that 
#are the inputs to the models.


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
    #["cond_vae/masked_slice/20230704-2004_cond_vae_masked_slice_SSL_lr1.000e-03-e1000-bs8-gf_dim8-daFalse-n_experts3_poisson_mix_training_2Dslice_decreased_interpolation_factor"]
    
    
    #["vae_convT/masked_sliced_full_aorta/20230703-2135_vae_convT_masked_sliced_full_aorta_SSL_lr1.000e-03-e100-bs16-gf_dim8-daFalse_2Dslice"]
    #["vae/masked_slice/20230627-1816_vae_masked_slice_lr1.000e-03-e800-bs8-gf_dim8-daFalse-f100_no_synthetic_validation",
    #                             "vae_convT/masked_slice/20230627-1759_vae_convT_masked_slice_lr1.000e-03-e800-bs8-gf_dim8-daFalse-f100",
    #                             "vae/masked_slice/20230627-1805_vae_masked_slice_lr1.000e-03-e800-bs8-gf_dim8-daFalse-f100"]
    #
    #["vae/masked_slice/20230626-1929_vae_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse_cut_out",
    #                            "vae_convT/masked_slice/20230622-1538_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e800-bs8-gf_dim8-daFalse_interpolation_training",
    #                            "vae_convT/masked_slice/20230622-1535_vae_convT_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse_interpolation_training",
    #                            "vae_convT/masked_slice/20230622-1521_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e800-bs8-gf_dim8-daFalse",
    #                            "vae_convT/masked_slice/20230622-1517_vae_convT_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse",
    #                            "vae/masked_slice/20230622-1542_vae_masked_slice_SSL_lr1.000e-03_scheduler-e800-bs8-gf_dim8-daFalse_interpolation_training",
    #                            "vae/masked_slice/20230622-1545_vae_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse_interpolation_training",
    #                            "vae/masked_slice/20230622-1558_vae_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse",
    #                            "vae/masked_slice/20230626-1929_vae_masked_slice_SSL_lr1.000e-03-e800-bs8-gf_dim8-daFalse_cut_out"]
    """
    "vae_convT/masked_slice/20230620-1427_vae_convT_interpolation_val_masked_slice_SSL_lr5.000e-03_scheduler-e150-bs8-gf_dim8-daFalse",
    "vae_convT/masked_slice/20230619-1947_vae_convT_masked_slice_SSL_lr1.000e-03-e350-bs8-gf_dim8-daFalse",
    "vae_convT/masked_slice/20230620-1420_vae_convT_interpolation_val_masked_slice_SSL_lr5.000e-03-e150-bs8-gf_dim8-daFalse"]
    """
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
            # We don-t have a self supervised model so no masks, but let's take the absolute difference between input and output
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
                    #print(f"Saved figure {fig_name} to {data_exp_dir}")
                    plt.close()
        # %%
