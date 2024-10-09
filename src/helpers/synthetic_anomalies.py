"""
Synthetic Anomaly Generation Script

This script generates synthetic anomalies in medical imaging data for anomaly detection 
experiments. It applies various types of deformations to the original images and saves the 
resulting synthetic data to an HDF5 file.

The main functions and their purposes are:

- generate_noise: Generates random noise to be added to images.
- calc_distance: Calculates the Euclidean distance between two points.
- create_mask: Creates a mask for a given image based on a center and width.
- create_deformation_chan: Applies a deformation to an image based on a given center and width.
- generate_deformation_chan: Generates deformed versions of input images and corresponding masks.
- create_hollow_noise: Creates hollow noise patterns in the images.
- create_cube_mask: Creates a cube mask for blending operations.
- select_random_image_for_blending: Selects a random image for blending.
- prepare_and_write_synthetic_data: Prepares and writes synthetic data with various deformations to an HDF5 file.
- load_create_syntetic_data: Loads or creates synthetic data based on specified configurations.

This script supports the following types of deformations:
- None: No deformation, original image.
- noisy: Adds random noise to the image.
- deformation: Applies a geometric deformation to the image.
- hollow circle: Adds a hollow circle of noise to the image.
- patch_interpolation: Blends patches from different images using interpolation.
- poisson_with_mixing: Applies Poisson blending with mixed gradients.
- poisson_without_mixing: Applies Poisson blending without mixed gradients.

The script is intended for use in an anomaly detection pipeline, where synthetic anomalies 
are introduced to create a more robust and challenging dataset for training and evaluation.
"""
# %%
# Import packages
import os
import sys
import h5py
import numpy as np
import random
import logging
import argparse
import yaml
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')




sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4DFlowAorticAnomalies')
from config import system as sys_config
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4DFlowAorticAnomalies/src')
from helpers.utils import make_dir_safely, create_suffix
from helpers.data_loader import load_data


from scipy.interpolate import RegularGridInterpolator
import copy

from scipy.ndimage import binary_fill_holes
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import cdist

# For the patch blending we import from another directory
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/git_repos/many-tasks-make-light-work')
from multitask_method.tasks.patch_blending_task import TestPatchInterpolationBlender, \
    TestPoissonImageEditingMixedGradBlender, TestPoissonImageEditingSourceGradBlender

from multitask_method.tasks.labelling import FlippedGaussianLabeller


labeller = FlippedGaussianLabeller(0.2)


# ==========================================
# ==========================================
# Random noise square patch
# ==========================================
# ==========================================
def generate_noise(batch_size, mean_range = (0, 0.1), std_range = (0, 0.2), n_samples = 30):
    """
    Generates noise to be added to the images
    :param batch_size: number of images in the batch
    :param mean_range: range of means for the normal distribution
    :param std_range: range of standard deviations for the normal distribution
    :return: mean, std, noise
    """
    means = np.random.uniform(mean_range[0], mean_range[1], n_samples)
    stds = np.random.uniform(std_range[0], std_range[1], n_samples)
    #noise_parameters = [(0.0, 0.02), (0.05, 0.05),(0.1, 0.1),(0.1, 0.2),(0.1, 0.3), (0.2, 0.1), (0.2, 0.2) ,(0.2, 0.4), (0.3, 0.1), (0.3, 0.2), (0.3, 0.3)] # additive noise to be tested, each tuple has mean and stdev for the normal distribution

    for (mean, std) in zip(means, stds):

        # Add some random noise to the image
        part_noise = np.random.normal(mean, std, (batch_size, 5, 5, 24, 4))
        full_noise = np.zeros((batch_size, 32, 32, 24, 4))
        full_noise[:, 14:19, 14:19,:, :] = part_noise

        yield mean, std, full_noise

def calc_distance(xyz0 = [], xyz1 = []):
    delta_OX = (xyz0[0] - xyz1[0])**2
    delta_OY = (xyz0[1] - xyz1[1])**2
    delta_OZ = (xyz0[2] - xyz1[2])**2
    return (delta_OX+delta_OY+delta_OZ)**0.5 

def create_mask(im,center,width):
    dims = np.shape(im)
    mask = np.zeros_like(im)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                dist_i = calc_distance([i,j,k],center)
                if dist_i<width:
                    mask[i,j,k]=1
    return mask


# ==========================================
# ==========================================
# Deformation
# ==========================================
# ==========================================
def create_deformation_chan(im,center,width,polarity=1):
    # Create the deformation and apply it to all channels
    dims = np.array(np.shape(im))
    mask = np.zeros_like(im)
    
    center = np.array(center)
    wv,xv,yv,zv = np.arange(dims[0]),np.arange(dims[1]),np.arange(dims[2]),np.arange(dims[3])
    interp_samp = RegularGridInterpolator((wv, xv, yv, zv), im)
    
    for i in range(dims[1]):
        for j in range(dims[2]):
            for k in range(dims[3]):
                dist_i = calc_distance([i,j,k],center)
                displacement_i = (dist_i/width)**2
                
                if displacement_i < 1.:
                    #within width
                    if polarity > 0:
                        #push outward
                        diff_i = np.array([i,j,k])-center
                        new_coor =  center + diff_i*displacement_i
                        
                    else:
                        #pull inward
                        cur_coor = np.array([i,j,k])
                        diff_i = cur_coor-center
                        new_coor = cur_coor + diff_i*(1-displacement_i)
                        
                    new_coor = np.clip(new_coor,np.zeros(len(dims[1:])),dims[1:]-1)
                    new_coor_chan = np.tile(np.arange(dims[0]),(len(dims),1)).T
                    new_coor_chan[:,1:] = new_coor
                    mask[:,i,j,k]= interp_samp(new_coor_chan)
                else:
                    mask[:,i,j,k] = im[:,i,j,k]
    return mask

def generate_deformation_chan(im):
    # Copy to avoid changing the original image
    im2 = copy.deepcopy(im)
    masks = np.zeros_like(im)

    for batch_i in range(im.shape[0]):
        im_in = im2[batch_i,:,:,:,:]
        dims = np.array(np.shape(im_in))

        # For the third dimesnion (time) we want the anomalies to be in the start
        core = dims[:3]/[4,4,4] #width of core region
        offset = (dims[:3]-core)/[2,2,18]#offset to center core
        
        min_width = np.round(0.05*dims[0])
        max_width = np.round(0.2*dims[0])

        sphere_center = []
        sphere_width = []

        for i,_ in enumerate(dims[:3]):
            sphere_center.append(np.random.randint(offset[i],offset[i]+core[i]))


        sphere_width = np.random.randint(min_width,max_width)
        mask_i = create_mask(im_in,sphere_center,sphere_width)
        
        
        sphere_polarity = 1
        if np.random.randint(2):#random sign
            sphere_polarity *= -1

        # Put channels first for the deformation function
        im_in = np.transpose(im_in, (3, 0, 1, 2))
        out_image = create_deformation_chan(im_in,sphere_center,sphere_width,sphere_polarity)
        #[c, w, h, t]
        # Put channels last for output
        out_image = np.transpose(out_image, (1, 2, 3, 0))
        im2[batch_i,:,:,:,:] = out_image
        masks[batch_i,:,:,:,:] = mask_i

    return im2, masks

# ==========================================
# ==========================================
# Hollow mask
# ==========================================
# ==========================================





def find_closest_distance_edge(mask, center_of_mass):
    # Find the closest distance from the center of mass to the edge of the mask (first zero)
    # Find the indices of zero values in the mask
    zero_indices = np.argwhere(mask == 0)

    # Calculate the Euclidean distances between the zero indices and the center of mass
    distances = cdist(zero_indices, [center_of_mass])

    # Find the minimum distance
    closest_distance = np.min(distances)

    return closest_distance

# Fill in the circle with random noise
def fill_circle(image, center, radius, mean, std):
    # The mean and std are the same as randomly generated for noisy image

    # Generate coordinate grids
    x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))

    # Calculate distance from center for each pixel
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # Create a mask based on which pixels are within the radius
    mask = distances <= radius
    
    # Fill in the circle with random noise
    noise = np.random.normal(mean, std, size=image.shape)
    return mask, noise

def create_hollow_noise(im, mean, std, ratio = 0.8):
    # We only create the mask in the x,y plane, we then propagate on time (later)
    noisy_masks = np.zeros_like(im)
    # Copy to avoid changing the original image
    image_copy = copy.deepcopy(im)
    for batch in range(im.shape[0]):
        # Take x,y dimensions
        #image = image[:,:, 0,0]
        image = image_copy[batch,:,:,0,0]

        # Binarize the image
        image[image != 0] = 1

        # Fill holes in the image
        image = binary_fill_holes(image).astype(int)

        # Find center of mass
        center = center_of_mass(image)

        # They return in (y,x) format
        center = [center[1], center[0]]

        # Round to integer
        center = np.round(center).astype(int)

        # Find the closest distance from the center of mass to the edge of the mask (first zero)
        radius = find_closest_distance_edge(image, center)

        # Generate random noise
        mask, noise = fill_circle(image, center, radius, mean, std)

        # Create a smaller circle with same noise
        radius_2 = radius * ratio

        mask_2, _ = fill_circle(image, center, radius_2, mean, std)

        # Subtract the two masks to create the hollow mask with noise
        final_noise = noise*mask - noise*mask_2

        # We need to extend the noise to the time dimension and channel dimensions
        # Note that we only apply the hollow noise on timsteps 0-12
        # Reshape the image to (x,y, 1, 1)
        final_noise_resaped = final_noise.reshape(final_noise.shape[0], final_noise.shape[1], 1, 1)

        # Create an array of zeros of size (x,y, 24, 4)
        noisy_mask = np.zeros((final_noise.shape[0], final_noise.shape[1], 24, 4), dtype=final_noise.dtype)    

        # Assign the noisy_mask image to the first 12 dimensions of 24
        noisy_mask[:, :, :12, :] = final_noise_resaped

        # Divide the magnitude noise by 10
        noisy_mask = (noisy_mask[:,:,:,:]/[10,1,1,1])

        # Add batch dimension
        noisy_mask = noisy_mask[np.newaxis, :, :, :, :]

        # Add to the array
        noisy_masks[batch,:,:,:,:] = noisy_mask

    return noisy_masks

# ==========================================
# ==========================================
# Patch blending
# 1. Interpolation
# 2. Mixed poisson blending (Color values are also blended)
# 3. Non-mixed poisson blending (Focus is more on the preservation of structure and content)
# These are taken from https://github.com/matt-baugh/many-tasks-make-light-work/tree/main
# Corrected the return value from interpolation function
# ==========================================
# ==========================================

def create_cube_mask(mask_size, WH, depth =7, inside = True):
    """
    This function creates a cube(rectangle) mask of size mask_size and width and height WH
    where the time dimesion is depth and starts at 0 rather than in the center like the other dimensions
    This is because we want the anomaly to be located in the first moments as it is when the blood pumps
    """
    # WH is the width height, we keep them the same
    if inside:
        # Create an mas of zeros
        mask = np.zeros(mask_size)
    else:
        # Create an mask of ones
        mask = np.ones(mask_size)
    
    # Set the cube region to zeros
    # Set the rectangle region to zeros
    start = (mask_size[0] // 2 - WH // 2, mask_size[1] // 2 - WH // 2, 0)
    end = (start[0] + WH, start[1] + WH, depth)
    
    
    if inside:
        mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = 1
    else:
        mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = 0
    
    return mask

def create_cube_mask_4D(mask_size, WH, depth=7, inside=True):
    """
    This function creates a 4D cube (rectangle) mask of size mask_size (a list of 4 elements)
    and width and height WH where the time dimension is depth and starts at 0 rather than in the center
    like the other dimensions. This is because we want the anomaly to be located in the first moments
    as it is when the blood pumps.
    """
    if inside:
        # Create a 3D mask of zeros
        mask = np.zeros(mask_size)
    else:
        # Create a 3D mask of ones
        mask = np.ones(mask_size)

    
    # Set the rectangle region to zeros
    start = (
        0,
        mask_size[1] // 2 - WH // 2,
        mask_size[2] // 2 - WH // 2,
        0# For the fourth dimension
    )
    end = (
        mask_size[0],
        start[0] + WH,
        start[1] + WH,
        depth # For the fourth dimension
    )

    if inside:
        mask[start[0]:end[0], start[1]:end[1], start[2]:end[2], start[3]:end[3]] = 1
    else:
        mask[start[0]:end[0], start[1]:end[1], start[2]:end[2], start[3]:end[3]] = 0

    return mask

def select_random_image_for_blending(loop_index, data, n_patients, z_slices = 64):
        # We pick a random image to blend but same location 
        z_slice = loop_index%z_slices
        patient = loop_index//z_slices
        # We pick a random patient different from the current one and take the same slice number
        random_patient = random.randint(0, n_patients - 1)
        while random_patient == patient:
            random_patient = random.randint(0, n_patients - 1)
        image_for_blend = data[random_patient * z_slice, ...]
        image_for_blend = image_for_blend.transpose(3, 0, 1, 2)
        return image_for_blend


def process_poisson_blending(image, j, data, n_patients, z_slices, mask_blending, labeller, dataset, i, data_shape, blending_task_class):
    """
    Helper function to process Poisson image blending.
    
    :param image: The input image.
    :param j: Current index of the image.
    :param data: Full dataset.
    :param n_patients: Number of patients.
    :param z_slices: Number of slices per patient.
    :param mask_blending: Mask used for blending.
    :param labeller: The labeller to be used.
    :param dataset: The dataset where results will be saved.
    :param i: Current index of the deformation.
    :param data_shape: Shape of the data.
    :param blending_task_class: Class of the blending task (e.g. mixed or non-mixed).
    """
    # Process image and apply Poisson blending
    image = np.squeeze(image, axis=0)
    image = image.transpose(3, 0, 1, 2)
    image_for_blend = select_random_image_for_blending(j, data, n_patients, z_slices=z_slices)
    
    # Instantiate the blender class
    poisson_task = blending_task_class(labeller, image_for_blend, mask_blending)
    
    # Create the blended image and anomaly mask
    blended_image, anomaly_mask = poisson_task(image, mask_blending)
    
    # Binarize the mask
    anomaly_mask[anomaly_mask != 0] = 1
    
    # Add a channel dimension to mask, repeating it for each channel
    anomaly_mask = np.repeat(anomaly_mask[..., np.newaxis], image.shape[0], axis=-1)
    
    # Reorder dimensions for saving
    blended_image = blended_image.transpose(1, 2, 3, 0)
    
    # Save the image and mask to the dataset
    dataset['images'][i * data_shape[0] + j, :, :, :, :] = blended_image
    dataset['masks'][i * data_shape[0] + j, :, :, :, :] = anomaly_mask

def prepare_and_write_synthetic_data(data,
                                    deformation_list,
                                    filepath_output,
                                    z_slices = 64):
    
    data_shape = data.shape # [Number of patients * z_slices, x, y, t, num_channels]
    n_patients = data_shape[0] // z_slices
    logging.info(f"Number of patients to process: {n_patients}")
    
    # ==========================================
    # we will stack all images along their z-axis
    # --> the network will analyze (x,y,t) volumes
    # ==========================================

    # For each patient, for each slice we create a new image with the anomaly
    # The dataset also contains the healthy images (None as deformation type)
    images_dataset_shape = [data_shape[0] * len(deformation_list),
                            data_shape[1],
                            data_shape[2],
                            data_shape[3],
                            data_shape[4]]
    
    dataset = {}
    hdf5_file = h5py.File(filepath_output, 'w')

    # Create the dataset
    dataset['images'] = hdf5_file.create_dataset("images", images_dataset_shape, dtype='float32')

    # Create the mask 
    dataset['masks'] = hdf5_file.create_dataset("masks", images_dataset_shape, dtype='float32')

    # Noise generator
    noise_generator = generate_noise(batch_size = 1, n_samples= data_shape[0] * len(deformation_list))

    # For the patch interpolation and poisson blending we need to create a mask
    # In our case the mask is the boundaries of the blending for the source and final image
    # Since we have the aorta in the middle of the image, we will create a mask that is a cube
    # Centered in the first two dimesions, on the third we make it closer to the beginning
    mask_shape = [data_shape[1], data_shape[2], data_shape[3]]
    mask_blending = create_cube_mask(mask_shape, WH= 15, depth= 12,  inside=True).astype(np.bool8)
    
    
    # For each type of deformation, we create a new image for each patient and each slice
    for i, deformation_type in enumerate(deformation_list):
        logging.info(f"Creating images for deformation type: {deformation_type}")
        
        if deformation_type == 'None':
            # We do not create a new image, we just copy the original image
            dataset['images'][i*data_shape[0]:(i+1)*data_shape[0],:,:,:,:] = data
            dataset['masks'][i*data_shape[0]:(i+1)*data_shape[0],:,:,:,:] = np.zeros_like(data)
        else:
            for j, image in enumerate(data):
                # Logging the progress out of total
                if j % 64 == 0:
                    logging.info(f"Processing image {j} of {data_shape[0]}")
                    


                # Add batch dimension to 1
                image = np.expand_dims(image, axis=0)
                if deformation_type == 'noisy':
                    # We add noise to the image
                    # We scale the noise of the channel down by a factor of 10 (not as high intensity as the other velocity channels)
                    mean, std, noise = next(noise_generator)
                    noise = noise/[10,1,1,1]
                    dataset['images'][i*data_shape[0] + j,:,:,:,:] = image + noise
                    # We make a mask of the noise so we binarize it
                    noise[noise != 0] = 1
                    
                    dataset['masks'][i*data_shape[0] + j,:,:,:,:] = noise
                    
                elif deformation_type == 'deformation':
                    # We create a deformation in the image
                    deformation, mask_deformation = generate_deformation_chan(image)
                    dataset['images'][i*data_shape[0] + j,:,:,:,:] = deformation

                    # Binarize the mask
                    mask_deformation[mask_deformation != 0] = 1
                    dataset['masks'][i*data_shape[0] + j,:,:,:,:] = mask_deformation
                    
                elif deformation_type == 'hollow circle':
                    noisy_mask = create_hollow_noise(image, mean=mean, std=std)
                    dataset['images'][i*data_shape[0] + j,:,:,:,:] = image + noisy_mask
                    # We make a mask of the noise so we binarize it
                    noisy_mask[noisy_mask != 0] = 1
                    dataset['masks'][i*data_shape[0] + j,:,:,:,:] = noisy_mask

                elif deformation_type == 'patch_interpolation':
                    # Here we don't except a batch dimension but channel should be first
                    image = np.squeeze(image, axis=0)
                    image = image.transpose(3, 0, 1, 2)
                    image_for_blend = select_random_image_for_blending(j, data, n_patients, z_slices = z_slices)
                    # Instantiate the patch interpolation class
                    patch_interp_task = TestPatchInterpolationBlender(labeller, image_for_blend, mask_blending)

                    # Create the blended image
                    blended_image, anomaly_mask = patch_interp_task(image, mask_blending)

                    # Binarize the mask
                    anomaly_mask[anomaly_mask != 0] = 1
                    # Add a channel dimension to mask repeating it for each channel
                    anomaly_mask = np.repeat(anomaly_mask[..., np.newaxis], image.shape[0], axis=-1)
                    
                    blended_image = blended_image.transpose(1, 2, 3, 0)
                    # Save image and mask
                    dataset['images'][i*data_shape[0] + j,:,:,:,:] = blended_image
                    dataset['masks'][i*data_shape[0] + j,:,:,:,:] = anomaly_mask


                elif deformation_type == 'poisson_with_mixing':
                    process_poisson_blending(image, j, data, n_patients, z_slices, mask_blending, labeller, dataset, i, data_shape, TestPoissonImageEditingMixedGradBlender)

                    
                elif deformation_type == 'poisson_without_mixing':
                    process_poisson_blending(image, j, data, n_patients, z_slices, mask_blending, labeller, dataset, i, data_shape, TestPoissonImageEditingSourceGradBlender)

                    
                else:
                    raise ValueError('deformation_type must be either None, noisy, deformation, hollow circle or patch and not {}'.format(deformation_type))

    hdf5_file.close()

    return 0            



def load_create_syntetic_data(data,
                        deformation_list,
                        preprocessing_method,
                        idx_start,
                        idx_end,
                        force_overwrite=False,
                        note = '',
                        ):
    # if preprocessing method contains full_aorta, then z_slices = 256 else 64
    if 'full_aorta' in preprocessing_method:
        z_slices = 256
    else:
        z_slices = 64
    savepath= sys_config.project_code_root + "data"
    make_dir_safely(savepath)
    if note != '':
        dataset_filepath = savepath + f'/{preprocessing_method}_anomalies_images_from_' + str(idx_start) + '_to_' + str(idx_end) + '_' + note + '.hdf5'

    else:
        dataset_filepath = savepath + f'/{preprocessing_method}_anomalies_images_from_' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'
    
    if not os.path.exists(dataset_filepath) or force_overwrite:
        logging.info(f"This configuration has not yet been preprocessed. Preprocessing now...")
        prepare_and_write_synthetic_data(
                                data = data,
                                deformation_list = deformation_list,
                                filepath_output = dataset_filepath,
                                z_slices= z_slices
                                )
        
        
        logging.info(f"Preprocessing done. Data saved to {dataset_filepath}")
    else:
        # Already preprocessed load file
        logging.info(f"Data already preprocessed. Loading file from {dataset_filepath}")
        
    
    return h5py.File(dataset_filepath, 'r')



if __name__ == '__main__':

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run inference with specific settings.")
    parser.add_argument('--config_path', type=str, help='Path to the configuration YAML file')
    # Parse arguments
    args = parser.parse_args()

    # Load the configuration
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Possible types of deformations
    deformation_list = config['deformation_list']
    suffix = create_suffix(config)

    logging.info(f"Creating synthetic anomalies for the following deformations: {deformation_list}")
    logging.info(f"Suffix: {suffix}")

    
    data_dict = load_data(config=config, sys_config=sys_config, idx_start_tr = 0, idx_end_tr = 1, idx_start_vl = config['idx_start_val'], idx_end_vl = config['idx_end_val'], idx_start_ts = 0, idx_end_ts = 1, suffix = suffix, include_compressed_sensing = config['include_compressed_sensing_data'], only_compressed_sensing = config['use_only_compressed_sensing_data'])
    images_vl = data_dict['images_vl']
    

    # Create synthetic anomalies
    data = load_create_syntetic_data(data = images_vl,
                        deformation_list = deformation_list,
                        preprocessing_method = config['preprocess_method'],
                        idx_start = config['idx_start_val'],
                        idx_end = config['idx_end_val'],
                        force_overwrite=True,
                        note = f"{suffix}{config['extra_synthetic_note']}"
                        )
    
