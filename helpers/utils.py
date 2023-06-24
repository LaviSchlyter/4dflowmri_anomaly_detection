import os 
import numpy as np
from scipy.ndimage import gaussian_filter

from scipy.special import eval_genlaguerre as L 
import torch
import sys
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/')
import config.system as sys_config
from helpers.loss_functions import l2loss, kl_loss_1d, kl_loss_tilted, compute_mmd
# For the patch blending we import from another directory
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/git_repos/many-tasks-make-light-work')
from multitask_method.tasks.patch_blending_task import \
    TestPoissonImageEditingMixedGradBlender, TestPoissonImageEditingSourceGradBlender, TestPatchInterpolationBlender

# TODO Remove (since you may probably ened up with the poissons ones )
from multitask_method.tasks.cutout_task import Cutout
from multitask_method.tasks.patch_blending_task import TestCutPastePatchBlender
from multitask_method.tasks.labelling import FlippedGaussianLabeller


labeller = FlippedGaussianLabeller(0.2)


# ==================================================================
# Verify leakage
# ==================================================================
def verify_leakage():
    basepath =  sys_config.project_data_root
    train_val = basepath + '/final_segmentations/train_val'
    test = basepath + '/final_segmentations/test'
    train_val_files = set(os.listdir(train_val))
    test_files = set(os.listdir(test))
    overlap = train_val_files.intersection(test_files)
    if overlap:
        raise ValueError('There is leakage between train_val and test')

# ==================================================================
# ==================================================================
# Apply Poisson Image Blending on the fly
# ==================================================================
# ==================================================================

def apply_blending(input_images, images_for_blend, mask_blending):
    # Put channels is second dimension
    input_images = np.transpose(input_images, (0,4,1,2,3))
    images_for_blend = np.transpose(images_for_blend, (0,4,1,2,3))
    # Inputs are all numpy arrays TODO: make sure they are (type hinting)
    # Accumlate the blended images and the anomaly masks
    #blended_images = np.empty(input_images.shape)
    blended_images = []
    # The anomaly masks will not have  channel dimension
    #anomaly_masks = np.empty((input_images.shape[0],input_images.shape[2], input_images.shape[3], input_images.shape[4]))
    anomaly_masks = []


    for input_, blender in zip(input_images, images_for_blend):
        # Random flip 
        if np.random.rand() > 0.5:
            # Apply Poisson blending with mixing
            blending_function = TestPoissonImageEditingMixedGradBlender(labeller, blender, mask_blending)
            #blending_function = TestPatchInterpolationBlender(labeller, blender, mask_blending)
            #blending_function = Cutout(labeller)
            blended_image, anomaly_mask = blending_function(input_, mask_blending)
            # output shape of blended image c,x,y,t
            # ouput shape of anomaly mask x,y,t

        else:
            # Apply Poisson blending with source
            blending_function = TestPoissonImageEditingSourceGradBlender(labeller, blender, mask_blending)
            #blending_function = TestPatchInterpolationBlender(labeller, blender, mask_blending)
            #blending_function = Cutout(labeller)
            blended_image, anomaly_mask = blending_function(input_, mask_blending)
            
            
            
            
        # Expand dims to add batch dimension
        blended_image = np.expand_dims(blended_image, axis=0)
        anomaly_mask = np.expand_dims(anomaly_mask, axis=0)
        blended_images.append(blended_image)
        anomaly_masks.append(anomaly_mask)
    blended_images = np.concatenate(blended_images, axis = 0)    
    anomaly_masks = np.concatenate(anomaly_masks, axis = 0)
    # Put channels back in last dimension
    blended_images = np.transpose(blended_images, (0,2,3,4,1))
    #anomaly_masks = np.transpose(anomaly_masks, (0,2,3,1))
    return blended_images, anomaly_masks

# ==================================================================
# ==================================================================
# Compute losses
# ==================================================================
# ==================================================================

def compute_losses_VAE(input_images, output_dict, config):
    """
    Computes the losses for the output images, the latent space and the residual
    """
    # Compute the reconstruction loss
    gen_loss = l2loss(input_images, output_dict['decoder_output'])

    # Compute the residual loss
    true_res = torch.abs(input_images - output_dict['decoder_output'])
    res_loss = l2loss(true_res, output_dict['res'])

    # Compute the latent loss
    lat_loss = kl_loss_1d(output_dict['mu'], output_dict['z_std'])   
    gen_factor_loss = config['gen_loss_factor']*gen_loss

    # Total loss
    loss = torch.mean(gen_factor_loss + lat_loss) 

    # Val loss
    val_loss = torch.mean(gen_loss + lat_loss)

    # Save the losses in a dictionary
    dict_loss = {'loss': loss,'val_loss': val_loss, 'gen_factor_loss': gen_factor_loss,'gen_loss': gen_loss, 'res_loss': res_loss, 'lat_loss': lat_loss}
    return dict_loss


def compute_losses_TVAE(input_images, output_dict, config):
    
    # Compute the reconstruction loss
    gen_loss = l2loss(input_images, output_dict['decoder_output'])

    # Compute the kld loss
    kld_loss = kl_loss_tilted(output_dict['mu'], config['mu_star'])

    # Factor loss
    gen_factor_loss = config['gen_loss_factor']*gen_loss

    # Total loss
    loss = torch.mean(gen_factor_loss + kld_loss)

    # Val loss
    val_loss = torch.mean(gen_loss + kld_loss)

    # Save the losses in a dictionary
    dict_loss = {'loss': loss,'val_loss': val_loss, 'gen_factor_loss': gen_factor_loss,'gen_loss': gen_loss, 'kld_loss': kld_loss}
    return dict_loss

def compute_losses_MMDVAE(input_images, output_dict, config, device, samples_to_generate=100):

    # Compute the reconstruction loss
    gen_loss = l2loss(input_images, output_dict['decoder_output'])

    # Compute the MMD regularization loss
    # Generate random samples following normal distribution
    true_samples = torch.randn(samples_to_generate, 32*config['gf_dim'],2,2,3, device = device)
    mmd_loss = compute_mmd(output_dict['z'], true_samples)

    # Factor loss
    gen_factor_loss = config['gen_loss_factor']*gen_loss

    mmd_factor_loss = config['mmd_loss_factor']*mmd_loss

    # Total loss
    loss = torch.mean(gen_factor_loss + mmd_factor_loss)

    # Val loss
    val_loss = torch.mean(gen_loss + mmd_loss)

    # Save the losses in a dictionary
    dict_loss = {'loss': loss,'val_loss': val_loss, 'gen_factor_loss': gen_factor_loss, 'mmd_factor_loss': mmd_factor_loss, 'gen_loss': gen_loss, 'mmd_loss': mmd_loss}
    return dict_loss
    





# ==================================================================
# ==================================================================


# TVAE
def kld(mu, tau, d):
    # no need to include z, since we run gradient descent...
    return -tau*np.sqrt(np.pi/2)*L(1/2, d/2 -1, -(mu**2)/2) + (mu**2)/2

# convex optimization problem
def kld_min(tau, d):
    steps = [1e-1, 1e-2, 1e-3, 1e-4]
    dx = 5e-3

    # inital guess (very close to optimal value)
    x = np.sqrt(max(tau**2 - d, 0))

    # run gradient descent (kld is convex)
    for step in steps:
        for i in range(1000): # TODO update this to 10000
            y1 = kld(x-dx/2, tau, d)
            y2 = kld(x+dx/2, tau, d)

            grad = (y2-y1)/dx
            x -= grad*step

    return x
# ==========================================        
# function to normalize the input arrays (intensity and velocity) to a range between 0 to 1.
# magnitude normalization is a simple division by the largest value.
# velocity normalization first calculates the largest magnitude velocity vector
# and then scales down all velocity vectors with the magnitude of this vector.
# ==========================================        
def normalize_image(image):

    # ===============    
    # initialize with zeros
    # ===============
    normalized_image = np.zeros((image.shape))
    
    # ===============
    # normalize magnitude channel
    # ===============
    normalized_image[...,0] = image[...,0] / np.amax(image[...,0])
    
    # ===============
    # normalize velocities
    # ===============
    
    # extract the velocities in the 3 directions
    velocity_image = np.array(image[...,1:4])
    
    # denoise the velocity vectors
    velocity_image_denoised = gaussian_filter(velocity_image, 0.5)
    
    # compute per-pixel velocity magnitude    
    velocity_mag_image = np.linalg.norm(velocity_image_denoised, axis=-1)
    
    # velocity_mag_array = np.sqrt(np.square(velocity_arrays[...,0])+np.square(velocity_arrays[...,1])+np.square(velocity_arrays[...,2]))
    # find max value of 95th percentile (to minimize effect of outliers) of magnitude array and its index
    vpercentile = np.percentile(velocity_mag_image, 95)    
    normalized_image[...,1] = velocity_image_denoised[...,0] / vpercentile
    normalized_image[...,2] = velocity_image_denoised[...,1] / vpercentile
    normalized_image[...,3] = velocity_image_denoised[...,2] / vpercentile  
  
    return normalized_image

def normalize_image_new(image):

    # ===============
    # initialize with zeros
    # ===============
    normalized_image = np.zeros((image.shape))

    # ===============
    # normalize magnitude channel
    # ===============
    normalized_image[...,0] = image[...,0] / np.amax(image[...,0])

    # ===============
    # normalize velocities
    # ===============

    # extract the velocities in the 3 directions
    velocity_image = np.array(image[...,1:4])

    # denoise the velocity vectors
    velocity_image_denoised = gaussian_filter(velocity_image, 0.5)

    # compute per-pixel velocity magnitude
    velocity_mag_image = np.linalg.norm(velocity_image_denoised, axis=-1)

    # velocity_mag_array = np.sqrt(np.square(velocity_arrays[...,0])+np.square(velocity_arrays[...,1])+np.square(velocity_arrays[...,2]))
    # find max value of 95th percentile (to minimize effect of outliers) of magnitude array and its index
    # vpercentile_min = np.percentile(velocity_mag_image, 5)
    # vpercentile_max = np.percentile(velocity_mag_image, 95)

    normalized_image[...,1] = 2.*(velocity_image_denoised[...,0] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1
    normalized_image[...,2] = 2.*(velocity_image_denoised[...,1] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1
    normalized_image[...,3] = 2.*(velocity_image_denoised[...,2] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1


    return normalized_image
# ==================================================================    
# ==================================================================    
def make_dir_safely(dirname):
    # directory = os.path.dirname(dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)




def crop_or_pad_Bern_all_slices(data, new_shape):
    #processed_data = np.zeros(new_shape)
    
    # axis 0 is the x-axis and we crop from top since aorta is at the bottom
    # axis 1 is the y-axis and we pad 
    # The axis two we leave since it'll just be the batch dimension
    delta_axis0 = data.shape[0] - new_shape[0]
    delta_axis1 = data.shape[1] - new_shape[1]
    if len(new_shape) == 5: # Image (x,y,None, t,c) - the z will be batch and will vary doesn't need to be equal
        processed_data = np.zeros((new_shape[0], new_shape[1], data.shape[2], new_shape[3], new_shape[4]))
        if delta_axis1 <= 0:
        # The x is always cropped, y padded
            processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,...]
        else:
            # x croped and y cropped equally either way
            processed_data[:, :,:,:data.shape[3],... ] = data[delta_axis0:, (delta_axis1//2):-(delta_axis1//2),...]


    if len(new_shape) == 4: # Label
        processed_data = np.zeros((new_shape[0], new_shape[1], data.shape[2], new_shape[3]))
        # The x is always cropped, y always padded
        processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,...]
    return processed_data

def crop_or_pad_normal_slices(data, new_shape):
    
    processed_data = np.zeros(new_shape)
    # axis 0 is the x-axis and we crop from top since aorta is at the bottom
    # axis 1 is the y-axis and we crop equally from both sides
    # axis 2 is the z-axis and we crop from the right (end of the image) since aorta is at the left
    delta_axis0 = data.shape[0] - new_shape[0]
    
    if len(new_shape) == 5: # Image
        # The x is always cropped, y always padded, z_cropped
        try:
            processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:new_shape[1], :new_shape[2],...]
        except:
            processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:new_shape[1], :new_shape[2],:new_shape[3],...]

    if len(new_shape) == 4: # Label
        # The x is always cropped, y always padded, z_cropped
        try:
            processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:new_shape[1], :new_shape[2],...]
        except:
            processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:new_shape[1], :new_shape[2],:new_shape[3],...]
    return processed_data


def crop_or_pad_Bern_slices(data, new_shape):
    processed_data = np.zeros(new_shape)
    # axis 0 is the x-axis and we crop from top since aorta is at the bottom
    # axis 1 is the y-axis and we pad
    # axis 2 is the z-axis and we crop from the right (end of the image) since aorta is at the left
    delta_axis0 = data.shape[0] - new_shape[0]
    if len(new_shape) == 5: # Image
        # The x is always cropped, y always padded, z_cropped
        try:
            # Pad time
            processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:, :new_shape[2],...]
        except:
            # Crop time
            processed_data[:, :data.shape[1],:,:,... ] = data[delta_axis0:,:, :new_shape[2],:new_shape[3],...]

    if len(new_shape) == 4: # Label
        # The x is always cropped, y always padded, z_cropped
        try:
            
            processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:, :new_shape[2],...]
        except:
            processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:, :new_shape[2],:new_shape[3],...]
    return processed_data

# ==================================================================
# crop or pad functions to change image size without changing resolution
# ==================================================================    
def crop_or_pad_4dvol_along_0(vol, n):    
    x = vol.shape[0]
    x_s = (x - n) // 2
    x_c = (n - x) // 2
    if x > n: # original volume has more slices that the required number of slices
        vol_cropped = vol[x_s:x_s + n, :, :, :, :]
    else: # original volume has equal of fewer slices that the required number of slices
        vol_cropped = np.zeros((n, vol.shape[1], vol.shape[2], vol.shape[3], vol.shape[4]))
        vol_cropped[x_c:x_c + x, :, :, :, :] = vol
    return vol_cropped

# ==================================================================
# crop or pad functions to change image size without changing resolution
# ==================================================================    
def crop_or_pad_4dvol_along_1(vol, n):    
    x = vol.shape[1]
    x_s = (x - n) // 2
    x_c = (n - x) // 2
    if x > n: # original volume has more slices that the required number of slices
        vol_cropped = vol[:, x_s:x_s + n, :, :, :]
    else: # original volume has equal of fewer slices that the required number of slices
        vol_cropped = np.zeros((vol.shape[0], n, vol.shape[2], vol.shape[3], vol.shape[4]))
        vol_cropped[:, x_c:x_c + x, :, :, :] = vol
    return vol_cropped

# ==================================================================
# crop or pad functions to change image size without changing resolution
# ==================================================================    
def crop_or_pad_4dvol_along_2(vol, n):    
    x = vol.shape[2]
    x_s = (x - n) // 2
    x_c = (n - x) // 2
    if x > n: # original volume has more slices that the required number of slices
        vol_cropped = vol[:, :, x_s:x_s + n, :, :]
    else: # original volume has equal of fewer slices that the required number of slices
        vol_cropped = np.zeros((vol.shape[0], vol.shape[1], n, vol.shape[3], vol.shape[4]))
        vol_cropped[:, :, x_c:x_c + x, :, :] = vol
    return vol_cropped

# ==================================================================
# crop or pad functions to change image size without changing resolution
# ==================================================================    
def crop_or_pad_4dvol_along_3(vol, n):    
    x = vol.shape[3]
    x_s = (x - n) // 2
    x_c = (n - x) // 2
    if x > n: # original volume has more slices that the required number of slices
        vol_cropped = vol[:, :, :, x_s:x_s + n, :]
    else: # original volume has equal of fewer slices that the required number of slices
        vol_cropped = np.zeros((vol.shape[0], vol.shape[1], vol.shape[2], n, vol.shape[4]))
        vol_cropped[:, :, :, x_c:x_c + x, :] = vol
    return vol_cropped

# ==================================================================
# crop or pad functions to change image size without changing resolution
# ==================================================================    
def crop_or_pad_4dvol(vol, target_size):
    
    vol = crop_or_pad_4dvol_along_0(vol, target_size[0])
    vol = crop_or_pad_4dvol_along_1(vol, target_size[1])
    vol = crop_or_pad_4dvol_along_2(vol, target_size[2])
    vol = crop_or_pad_4dvol_along_3(vol, target_size[3])
                
    return vol


