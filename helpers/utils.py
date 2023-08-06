import os 
import timeit
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


from multitask_method.tasks.cutout_task import Cutout
from multitask_method.tasks.patch_blending_task import TestCutPastePatchBlender
from multitask_method.tasks.labelling import FlippedGaussianLabeller

# For back transformation
import SimpleITK as sitk
from tvtk.api import tvtk, write_data



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
# TODO: edge case if anomaly would be in several places and then you'd start mixing slices together....
# Chek that by looking at the original slices and make sure they follow each other with a step of 1
def find_subsequence_and_complete(slices, subseq):
    """
        Find a subsequence in an array of slices and complete the subsequence if not found.

        This function applies a modulus operation to the input array `slices` and 
        attempts to find the subsequence `subseq` in the result. If the full subsequence 
        is not found, the function finds the longest continuous sequence and completes 
        it to match the target subsequence.

        Parameters
        ----------
        slices : numpy.ndarray
            An array of integers representing slices. Values should be positive.
        subseq : numpy.ndarray
            The target subsequence to be found in `slices`. 

        Returns
        -------
        numpy.ndarray
            The original slice values that correspond to the found (or completed) subsequence.
            If the subsequence cannot be found or completed, the function returns None.

        Examples
        --------
        >>> slices = np.array([35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45])
        >>> sequence = np.array([0, 1, 2])
        >>> find_subsequence_and_complete(slices, sequence)
        array([36, 37, 38])

    """
    mod_slices = slices % len(subseq)  # modulus operation
    
    subseq_len = len(subseq)

    # iterate over mod_slices to find subsequence
    for i in range(len(mod_slices) - subseq_len + 1):
        if np.all(mod_slices[i:i+subseq_len] == subseq):  # found subsequence
            return slices[i:i+subseq_len]  # return the corresponding original slices

    # if no full sequence is found, look for the longest subsequence
    subsequences = [[0], [0, 1], [1, 2]]
    for subseq in reversed(subsequences):
        for i in range(len(mod_slices) - len(subseq) + 1):
            if np.all(mod_slices[i:i+len(subseq)] == subseq):  # found subsequence
                if subseq[0] == 0:  # if subsequence starts with 0, append next slice
                    sequence = np.concatenate((slices[i:i+len(subseq)], [slices[i+len(subseq)] if i+len(subseq) < len(slices) else slices[-1]+1]))
                    if len(sequence) < subseq_len: # if sequence length is less than subseq_len, append next slice
                        sequence = np.concatenate((sequence, [sequence[-1]+1]))
                    return sequence
                else:  # if subsequence starts with 1 or 2, prepend previous slice
                    sequence = np.concatenate(([slices[i-1] if i > 0 else slices[0]-1], slices[i:i+len(subseq)]))
                    if len(sequence) < subseq_len: # if sequence length is less than subseq_len, append next slice
                        sequence = np.concatenate((sequence, [sequence[-1]+1]))
                    return sequence

    # If we only have one element, we need to create two more slices based on the mod_slices value
    if len(slices) == 1:
        mod_value = mod_slices[0]
        if mod_value == 0:
            return np.array([slices[0], slices[0]+1, slices[0]+2])
        elif mod_value == 1:
            return np.array([slices[0]-1, slices[0], slices[0]+1])
        else: # mod_value == 2
            return np.array([slices[0]-2, slices[0]-1, slices[0]])

    return np.array([0,1,2])  # if no such subsequence or continuous sequence found we give the original slices thus image
def apply_blending(input_images, images_for_blend, mask_blending):
    # If the input_images are 4D + batch (b,x,y,t,c) we put channels in second dimension
    if len(input_images.shape) == 5:
        input_images = np.transpose(input_images, (0,4,1,2,3))
        images_for_blend = np.transpose(images_for_blend, (0,4,1,2,3))
    elif len(input_images.shape) == 6:
        # If the input_images are 5D + batch (b,x,y,z,t,c) we put channels in second dimension
        input_images = np.transpose(input_images, (0,5,1,2,3,4))
        images_for_blend = np.transpose(images_for_blend, (0,5,1,2,3,4))
    # Inputs are all numpy arrays TODO: make sure they are (type hinting)
    # Accumlate the blended images and the anomaly masks
    blended_images = []
    # The anomaly masks will not have  channel dimension
    anomaly_masks = []


    for input_, blender in zip(input_images, images_for_blend):
        
        # Random flip 
        if np.random.rand() > 0.5:
            # Apply Poisson blending with mixing
            blending_function = TestPoissonImageEditingMixedGradBlender(labeller, blender, mask_blending)
            #blending_function = TestPatchInterpolationBlender(labeller, blender, mask_blending)
            #blending_function = Cutout(labeller)
            #blending_function = TestPoissonImageEditingSourceGradBlender(labeller, blender, mask_blending)
            blended_image, anomaly_mask = blending_function(input_, mask_blending)
            # output shape of blended image c,x,y,t
            # ouput shape of anomaly mask x,y,t

        else:
            # Apply Poisson blending with source
            
            blending_function = TestPoissonImageEditingSourceGradBlender(labeller, blender, mask_blending)
            #blending_function = TestPatchInterpolationBlender(labeller, blender, mask_blending)
            #blending_function = Cutout(labeller)
            #blending_function = TestPoissonImageEditingMixedGradBlender(labeller, blender, mask_blending)
            blended_image, anomaly_mask = blending_function(input_, mask_blending)
            
        #
        # If we have the extended slices for the conditional network then we need to remove the repeated sequences
        
        if len(input_images.shape) == 6:
            z_slices_with_anomalies = np.unique(np.where(anomaly_mask != 0)[0])
            sequence = np.array([0, 1, 2]) # We have three slices
            z_slices_to_use = find_subsequence_and_complete(z_slices_with_anomalies, sequence)

            blended_image = blended_image[:, z_slices_to_use, ...] # Channel in first dimension
            anomaly_mask = anomaly_mask[z_slices_to_use, ...] # No channel dimension
        
        
            
            
        # Expand dims to add batch dimension
        blended_image = np.expand_dims(blended_image, axis=0)
        anomaly_mask = np.expand_dims(anomaly_mask, axis=0)
        blended_images.append(blended_image)
        anomaly_masks.append(anomaly_mask)
    blended_images = np.concatenate(blended_images, axis = 0)    
    anomaly_masks = np.concatenate(anomaly_masks, axis = 0)
    # Put channels back in last dimension for the blended images
    if len(input_images.shape) == 5:
        blended_images = np.transpose(blended_images, (0,2,3,4,1))
    elif len(input_images.shape) == 6:
        blended_images = np.transpose(blended_images, (0,2,3,4,5,1))
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
    #true_res = torch.abs(input_images - output_dict['decoder_output'])
    #res_loss = l2loss(true_res, output_dict['res'])
    res_loss = torch.zeros_like(gen_loss)

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





# ==================================================================
# ==================================================================
# Utils for backtrasnforming the anomaly scores
# ==================================================================
# ==================================================================
# We need to pad it back to original 36,36,64 - it was down to 32 for network reasons 

def expand_normal_slices(data, original_shape):
    
    # Create an array of zeros with the original shape
    expanded_data = np.zeros(original_shape)

    # Compute the difference in the first two dimensions
    delta_axis0 = original_shape[0] - data.shape[0]
    delta_axis1 = original_shape[1] - data.shape[1]
    
    # Place the cropped data back into the array
    expanded_data[delta_axis0:,:data.shape[1], :data.shape[2],:data.shape[3],... ] = data

    return expanded_data

def resample_back(anomaly_score, sitk_original_image, geometry_dict):

    # We sum the anomaly score over the channel dimension
    # For ssl we only have one channel
    # For reconstruction based we have 4
    anomaly_score = np.sum(anomaly_score, axis=-1)

    temp_for_time_stacking = []
    # Loop over the time dimension
    for time_slice in np.arange(anomaly_score.shape[3]):
        temp_for_slice_stacking = []
        # Loop over the slice dimension
        for slice_i in np.arange(anomaly_score.shape[2]):
            
            resampled_sitk_image = sitk.GetImageFromArray(anomaly_score[:,:,slice_i:slice_i+1,time_slice].transpose([2,1,0]))
            # You need to set the orgin, direction and spacing 
            resampled_sitk_image.SetOrigin(geometry_dict[f"slice_{slice_i}"]["origin"])
            # The two following remain the same, they are not saved in the geometry dict
            resampled_sitk_image.SetDirection((1,0,0,0,1,0,0,0,1))
            resampled_sitk_image.SetSpacing((1,1,1))
            # We also need to get the transform and inverse it
            inverse_transform = geometry_dict[f"slice_{slice_i}"]["transform"].GetInverse()
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(sitk_original_image.GetSize()) # set to original image size
            resampler.SetOutputSpacing(sitk_original_image.GetSpacing()) # set to original image spacing
            resampler.SetOutputDirection(sitk_original_image.GetDirection()) # set to original image direction
            resampler.SetOutputOrigin(sitk_original_image.GetOrigin()) # set to original image origin
            resampler.SetInterpolator(sitk.sitkLinear) 
            resampler.SetTransform(inverse_transform) 

            resampled_slice = resampler.Execute(resampled_sitk_image) # apply to the resampled slice
            resampled_slice = sitk.GetArrayFromImage(resampled_slice)
            temp_for_slice_stacking.append(resampled_slice)

        # We need to sum over the 64 original sized images
        temp_for_time_stacking.append(np.sum(np.array(temp_for_slice_stacking), axis=0))

    # Stack the time dimension
    anomaly_score_4d_image = np.stack(temp_for_time_stacking, axis=-1)
    # (x,y,z,t)

    # Might be good idea to save this. 
    return anomaly_score_4d_image

# ==================================================================
# To visualize the backtransformed anomaly scores we need to convert them to vtk
# ==================================================================

def convert_to_vtk(backtransformed_anomaly_score, subject_id, save_dir):
    # Note that you will need to threshold just above zero to get the
    # anomaly score to show up in paraview since we fill up the background
    time_slices = backtransformed_anomaly_score.shape[3]
    
    for time_slice in np.arange(time_slices):

        # Anomaly score for this time slice
        anomaly_score = backtransformed_anomaly_score[..., time_slice]

        # Set the grid
        # X,Y,Z
        dim = anomaly_score.shape
        # Generate the grid
        xx,yy,zz = np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]]
        pts_anomaly_score = np.empty(anomaly_score.shape + (3,), dtype=int)
        pts_anomaly_score[..., 0] = xx
        pts_anomaly_score[..., 1] = yy
        pts_anomaly_score[..., 2] = zz

        pts_anomaly_score = pts_anomaly_score.transpose(2, 1, 0, 3).copy()
        pts_anomaly_score.shape = pts_anomaly_score.size // 3, 3
        anomaly_score_vec = np.empty(anomaly_score.shape + (1,), dtype=float)
        anomaly_score_vec[..., 0] = anomaly_score

        anomaly_score_vec = anomaly_score_vec.transpose(2, 1, 0, 3).copy()
        
        anomaly_score_vec.shape = anomaly_score_vec.size
        
        
        anomaly_score_grid = tvtk.StructuredGrid(dimensions=xx.shape, points=pts_anomaly_score)
        anomaly_score_grid.point_data.scalars = anomaly_score_vec
        anomaly_score_grid.point_data.scalars.name = 'anomaly_score_intensity'
        write_data(anomaly_score_grid, os.path.join(save_dir,f"{subject_id}_anomaly_score_t{time_slice}.vtk"))
    
    
# ==================================================================
# Combine the functions to backtransform the anomaly scores
# ==================================================================
def backtransform_anomaly_scores(anomaly_score, subject_id, output_dir, geometry_dict):
    # Expand the anomaly score to the original size
    anomaly_score = expand_normal_slices(anomaly_score,[36,36,64,24,4])
    # Backtransform the anomaly score
    backtransformed_anomaly_score = resample_back(anomaly_score, geometry_dict)
    # Save the backtransformed anomaly score
    #np.save(os.path.join(output_dir, subject_id, f"{subject_id}_backtransformed_anomaly_score.npy"), backtransformed_anomaly_score)
    # Convert to vtk
    convert_to_vtk(backtransformed_anomaly_score, subject_id, output_dir)