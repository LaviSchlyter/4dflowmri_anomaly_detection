import os 
import SimpleITK as sitk
from scipy import interpolate
from tvtk.api import tvtk, write_data
import numpy as np
from skimage.morphology import skeletonize_3d, dilation, binary_erosion
from scipy.ndimage import gaussian_filter
import torch
import sys
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection')
import config.system as sys_config
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/src')
from helpers.loss_functions import l2loss, kl_loss_1d
# For the patch blending we import from another directory
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/git_repos/many-tasks-make-light-work')
from multitask_method.tasks.patch_blending_task import \
    TestPoissonImageEditingMixedGradBlender, TestPoissonImageEditingSourceGradBlender, TestPatchInterpolationBlender
from multitask_method.tasks.cutout_task import Cutout
from multitask_method.tasks.patch_blending_task import TestCutPastePatchBlender
from multitask_method.tasks.labelling import FlippedGaussianLabeller

labeller = FlippedGaussianLabeller(0.2)


# ==================================================================
# Key basic functions
# ==================================================================

def verify_leakage():
    basepath =  sys_config.project_data_root
    train_val_balanced = basepath + '/segmentations/final_segmentations/train_val_balanced'
    test_balanced = basepath + '/segmentations/final_segmentations/test_balanced'

    train_val_balanced_files = set(os.listdir(train_val_balanced))
    test_balanced_files = set(os.listdir(test_balanced))

    overlap_balanced = train_val_balanced_files.intersection(test_balanced_files)

    if overlap_balanced:
        raise ValueError('There is leakage between train_val_balanced and test_balanced')
    print('No leakage between train_val and test')

def normalize_image(image, with_percentile = None, with_set_range = None):

    # initialize with zeros
    normalized_image = np.zeros((image.shape))

    # normalize magnitude channel
    normalized_image[...,0] = image[...,0] / np.amax(image[...,0])

    # normalize velocities

    # extract the velocities in the 3 directions
    velocity_image = np.array(image[...,1:4])

    # denoise the velocity vectors on the spatial and time dimensions but not across channels
    velocity_image_denoised = gaussian_filter(velocity_image, sigma=(0.5,0.5,0.5,0.5,0))

    if with_percentile is not None and len(with_percentile) == 2:
        
        vpercentile_min = np.percentile(velocity_image_denoised, with_percentile[0])
        vpercentile_max = np.percentile(velocity_image_denoised, with_percentile[1])

        # Normalize the velocity vectors
        normalized_image[...,1] = 2.*(velocity_image_denoised[...,0] - vpercentile_min)/ (vpercentile_max - vpercentile_min)-1
        normalized_image[...,2] = 2.*(velocity_image_denoised[...,1] - vpercentile_min)/ (vpercentile_max - vpercentile_min)-1
        normalized_image[...,3] = 2.*(velocity_image_denoised[...,2] - vpercentile_min)/ (vpercentile_max - vpercentile_min)-1

        # Clip the values to be in the range [-1,1]
        normalized_image = np.clip(normalized_image, -1, 1)
    
    elif with_set_range is not None and len(with_set_range) == 2:

            # Calculate the scaling factor
            scaling_factor = 2 / (with_set_range[1] - with_set_range[0])

            # Normalize the velocity vectors
            normalized_image[..., 1] = (velocity_image_denoised[..., 0] - with_set_range[0]) * scaling_factor - 1
            normalized_image[..., 2] = (velocity_image_denoised[..., 1] - with_set_range[0]) * scaling_factor - 1
            normalized_image[..., 3] = (velocity_image_denoised[..., 2] - with_set_range[0]) * scaling_factor - 1

    else:    

        normalized_image[...,1] = 2.*(velocity_image_denoised[...,0] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1
        normalized_image[...,2] = 2.*(velocity_image_denoised[...,1] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1
        normalized_image[...,3] = 2.*(velocity_image_denoised[...,2] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1

    return normalized_image
  
def make_dir_safely(dirname):
    # directory = os.path.dirname(dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

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

def create_suffix(config):
    parts = []

    if config['with_rotation']:
        parts.append('_with_rotation')
    else:
        parts.append('_without_rotation')

    if config['use_only_compressed_sensing_data']:
        parts.append('_only_cs')
    elif config['include_compressed_sensing_data']:
        parts.append('_with_cs')
    else:
        parts.append('_without_cs')
    
    if config['skip_points_on_centerline']:
        parts.append('_skip')
    else:
        parts.append('_no_skip')
    
    if config['use_updated_ordering_method']:
        parts.append('_updated_ao')
    else:
        parts.append('')

    parts.append(f'_S{config["smoothing"]}')

    # This was added after having to rebalance the dataset to have more age distribution in the test set
    parts.append('_balanced')

    return ''.join(parts)

# ==================================================================
# Helper functions for centerline slicing and extraction (PREPROCESSING)
# ==================================================================


def extract_slice_from_sitk_image(sitk_image, point, Z, X, new_size, fill_value=0):
    """
    Extract oblique slice from SimpleITK image. Efficient, because it rotates the grid and
    only samples the desired slice.

    """
    num_dim = sitk_image.GetDimension()

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())

    new_size = [int(el) for el in new_size]  # SimpleITK expects lists, not ndarrays
    point = [float(el) for el in point]

    rotation_center = sitk_image.TransformContinuousIndexToPhysicalPoint(point)

    X = X / np.linalg.norm(X)
    Z = Z / np.linalg.norm(Z)
    assert np.dot(X, Z) < 1e-12, 'the two input vectors are not perpendicular!'
    Y = np.cross(Z, X)

    orig_frame = np.array(orig_direction).reshape(num_dim, num_dim)
    new_frame = np.array([X, Y, Z])

    # important: when resampling images, the transform is used to map points from the output image space into the input image space
    rot_matrix = np.dot(orig_frame, np.linalg.pinv(new_frame))
    transform = sitk.AffineTransform(rot_matrix.flatten(), np.zeros(num_dim), rotation_center)

    phys_size = new_size * orig_spacing
    new_origin = rotation_center - phys_size / 2

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputSpacing(orig_spacing)
    resample_filter.SetOutputDirection(orig_direction)
    resample_filter.SetOutputOrigin(new_origin)
    resample_filter.SetInterpolator(sitk.sitkLinear)
    resample_filter.SetTransform(transform)
    resample_filter.SetDefaultPixelValue(fill_value)

    resampled_sitk_image = resample_filter.Execute(sitk_image)
    output_dict = {}
    output_dict['resampled_sitk_image'] = resampled_sitk_image
    output_dict['transform'] = transform
    output_dict['origin'] = resampled_sitk_image.GetOrigin()
    return output_dict

def rotate_vectors(vectors, rotation_matrix):
    """
    Rotates a 2D array of 3D vectors using a 3D rotation matrix.
    """
    # Reshape the vectors array for matrix multiplication
    vectors_flat = vectors.reshape(-1, 3)
    vectors_flat[:, [0, 2]] = vectors_flat[:, [2, 0]]
    rotated_vectors_flat = np.dot(rotation_matrix, vectors_flat.T).T
    
    # Reshape back to original shape
    rotated_vectors = rotated_vectors_flat.reshape(vectors.shape)
    
    
    return rotated_vectors


def interpolate_and_slice(image,
                          coords,
                          size, 
                          smoothness=200):


    # Now that I also want to return the geometries, I need to return a dictionary
    slice_dict = {}
    geometry_dict = {}
    #coords are a bit confusing in order...
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]


    coords = np.array([z,y,x]).transpose([1,0])

    #convert the image to SITK (here let's use the intensity for now)
    sitk_image = sitk.GetImageFromArray(image[:,:,:])

    # spline parametrization
    params = [i / (size[2] - 1) for i in range(size[2])]
    tck, _ = interpolate.splprep(np.swapaxes(coords, 0, 1), k=3, s=smoothness)

    # derivative is tangent to the curve
    points = np.swapaxes(interpolate.splev(params, tck, der=0), 0, 1)
    Zs = np.swapaxes(interpolate.splev(params, tck, der=1), 0, 1)
    direc = np.array(sitk_image.GetDirection()[3:6])

    slices = []
    for i in range(len(Zs)):
        geometry_dict[f"slice_{i}"] = {}
        # I define the x'-vector as the projection of the y-vector onto the plane perpendicular to the spline
        xs = (direc - np.dot(direc, Zs[i]) / (np.power(np.linalg.norm(Zs[i]), 2)) * Zs[i])
        output_dict = extract_slice_from_sitk_image(sitk_image, points[i], Zs[i], xs, list(size[:2]) + [1], fill_value=0)
        sitk_slice = output_dict['resampled_sitk_image']
        geometry_dict[f"slice_{i}"]['transform'] = output_dict['transform']
        geometry_dict[f"slice_{i}"]['origin'] = output_dict['origin']
        # Add centerline points to geometry_dict
        geometry_dict[f"slice_{i}"]['centerline_points'] = points[i]
        np_image = sitk.GetArrayFromImage(sitk_slice).transpose(2, 1, 0)
        slices.append(np_image)

    # stick slices together
    slice_dict['straightened'] = np.concatenate(slices, axis=2)
    slice_dict['geometry_dict'] = geometry_dict
    return slice_dict


def nearest_neighbors(q,points,num_neighbors=2,exclude_self=True):
    d = ((points-q)**2).sum(axis=1)  # compute distances
    ndx = d.argsort() # indirect sort 
    start_ind = 1 if exclude_self else 0
    end_ind = start_ind+num_neighbors
    ret_inds = ndx[start_ind:end_ind]
    return ret_inds

def calc_angle(v1, v2, reflex=False):
    dot_prod = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    #round dot_prod for numerical stability
    angle = np.arccos(np.around(dot_prod,6))
    
    if (reflex == False):
        return angle
    else:
        return 2 * np.pi - angle
    
# TODO: Limitation because of the while loop, how do you find the last point
def order_points(candidate_points, angle_threshold=np.pi/2.):
    ordered_points = []
    
    #take first point
    ordered_points.append(candidate_points[0])
    nn = nearest_neighbors(ordered_points[-1], candidate_points,num_neighbors=1)
    #take second point
    ordered_points.append(candidate_points[nn[0]])
    
    remove = 0
    while(len(ordered_points)<len(candidate_points)):
        
        #get 10 nearest neighbors of latest point
        nn = nearest_neighbors(ordered_points[-1], candidate_points,num_neighbors=10)
        # Taking the current point and the previous, we compute the angle to the current and eventual neighbourg
        # making sure its acute
        found = 0
        
        for cp_i in nn:
            ang = calc_angle(ordered_points[-2]-ordered_points[-1], candidate_points[cp_i]-ordered_points[-1])
            if ang > (angle_threshold):
                found =1

                ordered_points.append(candidate_points[cp_i])
            if found == 1:
                break 
        if found ==0:
            if remove >5:
                break
            
            candidate_points = list(candidate_points)
            candidate_points = [arr for arr in candidate_points if not np.array_equal(arr, ordered_points[-1])]
            candidate_points = np.array(candidate_points)
            ordered_points.pop()
            remove += 1
    ordered_points = np.array(ordered_points)

    return(ordered_points)

def skeleton_points(segmented, dilation_k=0, erosion_k = 0):
    # Average the segmentation over time (the geometry should be the same over time)
    avg = np.average(segmented, axis = 3)
    if dilation_k > 0:
        avg = binary_erosion(avg, selem=np.ones((erosion_k, erosion_k,erosion_k)))
        avg = dilation(avg, selem=np.ones((dilation_k, dilation_k,dilation_k)))
        
    # Compute the centerline points of the skeleton
    skeleton = skeletonize_3d(avg[:,:,:])
   
    # Get the points of the centerline as an array
    points = np.array(np.where(skeleton != 0)).transpose([1,0])

    # Order the points in ascending order with x
    points = points[points[:,0].argsort()[::-1]]
    
    return points


# ==================================================================
# ==================================================================
# Helper functions during training
# ==================================================================
# ==================================================================

def save_inputs_outputs(n_image, epoch, input_, ouput_, config, labels=None, training=True):
    if training:
        path_inter_inputs = os.path.join(config['exp_path'], 'intermediate_results/training/inputs')
        path_inter_outputs = os.path.join(config['exp_path'], 'intermediate_results/training/outputs')
    else:
        path_inter_inputs = os.path.join(config['exp_path'], 'intermediate_results/validation/inputs')
        path_inter_outputs = os.path.join(config['exp_path'], 'intermediate_results/validation/outputs')
    
    make_dir_safely(path_inter_inputs)
    make_dir_safely(path_inter_outputs)
    if config['self_supervised']:
        # Apply sigmoid to output
        ouput_ = torch.sigmoid(ouput_)
        if training:
            path_inter_masks = os.path.join(config['exp_path'], 'intermediate_results/training/masks')
        else:
            path_inter_masks = os.path.join(config['exp_path'], 'intermediate_results/validation/masks')
        make_dir_safely(path_inter_masks)
        labels = labels.cpu().detach().numpy()
        np.save(os.path.join(path_inter_masks,f"mask_image_{n_image}_epoch_{epoch}.npy"), labels)


    input_cpu = input_.cpu().detach().numpy()
    output_cpu = ouput_.cpu().detach().numpy()
    np.save(os.path.join(path_inter_inputs,f"input_image_{n_image}_epoch_{epoch}.npy"), input_cpu)
    np.save(os.path.join(path_inter_outputs,f"output_image_{n_image}_epoch_{epoch}.npy"), output_cpu)
    
# Not used in paper, experimental phase (for conditional network)
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
    mod_slices = slices % len(subseq)
    
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

# ==================================================================
# Apply Poisson Image Blending on the fly during training
# ==================================================================

def apply_blending(input_images, images_for_blend, mask_blending, config):
    blending_methods = {
        "mixed_grad": TestPoissonImageEditingMixedGradBlender,
        "source_grad": TestPoissonImageEditingSourceGradBlender,
        "interpolation": TestPatchInterpolationBlender
    }
    # Retrieve the blending method from config
    method_key = config.get('blending', {}).get('method', 'mixed_grad')
    blending_function = blending_methods[method_key]
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
        
        # Random flip #TODO: Remove the random flip but kept for reproducibility of article results, would be better to remove
        if np.random.rand() > 0.5:
            blended_image, anomaly_mask = blending_function(labeller, blender, mask_blending)(input_, mask_blending)
        else:
            blended_image, anomaly_mask = blending_function(labeller, blender, mask_blending)(input_, mask_blending)
            
        
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
# Compute the XYZ Euler angles and trace for each 3x3 rotation matrix in a batch
# ==================================================================

def compute_euler_angles_xyz_and_trace(rotation_matrices):
    """
    Computes the XYZ Euler angles and trace for each 3x3 rotation matrix in a batch.
    Assumes the rotation_matrices are in shape [batch_size, 3, 3].

    :param rotation_matrices: A batch of 3x3 rotation matrices.
    :return: A tuple of (euler_angles, traces), where euler_angles is a tensor of shape [batch_size, 3]
             containing the XYZ Euler angles for each matrix, and traces is a tensor of shape [batch_size, 1]
             containing the trace of each matrix.
    """
    batch_size = rotation_matrices.size(0)
    euler_angles = torch.zeros(batch_size, 3, device=rotation_matrices.device)  # [batch_size, 3] for XYZ Euler angles
    traces = torch.zeros(batch_size, 1, device=rotation_matrices.device)  # [batch_size, 1] for trace

    for i in range(batch_size):
        matrix = rotation_matrices[i]

        # Compute trace
        traces[i] = torch.trace(matrix)

        # Compute Euler angles (XYZ rotation sequence)
        sy = torch.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = torch.atan2(-matrix[1, 2], matrix[2, 2])
            y = torch.atan2(matrix[0, 2], sy)
            z = torch.atan2(-matrix[0, 1], matrix[0, 0])
        else:
            x = torch.atan2(-matrix[1, 2], matrix[1, 1])
            y = torch.atan2(matrix[0, 2], sy)
            z = 0

        euler_angles[i] = torch.tensor([x, y, z], device=rotation_matrices.device)

    dict_results = {'euler_angles': euler_angles, 'trace': traces}

    return dict_results

# ==================================================================
# Compute losses
# ==================================================================


def compute_losses(input_images, output_dict, config, input_dict= None):
    # Compute the standard VAE losses
    gen_loss = l2loss(input_images, output_dict['decoder_output'])
    res_loss = torch.zeros_like(gen_loss)  # Assuming residual loss is computed elsewhere or not needed here
    lat_loss = kl_loss_1d(output_dict['mu'], output_dict['z_std'])
    gen_factor_loss = config['gen_loss_factor'] * gen_loss

    # Initialize auxiliary losses
    euler_loss = torch.tensor(0.0)
    trace_loss = torch.tensor(0.0)

    # Check if the model is encoder decoder and compute auxiliary losses
    if config['model'].__contains__('conv_enc_dec_aux'):
        # Assuming you have ground truth values for Euler angles and trace
        true_euler_angles = input_dict['euler_angles']
        true_trace = input_dict['trace']

        # Compute auxiliary losses
        predicted_euler_angles = output_dict['euler_angles']
        predicted_trace = output_dict['trace']
        
        euler_loss_factor = config.get('euler_angle_loss_factor', 1.0)
        # Use MSE loss for Euler angles and trace
        euler_loss = torch.mean((predicted_euler_angles - true_euler_angles) ** 2, dim=1)
        euler_loss = euler_loss_factor * euler_loss
        trace_loss = torch.mean((predicted_trace - true_trace) ** 2, dim=1)

        # Factor to adjust the importance of auxiliary losses
        aux_loss_factor = config.get('aux_loss_factor', 1.0)

        # Incorporate auxiliary losses into the total loss
        total_aux_loss = aux_loss_factor * (euler_loss + trace_loss)
    else:
        total_aux_loss = torch.tensor(0.0)

    # Compute total loss
    total_loss = torch.mean(gen_factor_loss + lat_loss + total_aux_loss)

    # For validation loss, you might consider including or excluding the auxiliary losses
    val_loss = torch.mean(gen_loss + lat_loss + total_aux_loss)

    # Save the losses in a dictionary
    dict_loss = {
        'loss': total_loss,
        'val_loss': val_loss,
        'gen_factor_loss': gen_factor_loss,
        'gen_loss': gen_loss,
        'res_loss': res_loss,
        'lat_loss': lat_loss,
        'euler_loss': euler_loss,
        'trace_loss': trace_loss,
        'total_aux_loss': total_aux_loss
    }

    return dict_loss


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




# ==================================================================
# ==================================================================
# Utils for backtransforming the anomaly scores
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

# To visualize the backtransformed anomaly scores we need to convert them to vtk
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
    # Convert to vtk
    convert_to_vtk(backtransformed_anomaly_score, subject_id, output_dir)



# ==================================================================    
# Helper functions for quadrants creation
# ==================================================================    

def compute_quadrant_mask_main_axes(ang, ang2):
    """
    Compute a mask that highlights a quadrant based on the main axes angles.

    Parameters:
    ang (float): Primary angle in radians.
    ang2 (float): Secondary angle in radians.

    Returns:
    torch.Tensor: A tensor representing the quadrant mask.
    """
    quad_mask = torch.zeros([1, 1, 32, 32])

    # Create a grid of coordinates ranging from -1 to 1
    d1 = torch.linspace(1., -1., quad_mask.shape[-2])
    d2 = torch.linspace(-1., 1., quad_mask.shape[-1])
    d1v, d2v = torch.meshgrid(d1, d2, indexing='ij')

    for i in range(quad_mask.shape[-2]):
        for j in range(quad_mask.shape[-1]):
            cur_coor = torch.complex(d2v[i, j], d1v[i, j])
            ref_angle = torch.polar(torch.tensor(1.), torch.tensor(ang, dtype=torch.float32))
            ref_angle2 = torch.polar(torch.tensor(1.), torch.tensor(ang2, dtype=torch.float32))
            pi_angle = torch.polar(torch.tensor(1.), torch.tensor(np.pi, dtype=torch.float32))

            # Calculate half angle difference between primary and secondary vectors
            cur_ref_diff = torch.angle(cur_coor * torch.conj(ref_angle))
            ref2_ref_diff = torch.angle(ref_angle2 * torch.conj(ref_angle)) / 2.
            ref2pi_ref_diff = torch.angle((ref_angle2 * pi_angle) * torch.conj(ref_angle)) / 2.

            # Determine which axis is on the same side of the midpoint
            if torch.sign(cur_ref_diff) == torch.sign(ref2_ref_diff):
                closer_ref_diff = ref2_ref_diff
            else:
                closer_ref_diff = ref2pi_ref_diff

            # Assign the mask value based on the angle difference
            if torch.abs(cur_ref_diff) < torch.abs(closer_ref_diff):
                quad_mask[0, 0, i, j] = 1

    return quad_mask

def create_all_quadrant_masks_main_axes(ax_angles):
    """
    Create masks for all quadrants based on the main axes angles (extracted thanks to sobel filters).
    1. A mask with quandrants, Posterior, Right, Anterior, Left

    Parameters:
    ax_angles (list): List of two angles [primary_angle, secondary_angle] in radians.

    Returns:
    torch.Tensor: A combined tensor of all quadrant masks with shape [1, 4, 32, 32].
    """
    primary_angle = ax_angles[0]  # Anterior-Posterior angle
    secondary_angle = ax_angles[1]  # Left-Right angle

    # Compute quadrant masks
    quad_mask_P = compute_quadrant_mask_main_axes(primary_angle, secondary_angle)  # Posterior
    quad_mask_R = compute_quadrant_mask_main_axes(secondary_angle, primary_angle)  # Right
    quad_mask_A = compute_quadrant_mask_main_axes(primary_angle + np.pi, secondary_angle + np.pi)  # Anterior
    quad_mask_L = compute_quadrant_mask_main_axes(secondary_angle + np.pi, primary_angle + np.pi)  # Left

    quad_masks = [quad_mask_P, quad_mask_R, quad_mask_A, quad_mask_L]

    # Stack masks into a single tensor of shape [1, 4, 32, 32]
    combined_mask = torch.cat(quad_masks, dim=1)

    return combined_mask



def compute_single_quadrant_mask_between_axes(ang, ang2):
    """
    Compute a mask that highlights a single quadrant based on the provided angles.

    Parameters:
    ang (float): Primary angle in radians.
    ang2 (float): Secondary angle in radians.

    Returns:
    torch.Tensor: A tensor representing the quadrant mask.
    """
    quad_mask = torch.zeros([32, 32])

    # Create a grid of coordinates ranging from -1 to 1
    d1 = torch.linspace(1., -1., 32)
    d2 = torch.linspace(-1., 1., 32)
    d1v, d2v = torch.meshgrid(d1, d2, indexing='ij')

    # Loop over each point in the grid
    for i in range(32):
        for j in range(32):
            # Calculate the current coordinate as a complex number
            cur_coor = torch.complex(d2v[i, j], d1v[i, j])
            # Create polar coordinates for the reference angles
            ref_angle = torch.polar(torch.tensor(1.), torch.tensor(ang, dtype=torch.float32))
            ref_angle2 = torch.polar(torch.tensor(1.), torch.tensor(ang2, dtype=torch.float32))
        
            # Calculate the midpoint angle between ref_angle and ref_angle2
            mid_angle = (ref_angle + ref_angle2) / 2.
            # Calculate angle differences
            cur_mid_diff = torch.angle(cur_coor * torch.conj(mid_angle))
            ref_mid_diff = torch.angle(ref_angle * torch.conj(mid_angle))
            ref_mid_diff2 = torch.angle(ref_angle2 * torch.conj(mid_angle))
        
            # Determine which reference angle is closer
            if torch.sign(cur_mid_diff) == torch.sign(ref_mid_diff):
                closer_ref_diff = ref_mid_diff
            else:
                closer_ref_diff = ref_mid_diff2
            
            # Assign the mask value based on the angle difference
            if torch.abs(cur_mid_diff) < torch.abs(closer_ref_diff):
                quad_mask[i, j] = 1

    return quad_mask

def compute_all_quadrant_masks_between_axes(ax_angles):
    """
    Compute masks for all quadrants based on the main axes angles. We take them in between.
    2. A mask with quandrants, Posterior-right, Anterior-right, Posterior-left, Anterior-left 

    Parameters:
    ax_angles (list): List of two angles [primary_angle, secondary_angle] in radians.

    Returns:
    torch.Tensor: A combined tensor of all quadrant masks with shape [1, 4, 32, 32].
    """
    primary_angle = ax_angles[0]
    secondary_angle = ax_angles[1]
    
    # Define the angles for each quadrant
    angles = [
        (primary_angle, secondary_angle),  # Quadrant between primary and secondary angles
        (primary_angle + np.pi, secondary_angle),  # Opposite of primary angle
        (secondary_angle + np.pi, primary_angle),  # Opposite of secondary angle
        (primary_angle + np.pi, secondary_angle + np.pi)  # Both angles rotated by 180 degrees
    ]
    
    # Compute masks for each quadrant
    masks = [compute_single_quadrant_mask_between_axes(a1, a2) for a1, a2 in angles]
    
    # Stack masks into a single tensor of shape [1, 4, 32, 32]
    return torch.stack(masks).unsqueeze(0)



