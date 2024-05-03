import os
import h5py
import numpy as np
import sys
from skimage.morphology import skeletonize_3d, dilation, cube, binary_erosion
from skimage.restoration import unwrap_phase


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/helpers/')

from utils import verify_leakage ,crop_or_pad_Bern_slices, normalize_image, normalize_image_new, make_dir_safely, crop_or_pad_normal_slices
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/')


import config.system as sys_config
import SimpleITK as sitk
from scipy import interpolate

#
# UTILS
#


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

# FULL AORTA PROCESSING UTILS

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



#====================================================================================
# CENTER LINE
#====================================================================================

def prepare_and_write_masked_data_bern(basepath,
                           filepath_output,
                            idx_start,
                            idx_end,
                           train_test,
                           suffix =''):

    # ==========================================
    # Study the the variation in the sizes along various dimensions (using the function 'find_shapes'),
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # ==========================================
    # For Bern the max sizes are:
    # x: 144, y: 112, z: 64, t: 33 (but because of network we keep 48) nope we actually going for 24 cause 33 is only very few people
    common_image_shape = [144, 112, 40, 24, 4] # [x, y, z, t, num_channels]
    common_label_shape = [144, 112, 40, 24] # [x, y, z, t]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)

    # ==========================================
    # ==========================================
    if ['train', 'val'].__contains__(train_test):
        seg_path = basepath + f'/final_segmentations/train_val'
        img_path = basepath + f'/preprocessed/controls/numpy'
    elif train_test == 'test':
        # For the img_path we need to look into the patients folder or the controls folder, try both, see further down
        seg_path = basepath + f'/final_segmentations/test'
        img_path = basepath + f'/preprocessed/patients/numpy'
    else:
        raise ValueError('train_test must be either train, val or test')
    
    seg_path_files = os.listdir(seg_path)
    # Sort
    seg_path_files.sort()
    patients = seg_path_files[idx_start:idx_end]
    num_images_to_load = len(patients)


    
    # ==========================================
    # we will stack all images along their z-axis
    # --> the network will analyze (x,y,t) volumes, with z-samples being treated independently.
    # ==========================================
    images_dataset_shape = [common_image_shape[2]*num_images_to_load,
                            common_image_shape[0],
                            common_image_shape[1],
                            common_image_shape[3],
                            common_image_shape[4]]

    # ==========================================
    # create a hdf5 file
    # ==========================================
    dataset = {}
    hdf5_file = h5py.File(filepath_output, "w")

    # ==========================================
    # write each subject's image and label data in the hdf5 file
    # ==========================================
    dataset['masked_images_%s' % train_test] = hdf5_file.create_dataset("masked_images_%s" % train_test, images_dataset_shape, dtype='float32')
    

    i = 0
    for patient in patients: 
        
        logging.info('loading subject ' + str(i+1) + ' out of ' + str(num_images_to_load)  + '...')
        logging.info('patient %s', patient)
        
        if ['train', 'val'].__contains__(train_test):
            image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        elif train_test == 'test':
            # We need to look into the patients folder or the controls folder, try both
            try:
                # With patients
                image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
            except:
                # With controls
                image = np.load(os.path.join(img_path.replace("patients", "controls"), patient.replace("seg_", "")))
        
        # normalize the image
        image = normalize_image_new(image)
        logging.info('Shape of image before network resizing',image.shape)
        # The images need to be sized as in the input of network
        
        
        # make all images of the same shape
        image = crop_or_pad_Bern_slices(image, common_image_shape)
        logging.info('Shape of image after network resizing',image.shape)
        # move the z-axis to the front, as we want to concantenate data along this axis
        image = np.moveaxis(image, 2, 0)

        label_data = np.load(os.path.join(seg_path, patient))
        # make all images of the same shape
        label_data = crop_or_pad_Bern_slices(label_data, common_label_shape)
        # move the z-axis to the front, as we want to concantenate data along this axis
        label_data = np.moveaxis(label_data, 2, 0)
        # cast labels as uints
        label_data = label_data.astype(np.uint8)


        temp_images_intensity = image[:,:,:,:,0] * label_data
        temp_images_vx = image[:,:,:,:,1] * label_data
        temp_images_vy = image[:,:,:,:,2] * label_data
        temp_images_vz = image[:,:,:,:,3] * label_data

        #recombine the images
        image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)

        # add the image to the hdf5 file
        dataset['masked_images_%s' % train_test][i*common_image_shape[2]:(i+1)*common_image_shape[2], :, :, :, :] = image

        # increment the index being used to write in the hdf5 datasets
        i = i + 1

    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0



def load_masked_data(basepath,
              idx_start,
              idx_end,
              train_test,
              force_overwrite=False,
              suffix =''):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    savepath = sys_config.project_code_root + 'data'
    dataset_filepath = savepath + f'/{train_test}{suffix}_masked_images_from_' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        logging.info('This configuration has not yet been preprocessed.')
        logging.info('Preprocessing now...')
        prepare_and_write_masked_data_bern(basepath = basepath,
                               filepath_output = dataset_filepath,
                               idx_start = idx_start,
                                 idx_end = idx_end,
                               train_test = train_test,
                               suffix = suffix)
    else:
        logging.info('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')



# ====================================================================================
# CROPPED AND STRAIGHTENED AORTA DATA Z-SLICES
#====================================================================================
def prepare_and_write_sliced_data_bern(basepath,
                           filepath_output,
                           idx_start,
                            idx_end,
                           train_test,
                           stack_z):

    # ==========================================
    # Study the the variation in the sizes along various dimensions (using the function 'find_shapes'),
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # ==========================================
    common_image_shape = [36, 36, 64, 24, 4] # [x, y, z, t, num_channels]

    #network_common_image_shape = [144, 112, None, 24, 4] # [x, y, t, num_channels]

    end_shape = [32, 32, 64, 24, 4]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)
    
    # ==========================================
    # ==========================================
    
    savepath_geometry = sys_config.project_code_root + 'data' + '/geometry_for_backtransformation'
    make_dir_safely(savepath_geometry)
    hand_seg_path_controls = basepath + '/segmenter_rw_pw_hard/controls'
    hand_seg_path_patients = basepath + '/segmenter_rw_pw_hard/patients'
    list_hand_seg_images = os.listdir(hand_seg_path_controls) + os.listdir(hand_seg_path_patients)
    # Sort the list
    list_hand_seg_images.sort()
    if ['train', 'val'].__contains__(train_test):

        seg_path = basepath + '/final_segmentations/train_val'
        img_path = basepath + '/preprocessed/controls/numpy'
    elif train_test == 'test':
        # For the img_path we need to look into the patients folder or the controls folder, try both, see further down
        seg_path = basepath + '/final_segmentations/test'
        img_path = basepath + '/preprocessed/patients/numpy'
    else:
        raise ValueError('train_test must be either train, val or test')
    
    seg_path_files = os.listdir(seg_path)
    # Sort
    seg_path_files.sort()

    patients = seg_path_files[idx_start:idx_end]
    num_images_to_load = len(patients)
    


    if stack_z == True:
        # ==========================================
        # we will stack all images along their z-axis
        # --> the network will analyze (x,y,t) volumes, with z-samples being treated independently.
        # ==========================================
        images_dataset_shape = [end_shape[2]*num_images_to_load,
                                end_shape[0],
                                end_shape[1],
                                end_shape[3],
                                end_shape[4]]
    else:
        # ==========================================
        # If we are not stacking along z (the centerline of the cropped aorta),
        # we are stacking along y (so shape[1])
        # ==========================================
        images_dataset_shape = [end_shape[1]*num_images_to_load,
                                end_shape[0],
                                end_shape[2],
                                end_shape[3],
                                end_shape[4]]


    # ==========================================
    # create a hdf5 file
    # ==========================================
    dataset = {}
    hdf5_file = h5py.File(filepath_output, "w")

    # ==========================================
    # write each subject's image and label data in the hdf5 file
    # ==========================================
    if stack_z == True:
        dataset['sliced_images_%s' % train_test] = hdf5_file.create_dataset("sliced_images_%s" % train_test, images_dataset_shape, dtype='float32')
    else:
        dataset['straightened_images_%s' % train_test] = hdf5_file.create_dataset("straightened_images_%s" % train_test, images_dataset_shape, dtype='uint8')

    i = 0
    
    cnn_predictions = True
    # Filter the patients by name to get the ones from hand-segmented and cnn predictions
    for patient in patients: 

        logging.info('loading subject ' + str(i+1) + ' out of ' + str(num_images_to_load) + '...')
        logging.info('patient: ' + patient)

        # Check if hand or network segemented (slightly different kernel size on pre-processing)
        if patient in list_hand_seg_images:
            cnn_predictions = False

        # load the segmentation that was created with Nicolas's tool
        if ['train', 'val'].__contains__(train_test):
            image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        elif train_test == 'test':
            # We need to look into the patients folder or the controls folder, try both
            try:
                # With patients
                image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
            except:
                # With controls
                image = np.load(os.path.join(img_path.replace("patients", "controls"), patient.replace("seg_", "")))
        
        
        segmented = np.load(os.path.join(seg_path, patient))
        
        
        image = normalize_image_new(image)

        # Compute the centerline points of the skeleton
        if cnn_predictions:
            points_ = skeleton_points(segmented, dilation_k = 0)
            points_dilated = skeleton_points(segmented, dilation_k = 4,erosion_k = 4)
        else:
            points_ = skeleton_points(segmented, dilation_k = 0)
            points_dilated = skeleton_points(segmented, dilation_k = 2,erosion_k = 2)
        points = points_dilated.copy()


        """
        # Average the segmentation over time (the geometry should be the same over time)
        avg = np.average(segmented, axis = 3)

        # Compute the centerline points of the skeleton
        skeleton = skeletonize_3d(avg[:,:,:])

        # Get the points of the centerline as an array
        points = np.array(np.where(skeleton != 0)).transpose([1,0])
        """

        # Limit to sectors where ascending aorta is located
        points = points[np.where(points[:,1]<60)]
        points = points[np.where(points[:,0]<90)]

        # Order the points in ascending order with x
        points = points[points[:,0].argsort()[::-1]]

        temp = []
        for index, element in enumerate(points[2:]):
            if (index%2)==0:
                temp.append(element)


        coords = np.array(temp)
        

        #===========================================================================================
        # Parameters for the interpolation and creation of the files

        # We create Slices across time and channels in a double for loop
        temp_for_channel_stacking = []
        geometry_saved = False
        for channel in range(image.shape[4]):

            temp_for_time_stacking = []
            for t in range(image.shape[3]):
                if not geometry_saved:
                    save_for_backtransformation = True
                else:
                    save_for_backtransformation = False
                slice_dict = interpolate_and_slice(image[:,:,:,t,channel], coords, common_image_shape, smoothness=10)
                if save_for_backtransformation:
                    geometry_saved = True  
                    geometry_dict = slice_dict['geometry_dict']
                straightened = slice_dict['straightened']
                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        straightened = np.stack(temp_for_channel_stacking, axis=-1)

        image_out  = straightened
        # Save the geometries for backtransformation with name of patient
        np.save(savepath_geometry + '/' + patient.replace('seg',''), geometry_dict)

        # make all images of the same shape
        logging.info("Image shape before cropping and padding:" + str(image_out.shape))
        #image_out = crop_or_pad_Bern_all_slices(image_out, network_common_image_shape)
        image_out = crop_or_pad_normal_slices(image_out, end_shape)
        logging.info("Image shape after cropping and padding:" + str(image_out.shape))

        if stack_z == True:
            # move the z-axis to the front, as we want to stack the data along this axis
            image_out = np.moveaxis(image_out, 2, 0)

            # add the image to the hdf5 file
            dataset['sliced_images_%s' % train_test][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out

        else:
            # move the y-axis to the front, as we want to stack the data along this axis
            image_out = np.moveaxis(image_out, 1, 0)

            logging.info('After shuffling the axis' + str(image_out.shape))
            logging.info(str(np.max(image_out)))

            # add the image to the hdf5 file
            dataset['straightened_images_%s' % train_test][i*end_shape[1]:(i+1)*end_shape[1], :, :, :, :] = image_out

        # increment the index being used to write in the hdf5 datasets
        i = i + 1

    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0

# ==========================================
# ==========================================
def load_cropped_data_sliced(basepath,
              idx_start,
              idx_end,
              train_test,
              force_overwrite=False):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    savepath = sys_config.project_code_root + 'data'
    dataset_filepath = savepath + f'/{train_test}_sliced_images_from_' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        logging.info('This configuration has not yet been preprocessed.')
        logging.info('Preprocessing now...')
        prepare_and_write_sliced_data_bern(basepath = basepath,
                               filepath_output = dataset_filepath,
                               idx_start=idx_start,
                                 idx_end=idx_end,
                               train_test = train_test,
                               stack_z = True)
    else:
        logging.info('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')
# ====================================================================================
# *** CROPPED AND STRAIGHTENED AORTA DATA Z-SLICES *** END
#====================================================================================


# ====================================================================================
# *** MASKED SLICED DATA ****
#====================================================================================

def find_and_load_image(patient, basepaths):
    for path in basepaths:
        try:
            return np.load(os.path.join(path, patient.replace("seg_", "")))
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Image for patient {patient} not found in any of the provided paths.")


def prepare_and_write_masked_data_sliced_bern(basepath,
                           filepath_output,
                           idx_start,
                           idx_end,
                           train_test,
                           load_anomalous=False,
                           include_compressed_sensing=True,
                           only_compressed_sensing=False,
                           suffix ='',
                           updated_ao = False,
                           skip = True,
                           smoothness = 10,
                           unwrapped = False):

    # ==========================================
    # Study the the variation in the sizes along various dimensions (using the function 'find_shapes'),
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # ==========================================
    common_image_shape = [36, 36, 64, 24, 4] # [x, y, z, t, num_channels]
    
    end_shape = [32, 32, 64, 24, 4]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)

    # ==========================================
    # ==========================================

    # Log some parameters
    logging.info(f"include_compressed_sensing: {include_compressed_sensing}")
    logging.info(f"updated_ao: {updated_ao}")
    logging.info(f"skip: {skip}")
    logging.info(f"smoothness: {smoothness}")
    logging.info(f"only_compressed_sensing: {only_compressed_sensing}")


    savepath_geometry = sys_config.project_code_root + 'data' + f'/geometry_for_backtransformation'
    make_dir_safely(savepath_geometry)
    hand_seg_path_controls = [basepath + f'/segmenter_rw_pw_hard/controls',
                              basepath + f'/segmenter_rw_pw_hard/controls_compressed_sensing']
    hand_seg_path_patients = [basepath + f'/segmenter_rw_pw_hard/patients',
                              basepath + f'/segmenter_rw_pw_hard/patients_compressed_sensing']
    list_hand_seg_images = []
    for path in hand_seg_path_controls + hand_seg_path_patients:
        list_hand_seg_images.extend(os.listdir(path))
    # Sort the list
    list_hand_seg_images.sort()
    if ((['train', 'val'].__contains__(train_test)) and (not only_compressed_sensing)):
        seg_path = basepath + f'/final_segmentations/train_val'
        img_path = basepath + f'/preprocessed/controls/numpy'
        img_path_compressed = basepath + f'/preprocessed/controls/numpy_compressed_sensing'

    elif ((train_test == 'test') and (not only_compressed_sensing)):
        seg_path = basepath + f'/final_segmentations/test'
        img_path = basepath + f'/preprocessed/patients/numpy'
        img_path_compressed = basepath + f'/preprocessed/patients/numpy_compressed_sensing'
        img_paths = [img_path, img_path_compressed]
    elif ((['train', 'val'].__contains__(train_test)) and (only_compressed_sensing)):
        seg_path = basepath + f'/final_segmentations/train_val_compressed_sensing'
    
    elif ((train_test == 'test') and (only_compressed_sensing)):
        seg_path = basepath + f'/final_segmentations/test_compressed_sensing'
        
    else:
        raise ValueError('train_test must be either train, val or test')



    if only_compressed_sensing:
        # Use only paths with compressed sensing data
        img_paths_controls = [basepath + f'/preprocessed/controls/numpy_compressed_sensing']
        img_paths_patients = [basepath + f'/preprocessed/patients/numpy_compressed_sensing']
    else:
        # Original behavior, possibly including both types of data
        img_paths_controls = [basepath + f'/preprocessed/controls/numpy']
        img_paths_patients = [basepath + f'/preprocessed/patients/numpy']
        if include_compressed_sensing:
            img_paths_controls.append(basepath + f'/preprocessed/controls/numpy_compressed_sensing')
            img_paths_patients.append(basepath + f'/preprocessed/patients/numpy_compressed_sensing')



    # Paths for non compressed sensing data

    
    seg_path_files = os.listdir(seg_path)
    # Sort
    seg_path_files.sort()
    # Filter based on the include_compressed_sensing flag
    filtered_seg_path_files = []
    for file in seg_path_files:
        corresponding_img_file = file.replace('seg_', '')

        # Check for file existence based on the selected paths
        file_exists = any(os.path.exists(os.path.join(path, corresponding_img_file)) 
                        for path in (img_paths_controls + img_paths_patients))

        if file_exists:
            filtered_seg_path_files.append(file)

    # Use the filtered list for further processing
    patients = filtered_seg_path_files[idx_start:idx_end]
    num_images_to_load = len(patients)


    
    #patients = seg_path_files[idx_start:idx_end]
    #num_images_to_load = len(patients)

    # ==========================================
    # we will stack all images along their z-axis
    # --> the network will analyze (x,y,t) volumes, with z-samples being treated independently.
    # ==========================================
    images_dataset_shape = [end_shape[2]*num_images_to_load,
                            end_shape[0],
                            end_shape[1],
                            end_shape[3],
                            end_shape[4]]

    # ==========================================
    # create a hdf5 file
    # ==========================================
    dataset = {}
    hdf5_file = h5py.File(filepath_output, "w")

    # ==========================================
    # write each subject's image and label data in the hdf5 file
    # ==========================================
    dataset['sliced_images_%s' % train_test] = hdf5_file.create_dataset("sliced_images_%s" % train_test, images_dataset_shape, dtype='float32')

    # If test then add label depending on if anomalous or not, 1 or 0
    if train_test == 'test':
        dataset['labels_%s' % train_test] = hdf5_file.create_dataset("labels_%s" % train_test, (end_shape[2]*num_images_to_load,), dtype='uint8')

    # Add the geometry for backtransformation, each slice has its own geometry
    dataset['rotation_matrix'] = hdf5_file.create_dataset("rotation_matrix", (end_shape[2]*num_images_to_load,3,3), dtype='float32')
    
    cnn_predictions = True
    i = 0
    for patient in patients: 
        
        
        logging.info('loading subject ' + str(i+1) + ' out of ' + str(num_images_to_load) + '...')

        logging.info('patient: ' + patient)
        # Check if hand or network segemented (slightly different kernel size on pre-processing)
        if ['train', 'val'].__contains__(train_test):
            # For train and val, try both controls paths
            image = find_and_load_image(patient, img_paths_controls)
        elif train_test == 'test':
            # For test, try all four paths
            try:
                image = find_and_load_image(patient, img_paths_patients)
                label = np.ones(end_shape[2])
                logging.info('label: sick')
            except FileNotFoundError:
                image = find_and_load_image(patient, img_paths_controls)
                label = np.zeros(end_shape[2])
                logging.info('label: healthy')

        segmented_original = np.load(os.path.join(seg_path, patient))

        

        # Enlarge the segmentation slightly to be sure that there are no cutoffs of the aorta
        time_steps = segmented_original.shape[3]
        #segmented = dilation(segmented_original[:,:,:,3], cube(6))
        segmented = dilation(segmented_original[:,:,:,3], cube(3))

        temp_for_stack = [segmented for i in range(time_steps)]
        segmented = np.stack(temp_for_stack, axis=3)

        image = image.astype(float)

        if unwrapped:

            # Create a masked array where the background is masked
            segmented_bool = abs(segmented - 1).astype(bool)
            segmented_bool = np.expand_dims(segmented_bool, axis=4)
            # repeat the segmented_boolmentation mask for each channel
            segmented_bool = np.repeat(segmented_bool, 4, axis=4)

            # Make a masked array, where the mask is True where the image is True

            masked_image = np.ma.masked_array(image, segmented_bool)

            # Unwrapping the phase
            unwrapped_image = np.zeros_like(masked_image)
            for channel in range(1, masked_image.shape[-1]):
                # Iterate over each time step
                for t_ in range(masked_image.shape[3]):
                    # Extract the example_image for the current channel and time step
                    example_image = masked_image[..., t_, channel]

                    # Scale the intensity values to the range of phase (-pi to pi)
                    scaled_phase = (example_image / 4096.0) * 2.0 * np.pi - np.pi

                    # Perform phase unwrapping
                    unwrapped_phase = unwrap_phase(scaled_phase, wrap_around=False)

                    # Scale the unwrapped phase values back to the range of intensity 
                    scaled_intensity = ((unwrapped_phase + np.pi) / (2.0 * np.pi)) * 4096.0

                    # Assign the restored intensity values back to the original image array
                    unwrapped_image[..., t_, channel] = scaled_intensity

            # Clip the range of intensity values 25 % more and less than the original range [0, 4096] --> [-1024, 5120]
            # This aims to account for some noise in the unwrapping process
            unwrapped_image = np.clip(unwrapped_image, -1024, 5120)

            # Substract 2048 to center the values around 0 for the velocity channels
            image[...,1:] -= 2048.0
            
            # Add the first channel (magnitude) back to the unwrapped image
            unwrapped_image[...,0] = masked_image[...,0]

            #Fill masked values with 0
            image = np.ma.filled(unwrapped_image, 0.0)
        else:
            # Subtract 2048 to center the values around 0 for the velocity channels
            image[...,1:] -= 2048.0
            temp_images_intensity = image[:,:,:,:,0] * segmented # change these back if it works
            temp_images_vx = image[:,:,:,:,1] * segmented
            temp_images_vy = image[:,:,:,:,2] * segmented
            temp_images_vz = image[:,:,:,:,3] * segmented
            image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)

        # Normalize the image

        image = normalize_image_new(image, with_set_range=[-2048, 2048])


        """
        Original approach
        Normalize before segmentation
        
       
        # normalize image to -1 to 1
        image = normalize_image_new(image)
        seg_shape = list(segmented.shape)
        seg_shape.append(image.shape[-1])
        image = crop_or_pad_Bern_slices(image, seg_shape)

        temp_images_intensity = image[:,:,:,:,0] * segmented # change these back if it works
        temp_images_vx = image[:,:,:,:,1] * segmented
        temp_images_vy = image[:,:,:,:,2] * segmented
        temp_images_vz = image[:,:,:,:,3] * segmented

        # recombine the images
        image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)
        

        
        Second approach
        Normalize after segmentation (I also change to float) and set everything to zero
        

        image = image.astype(float)
        segmented_bool = abs(segmented - 1).astype(bool)
        if unwrapped:
            
            segmented_bool = np.expand_dims(segmented_bool, axis=4)
            segmented_bool = np.repeat(segmented_bool, 4, axis=4)
            
            masked_image = np.ma.masked_array(image, segmented_bool)


            unwrapped_image = np.zeros_like(masked_image)
            for channel in range(1, masked_image.shape[-1]):
                # Iterate over each time step
                for t_ in range(masked_image.shape[3]):
                    # Extract the example_image for the current channel and time step
                    example_image = masked_image[..., t_, channel]

                    # Scale the intensity values to the range of phase (-pi to pi)
                    scaled_phase = (example_image / 4096) * 2 * np.pi - np.pi

                    # Perform phase unwrapping
                    unwrapped_phase = unwrap_phase(scaled_phase, seed=None, wrap_around=False)

                    # Scale the unwrapped phase values back to the range of intensity 
                    scaled_intensity = ((unwrapped_phase + np.pi) / (2 * np.pi)) * 4096

                    # Assign the restored intensity values back to the original image array
                    unwrapped_image[..., t_, channel] = scaled_intensity

            # Give values to the masked unwrapped image
            unwrapped_image = np.ma.filled(unwrapped_image, 0.0)
            # Unwrapping is only done on the velocity channels 
            image[...,1:] = unwrapped_image[...,1:]
            # We therefore need to mask with segmentation the first channel
            image[...,0] = image[...,0] * segmented

        else:
            
            temp_images_intensity = image[:,:,:,:,0] * segmented # change these back if it works
            temp_images_vx = image[:,:,:,:,1] * segmented
            temp_images_vy = image[:,:,:,:,2] * segmented
            temp_images_vz = image[:,:,:,:,3] * segmented
            # recombine the images
            image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)
        

        # normalize image to -1 to 1
        if unwrapped:
            # Normalize using 2 nad 98 th percentile
            image = normalize_image_new(image, with_percentile=True)
            # Set all background to zero
            image[segmented_bool] = 0
            
        else:
            image = normalize_image_new(image)
            # Set all background to zero
            image[segmented_bool] = 0




        # End second approach
        

        # Third approach
        
        Original approach
        Normalize before segmentation for velocity channels but not magnitude channel
        
        

        
        # Remove from the first channel the background

        mask_mag_channel = np.expand_dims(abs(segmented -1), axis=-1)
        # We don't want to affect velocity channels
        mask_zeros_vel_channel = np.zeros_like(mask_mag_channel)
        # Repeat for each channel and concatenate
        mask_zeros_vel_channel = np.repeat(mask_zeros_vel_channel, 3, axis=-1)
        mask = np.concatenate([mask_mag_channel, mask_zeros_vel_channel], axis=-1)

        image_mag_masked = np.ma.masked_array(image, mask=mask)

        # Normalize all channels
        image = normalize_image_new(image_mag_masked)

        temp_images_intensity = image[:,:,:,:,0] * segmented # change these back if it works
        temp_images_vx = image[:,:,:,:,1] * segmented
        temp_images_vy = image[:,:,:,:,2] * segmented
        temp_images_vz = image[:,:,:,:,3] * segmented

        # recombine the images
        image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)

        # End of third approach

        

        # Fourth approach
        
        Unwrap and then normalize before segmentation for velocity channels but not magnitude channel
        1. Make a masked array that affects only the vel channels 
        2. Unwrap the phase of the masked array
        3. Fill the masked array with the unwrapped phase
        4. Create a mask where the magnitude channel is masked with segmentation
        5. Normalize the whole image
        6. Segment the last three channels        
        
        
        

        if unwrapped:
            segmented_bool = abs(segmented - 1).astype(bool)
            segmented_bool = np.expand_dims(segmented_bool, axis=4)
            segmented_bool = np.repeat(segmented_bool, 4, axis=4)
            # We want to mask the whole first channel and the velocities of segmentation
            segmented_bool[...,0] = True

            masked_image = np.ma.masked_array(image, segmented_bool)

            unwrapped_image = np.zeros_like(masked_image)
            for channel in range(1, masked_image.shape[-1]):
                # Iterate over each time step
                for t_ in range(masked_image.shape[3]):
                    # Extract the example_image for the current channel and time step
                    example_image = masked_image[..., t_, channel]

                    # Scale the intensity values to the range of phase (-pi to pi)
                    scaled_phase = (example_image / 4096) * 2 * np.pi - np.pi

                    # Perform phase unwrapping
                    unwrapped_phase = unwrap_phase(scaled_phase, seed=None, wrap_around=False)

                    # Scale the unwrapped phase values back to the range of intensity 
                    scaled_intensity = ((unwrapped_phase + np.pi) / (2 * np.pi)) * 4096

                    # Assign the restored intensity values back to the original image array
                    unwrapped_image[..., t_, channel] = scaled_intensity

            # Create a mask for the non-existent values in the unwrapped image
            mask_unwrapped = np.ma.getmask(unwrapped_image)

            # Create a mask array where we need to fill in the image
            fill_masked_image = np.ma.masked_array(image, mask=~mask_unwrapped)

            # Combine the masked unwrapped image with the masked original image
            image = unwrapped_image.filled(fill_value=0) + fill_masked_image.filled(fill_value=0)

            # Use segmentation mask for first channel
            image[...,0] = image[...,0] * segmented

            # Normalize the whole image
            image = normalize_image_new(image, with_percentile=True)

        else:
            image = normalize_image_new(image, with_percentile=False)

        
        
        
        # We can now segment the last three channels
        temp_images_intensity = image[:,:,:,:,0] * segmented
        temp_images_vx = image[:,:,:,:,1] * segmented
        temp_images_vy = image[:,:,:,:,2] * segmented
        temp_images_vz = image[:,:,:,:,3] * segmented
        # recombine the images
        image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)


        """


        



        

        if cnn_predictions:
            points_ = skeleton_points(segmented_original, dilation_k = 0)
            points_dilated = skeleton_points(segmented_original, dilation_k = 4,erosion_k = 4)
        else:
            points_ = skeleton_points(segmented_original, dilation_k = 0)
            points_dilated = skeleton_points(segmented_original, dilation_k = 2,erosion_k = 2)
        points = points_dilated.copy()


         
        if updated_ao:
            try:
            
                points_order_ascending_aorta = order_points(points[::-1], angle_threshold=3/2*np.pi/2.)
                logging.info('points order ascending aorta with angle threshold 3/2*np.pi/2.')
            except Exception as e:
                try:
                    points_order_ascending_aorta = order_points(points[::-1], angle_threshold=1/2*np.pi/2.)
                    logging.info('points order ascending aorta with angle threshold 1/2*np.pi/2.')
                except Exception as e:
                    points_order_ascending_aorta = np.array([0,0,0])
                    logging.info(f'An error occurred while processing {patient} ascending aorta: {e}')
            
            points = points[(points[:, 0] <= 90) & (points[:, 0] >= points_order_ascending_aorta[0][0]) & (points[:, 1] <= points_order_ascending_aorta[0][1])]

        else:
            points = points[np.where(points[:,1]<65)]
            points = points[np.where(points[:,0]<90)]
            points = points[points[:,0].argsort()[::-1]]

        
        #points = points[2:-2]
        #points = points[points[:,0].argsort()[::-1]]
        
        

        if skip:
            temp = []
            #for index, element in enumerate(points):
            for index, element in enumerate(points[2:]):
                if (index%2)==0:
                    temp.append(element)

            coords = np.array(temp)
        else:
            coords = np.array(points[2:])    

        #===========================================================================================
        # Parameters for the interpolation and creation of the files

        # We create Slices across time and channels in a double for loop
        temp_for_channel_stacking = []
        geometry_saved = False
        for channel in range(image.shape[4]):

            temp_for_time_stacking = []
            for t in range(image.shape[3]):
                if not geometry_saved:
                    save_for_backtransformation = True
                else:
                    save_for_backtransformation = False
                slice_dict = interpolate_and_slice(image[:,:,:,t,channel], coords, common_image_shape, smoothness=smoothness)
                if save_for_backtransformation:
                    geometry_saved = True  
                    geometry_dict = slice_dict['geometry_dict']
                straightened = slice_dict['straightened']

                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        straightened = np.stack(temp_for_channel_stacking, axis=-1)

        if suffix.__contains__('_without_rotation'):
            logging.info('Not rotating the vectors')
            straightened_rotated_vectors = straightened.copy()
            
            # We still want to save the rotation matrix that would be used to straighten the vectors
            for z_ in range(straightened.shape[2]):
                for t_ in range(straightened.shape[3]):
                    rotation_matrix_not_inverted = np.array(slice_dict['geometry_dict'][f'slice_{z_}']['transform'].GetMatrix()).reshape(3,3)
                # Populate the rotation matrix dataset
                dataset['rotation_matrix'][i*end_shape[2] + z_,:,:] = rotation_matrix_not_inverted
        else:
            logging.info('Rotating the vectors')
            # We need to rotate the vectors as well loop through slices
            straightened_rotated_vectors = straightened.copy()
            for z_ in range(straightened.shape[2]):
                for t_ in range(straightened.shape[3]):
                    rotation_matrix = np.array(slice_dict['geometry_dict'][f'slice_{z_}']['transform'].GetInverse().GetMatrix()).reshape(3,3)
                    rotation_matrix_not_inverted = np.array(slice_dict['geometry_dict'][f'slice_{z_}']['transform'].GetMatrix()).reshape(3,3)
                    straightened_rotated_vectors[:,:,z_,t_,1:] = rotate_vectors(straightened[:,:,z_,t_,1:], rotation_matrix)
                # Populate the rotation matrix dataset
                dataset['rotation_matrix'][i*end_shape[2] + z_,:,:] = rotation_matrix_not_inverted
                    

        image_out = straightened_rotated_vectors
        # Save the geometries for backtransformation with name of patient
        np.save(savepath_geometry + '/' + patient.replace('seg_',''), geometry_dict)

        # make all images of the same shape
        logging.info("Image shape before cropping and padding:" + str(image_out.shape))
        image_out = crop_or_pad_normal_slices(image_out, end_shape)
        logging.info("Image shape after cropping and padding:" + str(image_out.shape))

        # move the z-axis to the front, as we want to stack the data along this axis
        image_out = np.moveaxis(image_out, 2, 0)

        # add the image to the hdf5 file
        dataset['sliced_images_%s' % train_test][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out
        # If test then add label depending on if anomalous or not, 1 or 0
        if train_test == 'test':
            dataset['labels_%s' % train_test][i*end_shape[2]:(i+1)*end_shape[2]] = label
            

        # increment the index being used to write in the hdf5 datasets
        i = i + 1
        
        

    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0

# ==========================================
# ==========================================
def load_masked_data_sliced(basepath,
              idx_start,
              idx_end,
              train_test,
              force_overwrite=False,
              load_anomalous=False,
              include_compressed_sensing=True,      
              only_compressed_sensing = False,        
              suffix ='',
              updated_ao = False,
              skip = True,
              smoothness = 10,
              unwrapped = False):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    savepath = sys_config.project_code_root + 'data'
    dataset_filepath = savepath + f'/{train_test}{suffix}_masked_sliced_images_from_' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        logging.info('This configuration has not yet been preprocessed.')
        logging.info('Preprocessing now...')
        # Name of file
        logging.info('Name of file: ' + dataset_filepath)
        prepare_and_write_masked_data_sliced_bern(basepath = basepath,
                               filepath_output = dataset_filepath,
                               idx_start = idx_start,
                               idx_end = idx_end,
                               train_test = train_test,
                               load_anomalous= load_anomalous,
                               include_compressed_sensing=include_compressed_sensing,
                               only_compressed_sensing=only_compressed_sensing,
                               suffix = suffix,
                                updated_ao = updated_ao,
                                skip = skip,
                                smoothness = smoothness,
                                unwrapped = unwrapped
                               )
    else:
        logging.info('Already preprocessed this configuration. Loading now...')
        logging.info('Name of file: ' + dataset_filepath)

    return h5py.File(dataset_filepath, 'r')

# ====================================================================================
# *** MASKED SLICED DATA END ****
#====================================================================================

# ====================================================================================
# *** FULL AORTA SLICED  ****
#====================================================================================
def prepare_and_write_sliced_data_full_aorta_bern(basepath,
                           filepath_output,
                           idx_start,
                            idx_end,
                           train_test,
                           stack_z):

    # ==========================================
    # Study the the variation in the sizes along various dimensions (using the function 'find_shapes'),
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # ==========================================
    common_image_shape = [36, 36, 256, 24, 4] # [x, y, z, t, num_channels]

    #network_common_image_shape = [144, 112, None, 24, 4] # [x, y, t, num_channels]

    end_shape = [32, 32, 256, 24, 4]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)
    
    # ==========================================
    # ==========================================
    savepath_geometry = sys_config.project_code_root + 'data' + '/geometry_for_backtransformation'
    make_dir_safely(savepath_geometry)
    hand_seg_path_controls = basepath + '/segmenter_rw_pw_hard/controls'
    hand_seg_path_patients = basepath + '/segmenter_rw_pw_hard/patients'
    list_hand_seg_images = os.listdir(hand_seg_path_controls) + os.listdir(hand_seg_path_patients)
    # Sort the list
    list_hand_seg_images.sort()

    if ['train', 'val'].__contains__(train_test):

        seg_path = basepath + '/final_segmentations/train_val'
        img_path = basepath + '/preprocessed/controls/numpy'
    elif train_test == 'test':
        # For the img_path we need to look into the patients folder or the controls folder, try both, see further down
        seg_path = basepath + '/final_segmentations/test'
        img_path = basepath + '/preprocessed/patients/numpy'
    else:
        raise ValueError('train_test must be either train, val or test')
    
    seg_path_files = os.listdir(seg_path)
    # Sort
    seg_path_files.sort()
    
    patients = seg_path_files[idx_start:idx_end]
    num_images_to_load = len(patients)
    


    if stack_z == True:
        # ==========================================
        # we will stack all images along their z-axis
        # --> the network will analyze (x,y,t) volumes, with z-samples being treated independently.
        # ==========================================
        images_dataset_shape = [end_shape[2]*num_images_to_load,
                                end_shape[0],
                                end_shape[1],
                                end_shape[3],
                                end_shape[4]]
    else:
        # ==========================================
        # If we are not stacking along z (the centerline of the cropped aorta),
        # we are stacking along y (so shape[1])
        # ==========================================
        images_dataset_shape = [end_shape[1]*num_images_to_load,
                                end_shape[0],
                                end_shape[2],
                                end_shape[3],
                                end_shape[4]]


    # ==========================================
    # create a hdf5 file
    # ==========================================
    dataset = {}
    hdf5_file = h5py.File(filepath_output, "w")

    # ==========================================
    # write each subject's image and label data in the hdf5 file
    # ==========================================
    if stack_z == True:
        dataset['sliced_images_%s' % train_test] = hdf5_file.create_dataset("sliced_images_%s" % train_test, images_dataset_shape, dtype='float32')
    else:
        dataset['straightened_images_%s' % train_test] = hdf5_file.create_dataset("straightened_images_%s" % train_test, images_dataset_shape, dtype='uint8')

    i = 0
    
    cnn_predictions = True
    # Filter the patients by name to get the ones from hand-segmented and cnn predictions
    for patient in patients: 

        logging.info('loading subject ' + str(i+1) + ' out of ' + str(num_images_to_load) + '...')
        logging.info('patient: ' + patient)
        # Check if hand or network segemented (slightly different kernel size on pre-processing)
        if patient in list_hand_seg_images:
            cnn_predictions = False

        # load the segmentation that was created with Nicolas's tool
        if ['train', 'val'].__contains__(train_test):
            image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        elif train_test == 'test':
            # We need to look into the patients folder or the controls folder, try both
            try:
                # With patients
                image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
            except:
                # With controls
                image = np.load(os.path.join(img_path.replace("patients", "controls"), patient.replace("seg_", "")))
        
        
        segmented = np.load(os.path.join(seg_path, patient))
        
        
        image = normalize_image(image)

        # Compute the centerline points of the skeleton
        if cnn_predictions:
            points_ = skeleton_points(segmented, dilation_k = 0)
            points_dilated = skeleton_points(segmented, dilation_k = 4,erosion_k = 4)
        else:
            points_ = skeleton_points(segmented, dilation_k = 0)
            points_dilated = skeleton_points(segmented, dilation_k = 2,erosion_k = 2)
        points = points_dilated.copy()


        """
        # Average the segmentation over time (the geometry should be the same over time)
        avg = np.average(segmented, axis = 3)

        # Compute the centerline points of the skeleton
        skeleton = skeletonize_3d(avg[:,:,:])

        # Get the points of the centerline as an array
        points = np.array(np.where(skeleton != 0)).transpose([1,0])
        """


        # Order the points
        points = order_points(points)

        temp = []
        for index, element in enumerate(points[2:]):
            if (index%2)==0:
                temp.append(element)


        coords = np.array(temp)
        

        #===========================================================================================
        # Parameters for the interpolation and creation of the files

        # We create Slices across time and channels in a double for loop
        temp_for_channel_stacking = []
        geometry_saved = False
        for channel in range(image.shape[4]):

            temp_for_time_stacking = []
            for t in range(image.shape[3]):
                if not geometry_saved:
                    save_for_backtransformation = True
                else:
                    save_for_backtransformation = False

                slice_dict = interpolate_and_slice(image[:,:,:,t,channel], coords, common_image_shape, smoothness=10)
                if save_for_backtransformation:
                    geometry_saved = True  
                    geometry_dict = slice_dict['geometry_dict']
                straightened = slice_dict['straightened']
                
                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        straightened = np.stack(temp_for_channel_stacking, axis=-1)
        # We need to rotate the vectors as well loop through slices
        straightened_rotated_vectors = straightened.copy()
        for z_ in range(straightened.shape[2]):
            for t_ in range(straightened.shape[3]):
                rotation_matrix = np.array(slice_dict['geometry_dict'][f'slice_{z_}']['transform'].GetInverse().GetMatrix()).reshape(3,3)
                straightened_rotated_vectors[:,:,z_,t_,1:] = rotate_vectors(straightened[:,:,z_,t_,1:], rotation_matrix)

        image_out = straightened_rotated_vectors

        
        # Save the geometries for backtransformation with name of patient
        np.save(savepath_geometry + '/' + patient.replace('seg',''), geometry_dict)


        # make all images of the same shape
        logging.info("Image shape before cropping and padding:" + str(image_out.shape))
        #image_out = crop_or_pad_Bern_all_slices(image_out, network_common_image_shape)
        image_out = crop_or_pad_normal_slices(image_out, end_shape)
        logging.info("Image shape after cropping and padding:" + str(image_out.shape))

        if stack_z == True:
            # move the z-axis to the front, as we want to stack the data along this axis
            image_out = np.moveaxis(image_out, 2, 0)

            # add the image to the hdf5 file
            dataset['sliced_images_%s' % train_test][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out

        else:
            # move the y-axis to the front, as we want to stack the data along this axis
            image_out = np.moveaxis(image_out, 1, 0)

            logging.info('After shuffling the axis' + str(image_out.shape))
            logging.info(str(np.max(image_out)))

            # add the image to the hdf5 file
            dataset['straightened_images_%s' % train_test][i*end_shape[1]:(i+1)*end_shape[1], :, :, :, :] = image_out

        # increment the index being used to write in the hdf5 datasets
        i = i + 1

    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0

# ==========================================
# ==========================================
def load_cropped_data_sliced_full_aorta(basepath,
              idx_start,
              idx_end,
              train_test,
              force_overwrite=False):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    savepath = sys_config.project_code_root + 'data'
    dataset_filepath = savepath + f'/{train_test}_sliced_images_full_aorta_from_' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        logging.info('This configuration has not yet been preprocessed.')
        logging.info('Preprocessing now...')
        prepare_and_write_sliced_data_full_aorta_bern(basepath = basepath,
                               filepath_output = dataset_filepath,
                               idx_start=idx_start,
                                 idx_end=idx_end,
                               train_test = train_test,
                               stack_z = True)
    else:
        logging.info('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')

# ====================================================================================
# *** FULL AORTA SLICED END ****
#====================================================================================

# ====================================================================================
# *** MASKED SLICED DATA FULL AORTA ****
#====================================================================================
def prepare_and_write_masked_data_sliced_full_aorta_bern(basepath,
                           filepath_output,
                           idx_start,
                           idx_end,
                           train_test,
                           load_anomalous=False):

    # ==========================================
    # Study the the variation in the sizes along various dimensions (using the function 'find_shapes'),
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # ==========================================
    common_image_shape = [36, 36, 256, 24, 4] # [x, y, z, t, num_channels]
    
    end_shape = [32, 32, 256, 24, 4]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)

    # ==========================================
    # ==========================================
    savepath_geometry = sys_config.project_code_root + 'data' + '/geometry_for_backtransformation'
    make_dir_safely(savepath_geometry)
    hand_seg_path_controls = basepath + '/segmenter_rw_pw_hard/controls'
    hand_seg_path_patients = basepath + '/segmenter_rw_pw_hard/patients'
    list_hand_seg_images = os.listdir(hand_seg_path_controls) + os.listdir(hand_seg_path_patients)
    
    list_hand_seg_images.sort()

    
    if ['train', 'val'].__contains__(train_test):

        seg_path = basepath + '/final_segmentations/train_val'
        img_path = basepath + '/preprocessed/controls/numpy'
    elif train_test == 'test':
        # For the img_path we need to look into the patients folder or the controls folder, try both, see further down
        seg_path = basepath + '/final_segmentations/test'
        img_path = basepath + '/preprocessed/patients/numpy'
    else:
        raise ValueError('train_test must be either train, val or test')
    
    seg_path_files = os.listdir(seg_path)
    # Sort
    seg_path_files.sort()
    
    patients = seg_path_files[idx_start:idx_end]
    num_images_to_load = len(patients)

    # ==========================================
    # we will stack all images along their z-axis
    # --> the network will analyze (x,y,t) volumes, with z-samples being treated independently.
    # ==========================================
    images_dataset_shape = [end_shape[2]*num_images_to_load,
                            end_shape[0],
                            end_shape[1],
                            end_shape[3],
                            end_shape[4]]

    # ==========================================
    # create a hdf5 file
    # ==========================================
    dataset = {}
    hdf5_file = h5py.File(filepath_output, "w")

    # ==========================================
    # write each subject's image and label data in the hdf5 file
    # ==========================================
    dataset['sliced_images_%s' % train_test] = hdf5_file.create_dataset("sliced_images_%s" % train_test, images_dataset_shape, dtype='float32')
    # If test then add label depending on if anomalous or not, 1 or 0
    if train_test == 'test':
        dataset['labels_%s' % train_test] = hdf5_file.create_dataset("labels_%s" % train_test, (end_shape[2]*num_images_to_load,), dtype='uint8')
        
    
    cnn_predictions = True
    
    i = 0
    for patient in patients: 
        
        #logging.info('loading subject ' + str(n-idx_start+1) + ' out of ' + str(num_images_to_load) + '...')
        logging.info('loading subject ' + str(i+1) + ' out of ' + str(num_images_to_load) + '...')

        logging.info('patient: ' + patient)

        # Check if hand or network segemented (slightly different kernel size on pre-processing)
        if patient in list_hand_seg_images:
            cnn_predictions = False
        
        if ['train', 'val'].__contains__(train_test):
            image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        elif train_test == 'test':
            # We need to look into the patients folder or the controls folder, try both
            try:
                # With patients
                image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
                label = np.ones(end_shape[2])
            except:
                # With controls
                image = np.load(os.path.join(img_path.replace("patients", "controls"), patient.replace("seg_", "")))
                label = np.zeros(end_shape[2])
        segmented_original = np.load(os.path.join(seg_path, patient))
        

        # Enlarge the segmentation slightly to be sure that there are no cutoffs of the aorta
        time_steps = segmented_original.shape[3]
        #segmented = dilation(segmented_original[:,:,:,3], cube(6))
        segmented = dilation(segmented_original[:,:,:,3], cube(3))

        temp_for_stack = [segmented for i in range(time_steps)]
        segmented = np.stack(temp_for_stack, axis=3)

        # normalize image to -1 to 1
        image = normalize_image_new(image)
        seg_shape = list(segmented.shape)
        seg_shape.append(image.shape[-1])
        image = crop_or_pad_Bern_slices(image, seg_shape)


        temp_images_intensity = image[:,:,:,:,0] * segmented # change these back if it works
        temp_images_vx = image[:,:,:,:,1] * segmented
        temp_images_vy = image[:,:,:,:,2] * segmented
        temp_images_vz = image[:,:,:,:,3] * segmented

        # recombine the images
        image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)
        

        if cnn_predictions:
            points_ = skeleton_points(segmented_original, dilation_k = 0)
            points_dilated = skeleton_points(segmented_original, dilation_k = 4,erosion_k = 4)
        else:
            points_ = skeleton_points(segmented_original, dilation_k = 0)
            points_dilated = skeleton_points(segmented_original, dilation_k = 2,erosion_k = 2)
        points = points_dilated.copy()

        """

        # Average the segmentation over time (the geometry should be the same over time)
        avg = np.average(segmented_original, axis = 3)

        # Compute the centerline points of the skeleton
        skeleton = skeletonize_3d(avg[:,:,:])

        # Get the points of the centerline as an array
        points = np.array(np.where(skeleton != 0)).transpose([1,0])
        """

        points = order_points(points)

        temp = []
        for index, element in enumerate(points[2:]):
            if (index%2)==0:
                temp.append(element)

        coords = np.array(temp)
        

        #===========================================================================================
        # Parameters for the interpolation and creation of the files

        # We create Slices across time and channels in a double for loop
        temp_for_channel_stacking = []
        geometry_saved = False
        for channel in range(image.shape[4]):

            temp_for_time_stacking = []
            for t in range(image.shape[3]):
                if not geometry_saved:
                    save_for_backtransformation = True
                else:
                    save_for_backtransformation = False
                    
                slice_dict = interpolate_and_slice(image[:,:,:,t,channel], coords, common_image_shape,save_for_backtransformation, smoothness=10)
                if save_for_backtransformation:
                    geometry_saved = True  
                    geometry_dict = slice_dict['geometry_dict']
                straightened = slice_dict['straightened']
                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        straightened = np.stack(temp_for_channel_stacking, axis=-1)
        # We need to rotate the vectors as well loop through slices
        straightened_rotated_vectors = straightened.copy()
        for z_ in range(straightened.shape[2]):
            for t_ in range(straightened.shape[3]):
                rotation_matrix = np.array(slice_dict['geometry_dict'][f'slice_{z_}']['transform'].GetInverse().GetMatrix()).reshape(3,3)
                straightened_rotated_vectors[:,:,z_,t_,1:] = rotate_vectors(straightened[:,:,z_,t_,1:], rotation_matrix)

        image_out = straightened_rotated_vectors
        

        # Save the geometries for backtransformation with name of patient
        np.save(savepath_geometry + '/' + patient.replace('seg',''), geometry_dict)
            



        # make all images of the same shape
        logging.info("Image shape before cropping and padding:" + str(image_out.shape))
        image_out = crop_or_pad_normal_slices(image_out, end_shape)
        logging.info("Image shape after cropping and padding:" + str(image_out.shape))

        # move the z-axis to the front, as we want to stack the data along this axis
        image_out = np.moveaxis(image_out, 2, 0)

        # add the image to the hdf5 file
        dataset['sliced_images_%s' % train_test][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out
        if train_test == 'test':
            dataset['labels_%s' % train_test][i*end_shape[2]:(i+1)*end_shape[2]] = label

        # increment the index being used to write in the hdf5 datasets
        i = i + 1

    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0

# ==========================================
# ==========================================
def load_masked_data_sliced_full_aorta(basepath,
              idx_start,
              idx_end,
              train_test,
              force_overwrite=False,
              load_anomalous=False):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    savepath = sys_config.project_code_root + 'data'
    dataset_filepath = savepath + f'/{train_test}_masked_sliced_images_full_aorta_from_' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        logging.info('This configuration has not yet been preprocessed.')
        logging.info('Preprocessing now...')
        prepare_and_write_masked_data_sliced_full_aorta_bern(basepath = basepath,
                               filepath_output = dataset_filepath,
                               idx_start = idx_start,
                               idx_end = idx_end,
                               train_test = train_test,
                               load_anomalous= load_anomalous)
    else:
        logging.info('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')

# ====================================================================================
# *** MASKED SLICED DATA FULL AORTA END ****
#====================================================================================


if __name__ == '__main__':

    basepath =  sys_config.project_data_root #"/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady"
    savepath = sys_config.project_code_root + "data"
    make_dir_safely(savepath)

    # Make sure that any patient from the training/validation set is not in the test set
    verify_leakage()
    #masked_sliced_data_train_all = load_masked_data_sliced(basepath, idx_start=0, idx_end=41, train_test='train', suffix = '_without_rotation_with_cs_skip_updated_ao_S10', include_compressed_sensing = True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_validation_all = load_masked_data_sliced(basepath, idx_start=41, idx_end=51, train_test='val', suffix = '_without_rotation_with_cs_skip_updated_ao_S10', include_compressed_sensing = True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_test_all = load_masked_data_sliced(basepath, idx_start=0, idx_end=54, train_test='test', suffix = '_without_rotation_with_cs_skip_updated_ao_S10', include_compressed_sensing = True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    
    #masked_sliced_data_train_all = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='train', suffix = '_with_rotation_with_cs_skip_updated_ao_S10', include_compressed_sensing = True, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_validation_all = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='val', suffix = '_with_rotation_with_cs_skip_updated_ao_S10', include_compressed_sensing = True, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_test_all = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='test', suffix = '_with_rotation_with_cs_skip_updated_ao_S10', include_compressed_sensing = True, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10)

    #masked_sliced_data_train = load_masked_data_sliced(basepath, idx_start=0, idx_end=35, train_test='train', suffix = '_with_rotation_without_cs_skip_updated_ao_S10', include_compressed_sensing = False, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_train = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='train', suffix = '_without_rotation_without_cs_skip_updated_ao_S10', include_compressed_sensing = False, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_validation = load_masked_data_sliced(basepath, idx_start=35, idx_end=42, train_test='val', suffix = '_with_rotation_without_cs_skip_updated_ao_S10', include_compressed_sensing = False, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_validation = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='val', suffix = '_without_rotation_without_cs_skip_updated_ao_S10', include_compressed_sensing = False, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_test = load_masked_data_sliced(basepath, idx_start=0, idx_end=34, train_test='test', suffix = '_with_rotation_without_cs_skip_updated_ao_S10', include_compressed_sensing = False, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_test = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='test', suffix = '_without_rotation_without_cs_skip_updated_ao_S10', include_compressed_sensing = False, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10)
    
    
    #masked_sliced_data_train_only_cs = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='train', suffix = '_without_rotation_only_cs_skip_updated_ao_S10_centered_norm', include_compressed_sensing = True, only_compressed_sensing=True, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10, unwrapped = False)
    #masked_sliced_data_validation_only_cs = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='val', suffix = '_without_rotation_only_cs_skip_updated_ao_S10_centered_norm', include_compressed_sensing = True, only_compressed_sensing=True, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10, unwrapped = False)
    #masked_sliced_data_test_only_cs = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='test', suffix = '_without_rotation_only_cs_skip_updated_ao_S10_centered_norm', include_compressed_sensing = True, only_compressed_sensing=True, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10, unwrapped = False)
    #
    #masked_sliced_data_train_only_cs_unwrapped = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='train', suffix = '_without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped', include_compressed_sensing = True, only_compressed_sensing=True, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10, unwrapped = True)
    #masked_sliced_data_validation_only_cs_unwrapped = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='val', suffix = '_without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped', include_compressed_sensing = True, only_compressed_sensing=True, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10, unwrapped = True)
    #masked_sliced_data_test_only_cs_unwrapped = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='test', suffix = '_without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped', include_compressed_sensing = True, only_compressed_sensing=True, force_overwrite= True, skip = True, updated_ao = True, smoothness = 10, unwrapped = True)
    





    #masked_sliced_data_train = load_masked_data_sliced(basepath, idx_start=0, idx_end=35, train_test='train', suffix = '_with_rotation_without_cs_skip_updated_ao_S10', include_compressed_sensing = False, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_train_all = load_masked_data_sliced(basepath, idx_start=0, idx_end=41, train_test='train', suffix = '_without_rotation_with_cs_skip_updated_ao_S10', include_compressed_sensing = True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_train_only_cs = load_masked_data_sliced(basepath, idx_start=0, idx_end=10, train_test='train', suffix = '_without_rotation_only_cs_skip_updated_ao_S10_centered_norm', include_compressed_sensing = True, only_compressed_sensing=True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_train_only_cs_unwrapped = load_masked_data_sliced(basepath, idx_start=0, idx_end=10, train_test='train', suffix = '_without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped', include_compressed_sensing = True, only_compressed_sensing=True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10, unwrapped=True)
    #masked_sliced_data_validation = load_masked_data_sliced(basepath, idx_start=35, idx_end=42, train_test='val', suffix = '_with_rotation_without_cs_skip_updated_ao_S10', include_compressed_sensing = False, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_validation_all = load_masked_data_sliced(basepath, idx_start=41, idx_end=51, train_test='val', suffix = '_without_rotation_with_cs_skip_updated_ao_S10', include_compressed_sensing = True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_validation_only_cs = load_masked_data_sliced(basepath, idx_start=10, idx_end=14, train_test='val', suffix = '_without_rotation_only_cs_skip_updated_ao_S10_centered_norm', include_compressed_sensing = True, only_compressed_sensing=True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_validation_only_cs_unwrapped = load_masked_data_sliced(basepath, idx_start=10, idx_end=14, train_test='val', suffix = '_without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped', include_compressed_sensing = True, only_compressed_sensing=True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10, unwrapped=True)
    #masked_sliced_data_test = load_masked_data_sliced(basepath, idx_start=0, idx_end=34, train_test='test', suffix = '_with_rotation_without_cs_skip_updated_ao_S10', include_compressed_sensing = False, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_test_all = load_masked_data_sliced(basepath, idx_start=0, idx_end=54, train_test='test', suffix = '_without_rotation_with_cs_skip_updated_ao_S10', include_compressed_sensing = True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_test_only_cs = load_masked_data_sliced(basepath, idx_start=0, idx_end=17, train_test='test', suffix = '_without_rotation_only_cs_skip_updated_ao_S10_centered_norm', include_compressed_sensing = True, only_compressed_sensing=True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_test_only_cs_unwrapped = load_masked_data_sliced(basepath, idx_start=0, idx_end=17, train_test='test', suffix = '_without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped', include_compressed_sensing = True, only_compressed_sensing=True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10, unwrapped=True)
    #masked_sliced_data_train = load_masked_data_sliced(basepath, idx_start=0, idx_end=35, train_test='train', suffix = '_without_rotation_without_cs_skip_updated_ao_S10', include_compressed_sensing = False, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_validation = load_masked_data_sliced(basepath, idx_start=35, idx_end=42, train_test='val', suffix = '_without_rotation_without_cs_skip_updated_ao_S10', include_compressed_sensing = False, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)    
    #masked_sliced_data_test = load_masked_data_sliced(basepath, idx_start=0, idx_end=34, train_test='test', suffix = '_without_rotation_without_cs_skip_updated_ao_S10', include_compressed_sensing = False, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    #masked_sliced_data_validation = load_masked_data_sliced(basepath, idx_start=35, idx_end=42, train_test='val', suffix = '_with_rotation_without_cs', include_compressed_sensing = False, force_overwrite= False)    
    #masked_sliced_data_train = load_masked_data_sliced(basepath, idx_start=0, idx_end=48, train_test='train', suffix = '_without_rotation', include_compressed_sensing = True, force_overwrite=False)
    #masked_sliced_data_validation = load_masked_data_sliced(basepath, idx_start=48, idx_end=58, train_test='val', suffix = '_without_rotation', include_compressed_sensing = True, force_overwrite= True)
    #masked_sliced_data_validation = load_masked_data_sliced(basepath, idx_start=48, idx_end=58, train_test='val', suffix = '_with_rotation', include_compressed_sensing = True, force_overwrite= True)    
    #masked_sliced_data_test = load_masked_data_sliced(basepath, idx_start=0, idx_end=46, train_test='test', suffix = '_without_rotation', include_compressed_sensing = True, force_overwrite= False)
    #masked_sliced_data_test_cs = load_masked_data_sliced(basepath, idx_start=0, idx_end=8, train_test='test', suffix='_compressed_sensing')    
    