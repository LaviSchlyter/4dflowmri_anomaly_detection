"""
Preprocessing and Backtransformation of Anomaly Scores

This script contains multiple functions for preprocessing and backtransforming anomaly scores from 2D slice space
and time to the original 3D space and time. It includes gradient matching, masked data preparation, and slicing 
for both the ascending aorta and the full aorta.

Key Functions:
- `prepare_and_write_gradient_matching_data_bern`: Creates gradients for alignment with radiologist predictions.
- `prepare_and_write_masked_data_bern`: Prepares masked data. (Used during the experimental phase)
- `prepare_and_write_sliced_data_bern`: Prepares cropped and straightened aorta data slices. (Used during the experimental phase)
- `prepare_and_write_masked_data_sliced_bern`: Prepares masked and sliced data for ascending aorta. (Final version)
- `prepare_and_write_sliced_data_full_aorta_bern`: Prepares sliced data for the full aorta. (Used during the experimental phase)
- `prepare_and_write_masked_data_sliced_full_aorta_bern`: Prepares masked and sliced data for the full aorta. 

"""
import os
import h5py
import numpy as np
import sys
from skimage.morphology import dilation, cube
from skimage.restoration import unwrap_phase
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/helpers/')

from utils import (
    verify_leakage,
    crop_or_pad_Bern_slices,
    normalize_image,
    make_dir_safely,
    crop_or_pad_normal_slices,
    extract_slice_from_sitk_image,
    rotate_vectors,
    interpolate_and_slice,
    skeleton_points,
    order_points
)

sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/')


import config.system as sys_config

# ====================================================================================
# *** GRADIENT MATCHING TO ALIGN WITH RADIOLOGIST PREDICTION ****
#====================================================================================

def prepare_and_write_gradient_matching_data_bern(basepath, suffix =''):
    """
    This function will for each subject in the test set, create a gradient from foot to head, left to right and back to front.
    It will then use the segmentation of the subject and do the slicing procedure. This will later be used on the predicted anomaly score slices in order to make sure
    that we look at the correct anterior-posterior and left-right directions matching the validation from the radiologist.
    """

    common_image_shape = [36, 36, 64, 24, 4] # [x, y, z, t, num_channels]
    end_shape = [32, 32, 64, 24, 4]

    segmentation_path = basepath + '/final_segmentations/test_balanced'

    # sort the files
    seg_path_files = os.listdir(segmentation_path)
    seg_path_files.sort()

    save_gradient_matching_path = sys_config.project_code_root + 'data' + f'/gradient_matching{suffix}'
    make_dir_safely(save_gradient_matching_path)

    # Unfortunetaly it would have been good to do them directly if hand segmented or not, but too late now. TODO: 21.05.24
    cnn_predictions = True
    for patient in seg_path_files:
        # If already processed, skip
        if os.path.exists(save_gradient_matching_path + '/' + patient.replace('seg_','')):
            continue

        logging.info('loading subject ' + patient + '...')
        segmentation = np.load(os.path.join(segmentation_path, patient))

        # Create an image with the gradients
        sizex, sizey, sizez, sizet= segmentation.shape
        gradient_top_bottom = np.linspace(0, 1, sizex)[:, None, None, None]
        gradient_front_back = np.linspace(0, 1, sizey)[None, :, None, None]
        gradient_left_right = np.linspace(0, 1, sizez)[None, None, :, None]
        
        
        gradient_image = np.zeros((sizex, sizey, sizez, sizet, 4))

        # Assign the gradients to the corresponding channels
        gradient_image[..., 0] = gradient_top_bottom
        gradient_image[..., 1] = gradient_front_back
        gradient_image[..., 2] = gradient_left_right


        if suffix == '_seg':
            logging.info('Using the segmentation to create the gradient image')
            time_steps = segmentation.shape[3]
            segmented = dilation(segmentation[:,:,:,3], cube(3))
            # Enlarge the segmentation slightly to be sure that there are no cutoffs of the aorta
            

            temp_for_stack = [segmented for i in range(time_steps)]
            segmented = np.stack(temp_for_stack, axis=3)

            temp_images_intensity = gradient_image[:,:,:,:,0] * segmented 
            temp_images_vx = gradient_image[:,:,:,:,1] * segmented
            temp_images_vy = gradient_image[:,:,:,:,2] * segmented
            temp_images_vz = gradient_image[:,:,:,:,3] * segmented

            # recombine the images
            gradient_image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)



        if cnn_predictions:
            points_ = skeleton_points(segmentation, dilation_k = 0)
            points_dilated = skeleton_points(segmentation, dilation_k = 4,erosion_k = 4)
        else:
            points_ = skeleton_points(segmentation, dilation_k = 0)
            points_dilated = skeleton_points(segmentation, dilation_k = 2,erosion_k = 2)
        points = points_dilated.copy()
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

        temp = []
        for index, element in enumerate(points[2:]):
            if (index%2)==0:
                temp.append(element)

        coords = np.array(temp)

        temp_for_channel_stacking = []
        for channel in range(gradient_image.shape[4]):

            temp_for_time_stacking = []
            for t in range(gradient_image.shape[3]):
                slice_dict = interpolate_and_slice(gradient_image[:,:,:,t,channel], coords, common_image_shape, smoothness=10)
                straightened = slice_dict['straightened']
                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        gradient_image_out = np.stack(temp_for_channel_stacking, axis=-1)
        gradient_image_out = crop_or_pad_normal_slices(gradient_image_out, end_shape)

        # Save the gradient out image
        np.save(save_gradient_matching_path + '/' + patient.replace('seg_',''), gradient_image_out)
    return 0


#====================================================================================
# *** GRADIENT MATCHING TO ALIGN WITH RADIOLOGIST PREDICTION *** END
#====================================================================================

# ====================================================================================
# *** MASKED DATA **** - Used during experimental phase
#====================================================================================

def prepare_and_write_masked_data_bern(basepath,
                           filepath_output,
                            idx_start,
                            idx_end,
                           train_test,
                           suffix =''):

    # ==========================================
    # Study the the variation in the sizes along various dimensions. TODO: Needs update with the new data...
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # ==========================================
    common_image_shape = [144, 112, 40, 24, 4] # [x, y, z, t, num_channels]
    common_label_shape = [144, 112, 40, 24] # [x, y, z, t]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)

    # ==========================================
    # ==========================================
    if ['train', 'val'].__contains__(train_test):
        seg_path = basepath + f'/final_segmentations/train_val_balanced'
        img_path = basepath + f'/preprocessed/controls/numpy'
    elif train_test == 'test':
        # For the img_path we need to look into the patients folder or the controls folder, try both, see further down
        seg_path = basepath + f'/final_segmentations/test_balanced'
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
        image = normalize_image(image)
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
# CROPPED AND STRAIGHTENED AORTA DATA Z-SLICES - Used during experimental phase
#====================================================================================
def prepare_and_write_sliced_data_bern(basepath,
                           filepath_output,
                           idx_start,
                            idx_end,
                           train_test,
                           stack_z):

    # ==========================================
    # Study the the variation in the sizes along various dimensions. TODO: Needs update with the new data...
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

        seg_path = basepath + '/final_segmentations/train_val_balanced'
        img_path = basepath + '/preprocessed/controls/numpy'
    elif train_test == 'test':
        # For the img_path we need to look into the patients folder or the controls folder, try both, see further down
        seg_path = basepath + '/final_segmentations/test_balanced'
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
# *** MASKED SLICED DATA **** - Final version for ascending aorta
#====================================================================================

def find_and_load_image(patient, basepaths):
    """
    Attempts to find and load an image for a given patient from a list of base paths.

    Parameters:
    - patient (str): The patient identifier.
    - basepaths (list): List of paths to search for the patient image.

    Returns:
    - numpy.ndarray: Loaded image data.

    Raises:
    - FileNotFoundError: If the image file is not found in any of the provided paths.
    """
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
                           include_compressed_sensing=True,
                           only_compressed_sensing=False,
                           suffix ='',
                           updated_ao = False,
                           skip = True,
                           smoothness = 10,
                           unwrapped = False
                           ):
    """
    Prepares and writes masked data for a given range of subjects.

    Parameters:
    - basepath (str): Base path for the data.
    - filepath_output (str): Output file path for the HDF5 dataset.
    - idx_start (int): Start index for subject processing.
    - idx_end (int): End index for subject processing.
    - train_test (str): Indicates whether the data is for training, validation, or testing.
    - include_compressed_sensing (bool): Whether to include compressed sensing data.
    - only_compressed_sensing (bool): Whether to use only compressed sensing data.
    - suffix (str): Suffix for the file naming.
    - updated_ao (bool): Whether to use the updated version to order the aorta points.
    - skip (bool): Whether to skip every other point on the centerline.
    - smoothness (int): Smoothness parameter for interpolation.
    - unwrapped (bool): Whether to unwrap the phase. TODO: Does not work as expected. 30.05.24
    """
    
    # TODO: Could be updated since we have received new data  30.05.24
    common_image_shape = [36, 36, 64, 24, 4] # [x, y, z, t, num_channels]
    end_shape = [32, 32, 64, 24, 4] # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16

    logging.info(f"include_compressed_sensing: {include_compressed_sensing}")
    logging.info(f"updated_ao: {updated_ao}")
    logging.info(f"skip: {skip}")
    logging.info(f"smoothness: {smoothness}")
    logging.info(f"only_compressed_sensing: {only_compressed_sensing}")


    savepath_geometry = sys_config.project_code_root + 'data' + f'/geometry_for_backtransformation'
    make_dir_safely(savepath_geometry)

    hand_seg_path_controls = [
        os.path.join(basepath, 'segmenter_rw_pw_hard', 'controls'),
        os.path.join(basepath, 'segmenter_rw_pw_hard', 'controls_compressed_sensing')
    ]
    hand_seg_path_patients = [
        os.path.join(basepath, 'segmenter_rw_pw_hard', 'patients'),
        os.path.join(basepath, 'segmenter_rw_pw_hard', 'patients_compressed_sensing')
    ]
    list_hand_seg_images = []
    for path in hand_seg_path_controls + hand_seg_path_patients:
        list_hand_seg_images.extend(os.listdir(path))
    list_hand_seg_images.sort()

    # This code is left there such that future models could be trained solely 
    # on compressed sensing or sequential as more data is available

    # Update here because you'll have a different folder for either three setups
    if train_test in ['train', 'val'] and not only_compressed_sensing:
        seg_path = basepath + '/final_segmentations/train_val_balanced'
        img_path = basepath + '/preprocessed/controls/numpy'
        img_path_compressed = basepath + '/preprocessed/controls/numpy_compressed_sensing'

    elif train_test == 'test' and not only_compressed_sensing:
        seg_path = basepath + '/final_segmentations/test_balanced'
        img_path = basepath + '/preprocessed/patients/numpy'
        img_path_compressed = basepath + '/preprocessed/patients/numpy_compressed_sensing'
        img_paths = [img_path, img_path_compressed]

    elif train_test in ['train', 'val'] and only_compressed_sensing:
        seg_path = basepath + '/final_segmentations/train_val_compressed_sensing_balanced'
        
    elif train_test == 'test' and only_compressed_sensing:
        seg_path = basepath + f'/final_segmentations/test_compressed_sensing_balanced'
        
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
    seg_path_files = sorted(os.listdir(seg_path))
    
    
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
        # TODO: 30.05.24 - Nice update. For future versions uncomment
        #if patient in list_hand_seg_images:
        #    cnn_predictions = False
        
        if train_test in ['train', 'val']:
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
        segmented = dilation(segmented_original[:,:,:,3], cube(3))
        temp_for_stack = [segmented for i in range(time_steps)]
        segmented = np.stack(temp_for_stack, axis=3)
        image = image.astype(float)
        
        if unwrapped:
            #TODO: 30.05.24 - This did not work as expected. Future work needed
            logging.info('Unwrapping the phase')

            # Create a masked array where the background is masked
            segmented_bool = abs(segmented - 1).astype(bool)
            segmented_bool = np.expand_dims(segmented_bool, axis=4)
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

            if suffix.__contains__('centered_norm'):
                # Substract 2048 to center the values around 0 for the velocity channels
                image[...,1:] -= 2048.0
            
            # Add the first channel (magnitude) back to the unwrapped image
            unwrapped_image[...,0] = masked_image[...,0]

            #Fill masked values with 0
            image = np.ma.filled(unwrapped_image, 0.0)
        else:
            if suffix.__contains__('centered_norm'):
                
                # Subtract 2048 to center the values around 0 for the velocity channels
                image[...,1:] -= 2048.0
                temp_images_intensity = image[:,:,:,:,0] * segmented # change these back if it works
                temp_images_vx = image[:,:,:,:,1] * segmented
                temp_images_vy = image[:,:,:,:,2] * segmented
                temp_images_vz = image[:,:,:,:,3] * segmented
                image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)

        if suffix.__contains__('centered_norm'):
            logging.info('Centered and normalized image')
        # Normalize the image
            image = normalize_image(image, with_set_range=[-2048, 2048])

        else:
            # In this case we need to set the background to 0 after because background will be scaled to -1
            
            image = normalize_image(image)
            temp_images_intensity = image[:,:,:,:,0] * segmented # change these back if it works
            temp_images_vx = image[:,:,:,:,1] * segmented
            temp_images_vy = image[:,:,:,:,2] * segmented
            temp_images_vz = image[:,:,:,:,3] * segmented
            image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)
        

        # Extract the centerline points using skeletonization
        if cnn_predictions:
            points_ = skeleton_points(segmented_original, dilation_k = 0)
            points_dilated = skeleton_points(segmented_original, dilation_k = 4,erosion_k = 4)
        else:
            points_ = skeleton_points(segmented_original, dilation_k = 0)
            points_dilated = skeleton_points(segmented_original, dilation_k = 2,erosion_k = 2)
        points = points_dilated.copy()


         
        if updated_ao:
        # TODO: 30.05.24 - Future work on more robust ordering of the aorta points
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

        if '_without_rotation' in suffix:
            logging.info('Not rotating the vectors')
            straightened_rotated_vectors = straightened.copy()
            
            # We still want to save the rotation matrix that would be used to straighten the vectors
            for z_ in range(straightened.shape[2]):
                for t_ in range(straightened.shape[3]):
                    rotation_matrix_not_inverted = np.array(slice_dict['geometry_dict'][f'slice_{z_}']['transform'].GetMatrix()).reshape(3,3)
                # Populate the rotation matrix dataset
                dataset['rotation_matrix'][i*end_shape[2] + z_,:,:] = rotation_matrix_not_inverted
        elif '_with_rotation' in suffix:
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
        else:
            raise ValueError('suffix must contain _without_rotation or _with_rotation')
                    

        image_out = straightened_rotated_vectors
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
def load_masked_data_sliced(
        basepath,
        idx_start,
        idx_end,
        train_test,
        force_overwrite=False,
        load_anomalous=False,
        include_compressed_sensing=True,
        only_compressed_sensing=False,
        suffix='',
        updated_ao=False,
        skip=True,
        smoothness=10,
        unwrapped=False
):
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
    # Study the the variation in the sizes along various dimensions. TODO: Needs update with the new data...
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
    # Sort the list
    list_hand_seg_images.sort()

    if ['train', 'val'].__contains__(train_test):

        seg_path = basepath + '/final_segmentations/train_val_balanced'
        img_path = basepath + '/preprocessed/controls/numpy'
    elif train_test == 'test':
        # For the img_path we need to look into the patients folder or the controls folder, try both, see further down
        seg_path = basepath + '/final_segmentations/test_balanced'
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
        if train_test in ['train', 'val']:
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
                           train_test):

    # ==========================================
    # Study the the variation in the sizes along various dimensions. TODO: Needs update with the new data...
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

    
    if train_test in ['train', 'val']:

        seg_path = basepath + '/final_segmentations/train_val_balanced'
        img_path = basepath + '/preprocessed/controls/numpy'
    elif train_test == 'test':
        # For the img_path we need to look into the patients folder or the controls folder, try both, see further down
        seg_path = basepath + '/final_segmentations/test_balanced'
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
        
        if train_test in ['train', 'val']:
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
        image = normalize_image(image)
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
              force_overwrite=False
              ):

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
                               train_test = train_test)
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

    
    prepare_and_write_gradient_matching_data_bern(basepath = basepath, suffix = '')

    # Make sure that any patient from the training/validation set is not in the test set
    verify_leakage()

    masked_sliced_data_train_all = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='train', suffix = '_without_rotation_with_cs_skip_updated_ao_S10_balanced', include_compressed_sensing = True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    masked_sliced_data_validation_all = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='val', suffix = '_without_rotation_with_cs_skip_updated_ao_S10_balanced', include_compressed_sensing = True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    masked_sliced_data_test_all = load_masked_data_sliced(basepath, idx_start=0, idx_end=1, train_test='test', suffix = '_without_rotation_with_cs_skip_updated_ao_S10_balanced', include_compressed_sensing = True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)

    masked_sliced_data_train_all = load_masked_data_sliced(basepath, idx_start=0, idx_end=41, train_test='train', suffix = '_without_rotation_with_cs_skip_updated_ao_S10_balanced', include_compressed_sensing = True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    masked_sliced_data_validation_all = load_masked_data_sliced(basepath, idx_start=41, idx_end=51, train_test='val', suffix = '_without_rotation_with_cs_skip_updated_ao_S10_balanced', include_compressed_sensing = True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    masked_sliced_data_test_all = load_masked_data_sliced(basepath, idx_start=0, idx_end=54, train_test='test', suffix = '_without_rotation_with_cs_skip_updated_ao_S10_balanced', include_compressed_sensing = True, force_overwrite= False, skip = True, updated_ao = True, smoothness = 10)
    
