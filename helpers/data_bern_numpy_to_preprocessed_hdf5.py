import os
import h5py
import numpy as np
import sys
from skimage.morphology import skeletonize_3d, dilation, cube, binary_erosion
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/helpers/')

from utils import crop_or_pad_Bern_slices, normalize_image, normalize_image_new, make_dir_safely, crop_or_pad_4dvol, crop_or_pad_normal_slices
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
    return resampled_sitk_image

def interpolate_and_slice(image,
                          coords,
                          size, smoothness=200):


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
        # I define the x'-vector as the projection of the y-vector onto the plane perpendicular to the spline
        xs = (direc - np.dot(direc, Zs[i]) / (np.power(np.linalg.norm(Zs[i]), 2)) * Zs[i])
        sitk_slice = extract_slice_from_sitk_image(sitk_image, points[i], Zs[i], xs, list(size[:2]) + [1], fill_value=0)
        np_image = sitk.GetArrayFromImage(sitk_slice).transpose(2, 1, 0)
        slices.append(np_image)

    # stick slices together
    return np.concatenate(slices, axis=2)

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
def order_points(candidate_points):
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
            if ang > (np.pi/2.):
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
                           train_test):

    # ==========================================
    # Study the the variation in the sizes along various dimensions (using the function 'find_shapes'),
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # ==========================================
    # For Bern the max sizes are:
    # x: 144, y: 112, z: 64, t: 33 (but because of network we keep 48)
    common_image_shape = [144, 112, 40, 48, 4] # [x, y, z, t, num_channels]
    common_label_shape = [144, 112, 40, 48] # [x, y, z, t]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)

    # ==========================================
    # ==========================================
    if ['train', 'val'].__contains__(train_test):

        seg_path = basepath + '/final_segmentations/controls'
        img_path = basepath + '/preprocessed/controls/numpy'
    elif train_test == 'test':
        seg_path = basepath + '/final_segmentations/patients'
        img_path = basepath + '/preprocessed/patients/numpy'
    else:
        raise ValueError('train_test must be either train, val or test')
    
    patients = os.listdir(seg_path)[idx_start:idx_end]
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
        
        print('loading subject ' + str(i+1) + ' out of ' + str(num_images_to_load)  + '...')
        print('patient', patient)
        
        image_data = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        
        # normalize the image
        image_data = normalize_image_new(image_data)
        print('Shape of image before network resizing',image_data.shape)
        # The images need to be sized as in the input of network
        
        
        # make all images of the same shape
        image_data = crop_or_pad_Bern_slices(image_data, common_image_shape)
        print('Shape of image after network resizing',image_data.shape)
        # move the z-axis to the front, as we want to concantenate data along this axis
        image_data = np.moveaxis(image_data, 2, 0)

        label_data = np.load(os.path.join(seg_path, patient))
        # make all images of the same shape
        label_data = crop_or_pad_Bern_slices(label_data, common_label_shape)
        # move the z-axis to the front, as we want to concantenate data along this axis
        label_data = np.moveaxis(label_data, 2, 0)
        # cast labels as uints
        label_data = label_data.astype(np.uint8)


        temp_images_intensity = image_data[:,:,:,:,0] * label_data
        temp_images_vx = image_data[:,:,:,:,1] * label_data
        temp_images_vy = image_data[:,:,:,:,2] * label_data
        temp_images_vz = image_data[:,:,:,:,3] * label_data

        #recombine the images
        image_data = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)

        # add the image to the hdf5 file
        dataset['masked_images_%s' % train_test][i*common_image_shape[2]:(i+1)*common_image_shape[2], :, :, :, :] = image_data

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
              force_overwrite=False):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    savepath = sys_config.project_code_root + 'data'
    dataset_filepath = savepath + f'/{train_test}_masked_images_from_' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_masked_data_bern(basepath = basepath,
                               filepath_output = dataset_filepath,
                               idx_start = idx_start,
                                 idx_end = idx_end,
                               train_test = train_test)
    else:
        print('Already preprocessed this configuration. Loading now...')

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
    common_image_shape = [36, 36, 64, 48, 4] # [x, y, z, t, num_channels]

    #network_common_image_shape = [144, 112, None, 48, 4] # [x, y, t, num_channels]

    end_shape = [32, 32, 64, 48, 4]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)
    
    # ==========================================
    # ==========================================
    
    hand_seg_path_controls = basepath + '/segmenter_rw_pw_hard/controls'
    hand_seg_path_patients = basepath + '/segmenter_rw_pw_hard/patients'
    list_hand_seg_images = os.listdir(hand_seg_path_controls) + os.listdir(hand_seg_path_patients)
    if ['train', 'val'].__contains__(train_test):

        seg_path = basepath + '/final_segmentations/controls'
        img_path = basepath + '/preprocessed/controls/numpy'
    elif train_test == 'test':
        seg_path = basepath + '/final_segmentations/patients'
        img_path = basepath + '/preprocessed/patients/numpy'
    else:
        raise ValueError('train_test must be either train, val or test')
    
    patients = os.listdir(seg_path)[idx_start:idx_end]
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

        print('loading subject ' + str(i+1) + ' out of ' + str(num_images_to_load) + '...')
        print('patient: ' + patient)

        # Check if hand or network segemented (slightly different kernel size on pre-processing)
        if patient in list_hand_seg_images:
            cnn_predictions = False

        # load the segmentation that was created with Nicolas's tool
        image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        
        
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

        # Limit to sectors where ascending aorta is located
        points = points[np.where(points[:,1]<60)]
        points = points[np.where(points[:,0]<90)]

        # Order the points in ascending order with x
        points = points[points[:,0].argsort()[::-1]]

        temp = []
        for index, element in enumerate(points[5:]):
            if (index%5)==0:
                temp.append(element)


        coords = np.array(temp)
        

        #===========================================================================================
        # Parameters for the interpolation and creation of the files

        # We create Slices across time and channels in a double for loop
        temp_for_channel_stacking = []
        for channel in range(image.shape[4]):

            temp_for_time_stacking = []
            for t in range(image.shape[3]):
                straightened = interpolate_and_slice(image[:,:,:,t,channel], coords, common_image_shape)
                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        straightened = np.stack(temp_for_channel_stacking, axis=-1)

        image_out  = straightened

        # make all images of the same shape
        print("Image shape before cropping and padding:" + str(image_out.shape))
        #image_out = crop_or_pad_Bern_all_slices(image_out, network_common_image_shape)
        image_out = crop_or_pad_normal_slices(image_out, end_shape)
        print("Image shape after cropping and padding:" + str(image_out.shape))

        if stack_z == True:
            # move the z-axis to the front, as we want to stack the data along this axis
            image_out = np.moveaxis(image_out, 2, 0)

            # add the image to the hdf5 file
            dataset['sliced_images_%s' % train_test][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out

        else:
            # move the y-axis to the front, as we want to stack the data along this axis
            image_out = np.moveaxis(image_out, 1, 0)

            print('After shuffling the axis' + str(image_out.shape))
            print(str(np.max(image_out)))

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
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_sliced_data_bern(basepath = basepath,
                               filepath_output = dataset_filepath,
                               idx_start=idx_start,
                                 idx_end=idx_end,
                               train_test = train_test,
                               stack_z = True)
    else:
        print('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')
# ====================================================================================
# *** CROPPED AND STRAIGHTENED AORTA DATA Z-SLICES *** END
#====================================================================================


# ====================================================================================
# *** MASKED SLICED DATA ****
#====================================================================================
def prepare_and_write_masked_data_sliced_bern(basepath,
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
    common_image_shape = [36, 36, 64, 48, 4] # [x, y, z, t, num_channels]
    
    end_shape = [32, 32, 64, 48, 4]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)

    # ==========================================
    # ==========================================
    hand_seg_path_controls = basepath + '/segmenter_rw_pw_hard/controls'
    hand_seg_path_patients = basepath + '/segmenter_rw_pw_hard/patients'
    list_hand_seg_images = os.listdir(hand_seg_path_controls) + os.listdir(hand_seg_path_patients)
    if ['train', 'val'].__contains__(train_test):

        seg_path = basepath + '/final_segmentations/controls'
        img_path = basepath + '/preprocessed/controls/numpy'
    elif train_test == 'test':
        seg_path = basepath + '/final_segmentations/patients'
        img_path = basepath + '/preprocessed/patients/numpy'
    else:
        raise ValueError('train_test must be either train, val or test')
    
    patients = os.listdir(seg_path)[idx_start:idx_end]
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
    
    cnn_predictions = True
    i = 0
    for patient in patients: 
        
        #print('loading subject ' + str(n-idx_start+1) + ' out of ' + str(num_images_to_load) + '...')
        print('loading subject ' + str(i+1) + ' out of ' + str(num_images_to_load) + '...')

        print('patient: ' + patient)
        # Check if hand or network segemented (slightly different kernel size on pre-processing)
        if patient in list_hand_seg_images:
            cnn_predictions = False
        
        image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        segmented_original = np.load(os.path.join(seg_path, patient))
        

        # Enlarge the segmentation slightly to be sure that there are no cutoffs of the aorta
        time_steps = segmented_original.shape[3]
        segmented = dilation(segmented_original[:,:,:,3], cube(6))

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

        

        # Limit to sectors where ascending aorta is located
        points = points[np.where(points[:,1]<60)]
        points = points[np.where(points[:,0]<90)]

        # Order the points in ascending order with x
        points = points[points[:,0].argsort()[::-1]]

        temp = []
        for index, element in enumerate(points[5:]):
            if (index%5)==0:
                temp.append(element)

        coords = np.array(temp)
        

        #===========================================================================================
        # Parameters for the interpolation and creation of the files

        # We create Slices across time and channels in a double for loop
        temp_for_channel_stacking = []
        for channel in range(image.shape[4]):

            temp_for_time_stacking = []
            for t in range(image.shape[3]):
                straightened = interpolate_and_slice(image[:,:,:,t,channel], coords, common_image_shape)
                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        straightened = np.stack(temp_for_channel_stacking, axis=-1)
        image_out = straightened

        # make all images of the same shape
        print("Image shape before cropping and padding:" + str(image_out.shape))
        image_out = crop_or_pad_normal_slices(image_out, end_shape)
        print("Image shape after cropping and padding:" + str(image_out.shape))

        # move the z-axis to the front, as we want to stack the data along this axis
        image_out = np.moveaxis(image_out, 2, 0)

        # add the image to the hdf5 file
        dataset['sliced_images_%s' % train_test][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out

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
              load_anomalous=False):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    savepath = sys_config.project_code_root + 'data'
    dataset_filepath = savepath + f'/{train_test}_masked_sliced_images_from_' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_masked_data_sliced_bern(basepath = basepath,
                               filepath_output = dataset_filepath,
                               idx_start = idx_start,
                               idx_end = idx_end,
                               train_test = train_test,
                               load_anomalous= load_anomalous)
    else:
        print('Already preprocessed this configuration. Loading now...')

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
    common_image_shape = [36, 36, 256, 48, 4] # [x, y, z, t, num_channels]

    #network_common_image_shape = [144, 112, None, 48, 4] # [x, y, t, num_channels]

    end_shape = [32, 32, 256, 48, 4]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)
    
    # ==========================================
    # ==========================================
    
    hand_seg_path_controls = basepath + '/segmenter_rw_pw_hard/controls'
    hand_seg_path_patients = basepath + '/segmenter_rw_pw_hard/patients'
    list_hand_seg_images = os.listdir(hand_seg_path_controls) + os.listdir(hand_seg_path_patients)

    if ['train', 'val'].__contains__(train_test):

        seg_path = basepath + '/final_segmentations/controls'
        img_path = basepath + '/preprocessed/controls/numpy'
    elif train_test == 'test':
        seg_path = basepath + '/final_segmentations/patients'
        img_path = basepath + '/preprocessed/patients/numpy'
    else:
        raise ValueError('train_test must be either train, val or test')
    
    patients = os.listdir(seg_path)[idx_start:idx_end]
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

        print('loading subject ' + str(i+1) + ' out of ' + str(num_images_to_load) + '...')
        print('patient: ' + patient)
        # Check if hand or network segemented (slightly different kernel size on pre-processing)
        if patient in list_hand_seg_images:
            cnn_predictions = False

        # load the segmentation that was created with Nicolas's tool
        image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        
        
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
        for index, element in enumerate(points[5:]):
            if (index%5)==0:
                temp.append(element)


        coords = np.array(temp)
        

        #===========================================================================================
        # Parameters for the interpolation and creation of the files

        # We create Slices across time and channels in a double for loop
        temp_for_channel_stacking = []
        for channel in range(image.shape[4]):

            temp_for_time_stacking = []
            for t in range(image.shape[3]):
                straightened = interpolate_and_slice(image[:,:,:,t,channel], coords, common_image_shape, smoothness=10)
                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        straightened = np.stack(temp_for_channel_stacking, axis=-1)

        image_out  = straightened

        # make all images of the same shape
        print("Image shape before cropping and padding:" + str(image_out.shape))
        #image_out = crop_or_pad_Bern_all_slices(image_out, network_common_image_shape)
        image_out = crop_or_pad_normal_slices(image_out, end_shape)
        print("Image shape after cropping and padding:" + str(image_out.shape))

        if stack_z == True:
            # move the z-axis to the front, as we want to stack the data along this axis
            image_out = np.moveaxis(image_out, 2, 0)

            # add the image to the hdf5 file
            dataset['sliced_images_%s' % train_test][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out

        else:
            # move the y-axis to the front, as we want to stack the data along this axis
            image_out = np.moveaxis(image_out, 1, 0)

            print('After shuffling the axis' + str(image_out.shape))
            print(str(np.max(image_out)))

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
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_sliced_data_full_aorta_bern(basepath = basepath,
                               filepath_output = dataset_filepath,
                               idx_start=idx_start,
                                 idx_end=idx_end,
                               train_test = train_test,
                               stack_z = True)
    else:
        print('Already preprocessed this configuration. Loading now...')

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
    common_image_shape = [36, 36, 256, 48, 4] # [x, y, z, t, num_channels]
    
    end_shape = [32, 32, 256, 48, 4]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)

    # ==========================================
    # ==========================================

    hand_seg_path_controls = basepath + '/segmenter_rw_pw_hard/controls'
    hand_seg_path_patients = basepath + '/segmenter_rw_pw_hard/patients'
    list_hand_seg_images = os.listdir(hand_seg_path_controls) + os.listdir(hand_seg_path_patients)
    if ['train', 'val'].__contains__(train_test):

        seg_path = basepath + '/final_segmentations/controls'
        img_path = basepath + '/preprocessed/controls/numpy'
    elif train_test == 'test':
        seg_path = basepath + '/final_segmentations/patients'
        img_path = basepath + '/preprocessed/patients/numpy'
    else:
        raise ValueError('train_test must be either train, val or test')
    
    patients = os.listdir(seg_path)[idx_start:idx_end]
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
        
    
    cnn_predictions = True
    
    i = 0
    for patient in patients: 
        
        #print('loading subject ' + str(n-idx_start+1) + ' out of ' + str(num_images_to_load) + '...')
        print('loading subject ' + str(i+1) + ' out of ' + str(num_images_to_load) + '...')

        print('patient: ' + patient)

        # Check if hand or network segemented (slightly different kernel size on pre-processing)
        if patient in list_hand_seg_images:
            cnn_predictions = False
        
        image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        segmented_original = np.load(os.path.join(seg_path, patient))
        

        # Enlarge the segmentation slightly to be sure that there are no cutoffs of the aorta
        time_steps = segmented_original.shape[3]
        segmented = dilation(segmented_original[:,:,:,3], cube(6))

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
        for index, element in enumerate(points[5:]):
            if (index%5)==0:
                temp.append(element)

        coords = np.array(temp)
        

        #===========================================================================================
        # Parameters for the interpolation and creation of the files

        # We create Slices across time and channels in a double for loop
        temp_for_channel_stacking = []
        for channel in range(image.shape[4]):

            temp_for_time_stacking = []
            for t in range(image.shape[3]):
                straightened = interpolate_and_slice(image[:,:,:,t,channel], coords, common_image_shape, smoothness=10)
                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        straightened = np.stack(temp_for_channel_stacking, axis=-1)
        image_out = straightened

        # make all images of the same shape
        print("Image shape before cropping and padding:" + str(image_out.shape))
        image_out = crop_or_pad_normal_slices(image_out, end_shape)
        print("Image shape after cropping and padding:" + str(image_out.shape))

        # move the z-axis to the front, as we want to stack the data along this axis
        image_out = np.moveaxis(image_out, 2, 0)

        # add the image to the hdf5 file
        dataset['sliced_images_%s' % train_test][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out

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
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_masked_data_sliced_full_aorta_bern(basepath = basepath,
                               filepath_output = dataset_filepath,
                               idx_start = idx_start,
                               idx_end = idx_end,
                               train_test = train_test,
                               load_anomalous= load_anomalous)
    else:
        print('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')

# ====================================================================================
# *** MASKED SLICED DATA FULL AORTA END ****
#====================================================================================


if __name__ == '__main__':

    basepath =  sys_config.project_data_root #"/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady"
    savepath = sys_config.project_code_root + "data"
    make_dir_safely(savepath)

    masked_data_train = load_masked_data(basepath, idx_start=0, idx_end=5, train_test='train')
    masked_data_validation = load_masked_data(basepath, idx_start=5, idx_end=8, train_test='val')

    sliced_data_train = load_cropped_data_sliced(basepath, idx_start=0, idx_end=5, train_test='train')
    sliced_data_validation = load_cropped_data_sliced(basepath, idx_start=5, idx_end=8, train_test='val')
    
    masked_sliced_data_train = load_masked_data_sliced(basepath, idx_start=0, idx_end=5, train_test='train')
    masked_sliced_data_validation = load_masked_data_sliced(basepath, idx_start=4, idx_end=8, train_test='val')    

    sliced_data_full_aorta_train = load_cropped_data_sliced_full_aorta(basepath, idx_start=0, idx_end=5, train_test='train')
    sliced_data_full_aorta_validation = load_cropped_data_sliced_full_aorta(basepath, idx_start=5, idx_end=8, train_test='val')

    masked_sliced_data_full_aorta_train = load_masked_data_sliced_full_aorta(basepath, idx_start=0, idx_end=5, train_test='train')
    masked_sliced_data_full_aorta_validation = load_masked_data_sliced_full_aorta(basepath, idx_start=5, idx_end=8, train_test='val')

    
    

