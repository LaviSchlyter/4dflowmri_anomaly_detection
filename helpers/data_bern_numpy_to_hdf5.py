import h5py
import numpy as np
import sys
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/helpers/')
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/')
from utils import normalize_image, crop_or_pad_Bern_slices, make_dir_safely
import os


import config.system as sys_config





# This is when using an arbitrary number of slices
def prepare_and_write_data_bern(basepath,
                                       idx_start,
                                        idx_end,
                                        filepath_output,
                                        train_test,
                                        suffix = ''):
        
    # ==========================================
    # Study the the variation in the sizes along various dimensions (using the function 'find_shapes'), 
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # Update: We keep all the Bern data and leave the batch dimension open to their original size
    # ==========================================
    common_image_shape = [144, 112, 40, 24, 4] # [x, y, t, num_channels]
    common_label_shape = [144, 112, 40, 24] # [x, y,t]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)
    
    # ==========================================
    # ==========================================

    if ['train', 'val'].__contains__(train_test):

        seg_path = basepath + f'/final_segmentations/controls{suffix}'
        img_path = basepath + f'/preprocessed/controls/numpy{suffix}'
    elif train_test == 'test':
        seg_path = basepath + f'/final_segmentations/patients{suffix}'
        img_path = basepath + f'/preprocessed/patients/numpy{suffix}'
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
    
    labels_dataset_shape = [common_label_shape[2]*num_images_to_load,
                            common_label_shape[0],
                            common_label_shape[1],
                            common_label_shape[3]]  
        
    # ==========================================
    # create a hdf5 file
    # ==========================================
    dataset = {}
    hdf5_file = h5py.File(filepath_output, "w") 
    
    # ==========================================
    # write each subject's image and label data in the hdf5 file
    # ==========================================    
    dataset['images_%s' % train_test] = hdf5_file.create_dataset("images_%s" % train_test, images_dataset_shape, dtype='float32')       
    dataset['labels_%s' % train_test] = hdf5_file.create_dataset("labels_%s" % train_test, labels_dataset_shape, dtype='uint8')       
           
    i = 0
    for patient in patients: 
        print('loading subject ' + str(i+1) + ' out of ' + str(num_images_to_load) + '...')
        print('patient: ' + patient)
        
        # load the numpy image (saved by the dicom2numpy file)
        image_data = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        # normalize the image
        image_data = normalize_image(image_data)
        # make all images of the same shape
        image_data = crop_or_pad_Bern_slices(image_data, common_image_shape)
        # move the z-axis to the front, as we want to concantenate data along this axis
        image_data = np.moveaxis(image_data, 2, 0)                         
        # add the image to the hdf5 file
        dataset['images_%s' % train_test][i*common_image_shape[2]:(i+1)*common_image_shape[2], :, :, :, :] = image_data
    
        # load the numpy label (saved by the random walker segmenter)
        label_data = np.load(os.path.join(seg_path, patient))
        # make all labels of the same shape
        label_data = crop_or_pad_Bern_slices(label_data, common_label_shape)                  
        # move the z-axis to the front, as we want to concantenate data along this axis
        label_data = np.moveaxis(label_data, 2, 0)  
        # cast labels as uints
        label_data = label_data.astype(np.uint8)                       
        # add the image to the hdf5 file
        dataset['labels_%s' % train_test][i*common_label_shape[2]:(i+1)*common_label_shape[2], :, :, :] = label_data
        
        # increment the index being used to write in the hdf5 datasets
        i = i + 1
    
    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0


def load_data(basepath,
              idx_start,
              idx_end,
              train_test,
              force_overwrite=False,
              suffix = ''):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    savepath = sys_config.project_code_root + "data"
    make_dir_safely(savepath)
    dataset_filepath = savepath + f'/{train_test}_images_and_labels_from_' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_data_bern(basepath = basepath,
                               filepath_output = dataset_filepath,
                               idx_start = idx_start,
                               idx_end = idx_end,
                               train_test = train_test,
                               suffix = suffix)
    else:
        print('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')



if __name__ == "__main__":

    basepath =  sys_config.project_data_root #"/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady"
    data_train = load_data(basepath, idx_start=0, idx_end=5, train_test='train')
    data_val = load_data(basepath, idx_start=5, idx_end=10, train_test='val')
