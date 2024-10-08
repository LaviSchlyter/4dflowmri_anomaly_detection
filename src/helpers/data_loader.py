import sys
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection')

from helpers import data_bern_numpy_to_hdf5
from helpers import data_bern_numpy_to_preprocessed_hdf5
import logging
import os
import numpy as np
import h5py



# ==================================================================
# Load the data
# ==================================================================
def load_data(config, sys_config, idx_start_tr = 0, idx_end_tr = 1, idx_start_vl = 0, idx_end_vl = 1, idx_start_ts = 0, idx_end_ts = 1, with_test_labels = False, suffix = '', include_compressed_sensing = True, only_compressed_sensing = False):
    """
    Load the data from the numpy files and preprocess it according to the config file.
    
    Parameters
    ----------
    config : dict
        Configuration file.
    sys_config : dict
        System configuration file.
    idx_start_tr : int
        Index of the first training image to load.
    idx_end_tr : int
        Index of the last training image to load.
    idx_start_vl : int
        Index of the first validation image to load.
    idx_end_vl : int
        Index of the last validation image to load.
    idx_start_ts : int
        Index of the first test image to load.
    idx_end_ts : int
        Index of the last test image to load.
    with_test_labels : bool
        If True, the test labels are returned as well.
    
    Returns
    -------"""

    labels_ts = None
    rotation_tr = None
    rotation_vl = None
    rotation_ts = None
    
    if config['preprocess_method'] == 'none':

            logging.info('=============================================================================')
            logging.info(f"Preprocessing method: {config['preprocess_method']}")
            logging.info('Loading training data from: {}'.format(sys_config.project_data_root))

            data_tr = data_bern_numpy_to_hdf5.load_data(basepath=sys_config.project_data_root, 
                                                        idx_start=idx_start_tr, 
                                                        idx_end=idx_end_tr, 
                                                        train_test='train',

                                                        )
            

            images_tr = data_tr['images_train']            
            labels_tr = data_tr['labels_train']    
            logging.info(type(images_tr))    
            logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
            logging.info('Shape of training labels: %s' %str(labels_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t]

            logging.info('=============================================================================')
            logging.info('Loading validation data from: {}'.format(sys_config.project_data_root))

            data_vl = data_bern_numpy_to_hdf5.load_data(basepath=sys_config.project_data_root,
                                                        idx_start=idx_start_vl,
                                                        idx_end=idx_end_vl,
                                                        train_test='val',
                                                        )
            
            images_vl = data_vl['images_val']
            labels_vl = data_vl['labels_val']
            logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
            logging.info('Shape of validation labels: %s' %str(labels_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t]

            data_ts = data_bern_numpy_to_hdf5.load_data(basepath=sys_config.project_data_root,
                                                        idx_start=idx_start_ts,
                                                        idx_end=idx_end_ts,
                                                        train_test='test',
                                                        )
            
            images_ts = data_ts['images_test']
            if with_test_labels:
                labels_ts = data_ts['labels_test']
            

    # ================================================
    # === If mask preprocessing is selected ==========
    # ================================================
    elif config['preprocess_method'] == 'mask':

        logging.info('=============================================================================')
        logging.info(f"Preprocessing method: {config['preprocess_method']}")
        logging.info('Loading training data from: {}'.format(sys_config.project_data_root))

        data_tr = data_bern_numpy_to_preprocessed_hdf5.load_masked_data(basepath=sys_config.project_data_root,
                                                                        idx_start=idx_start_tr,
                                                                        idx_end=idx_end_tr,
                                                                        train_test='train')
        
        images_tr = data_tr['masked_images_train']
        logging.info(type(images_tr))
        logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]

        logging.info('=============================================================================')
        logging.info('Loading validation data from: {}'.format(sys_config.project_data_root))

        data_vl = data_bern_numpy_to_preprocessed_hdf5.load_masked_data(basepath=sys_config.project_data_root,
                                                                        idx_start=idx_start_vl,
                                                                        idx_end=idx_end_vl,
                                                                        train_test='val',
                                                                        )
        
        images_vl = data_vl['masked_images_val']
        logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('=============================================================================')

        data_ts = data_bern_numpy_to_preprocessed_hdf5.load_masked_data(basepath=sys_config.project_data_root,
                                                                        idx_start=idx_start_ts,
                                                                        idx_end=idx_end_ts,
                                                                        train_test='test',
                                                                        )
        
        images_ts = data_ts['masked_images_test']
        if with_test_labels:
                labels_ts = data_ts['labels_test']
        
    
    # ================================================
    # === If slicing preprocessing is selected ==========

    elif config['preprocess_method'] == 'slice':

        logging.info('=============================================================================')
        logging.info(f"Preprocessing method: {config['preprocess_method']}")
        logging.info('Loading training data from: {}'.format(sys_config.project_data_root))

        data_tr = data_bern_numpy_to_preprocessed_hdf5.load_cropped_data_sliced(basepath=sys_config.project_data_root,
                                                                                idx_start=idx_start_tr,
                                                                                idx_end=idx_end_tr,
                                                                                train_test='train',
                                                                                )
        
        images_tr = data_tr['sliced_images_train']
        logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('=============================================================================')

        logging.info('=============================================================================')
        logging.info('Loading validation data from: {}'.format(sys_config.project_data_root))

        data_vl = data_bern_numpy_to_preprocessed_hdf5.load_cropped_data_sliced(basepath=sys_config.project_data_root,
                                                                                idx_start=idx_start_vl,
                                                                                idx_end=idx_end_vl,
                                                                                train_test='val',
                                                                                )
        
        images_vl = data_vl['sliced_images_val']
        logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('=============================================================================')

        data_ts = data_bern_numpy_to_preprocessed_hdf5.load_cropped_data_sliced(basepath=sys_config.project_data_root,
                                                                                idx_start=idx_start_ts,
                                                                                idx_end=idx_end_ts,
                                                                                train_test='test',
                                                                                suffix = suffix,
                                                                                )
        
        images_ts = data_ts['sliced_images_test']
        if with_test_labels:
                labels_ts = data_ts['labels_test']


    # ================================================
    # ==== If masked slicing preprocessing is selected
    # ================================================

    elif config['preprocess_method'] == 'masked_slice':

        logging.info('=============================================================================')
        logging.info(f"Preprocessing method: {config['preprocess_method']}")
        logging.info('Loading training data from: {}'.format(sys_config.project_data_root))

        data_tr = data_bern_numpy_to_preprocessed_hdf5.load_masked_data_sliced(basepath=sys_config.project_data_root,
                                                                                idx_start=idx_start_tr,
                                                                                idx_end=idx_end_tr,
                                                                                train_test='train',
                                                                                suffix = suffix,
                                                                                include_compressed_sensing=include_compressed_sensing,
                                                                                only_compressed_sensing=only_compressed_sensing,
                                                                                )
        
        images_tr = data_tr['sliced_images_train']
        rotation_tr = data_tr['rotation_matrix']
        logging.info(type(images_tr))
        logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('Shape of training rotation matrix dataset: %s' %str(rotation_tr.shape)) 
        
        logging.info('=============================================================================')
        logging.info('Loading validation data from: {}'.format(sys_config.project_data_root))

        data_vl = data_bern_numpy_to_preprocessed_hdf5.load_masked_data_sliced(basepath=sys_config.project_data_root,
                                                                                idx_start=idx_start_vl,
                                                                                idx_end=idx_end_vl,
                                                                                train_test='val',
                                                                                suffix=suffix,
                                                                                include_compressed_sensing=include_compressed_sensing,
                                                                                only_compressed_sensing=only_compressed_sensing,
                                                                                )
        
        images_vl = data_vl['sliced_images_val']
        rotation_vl = data_vl['rotation_matrix']
        logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('Shape of validation rotation matrix dataset: %s' %str(rotation_vl.shape))
        logging.info('=============================================================================')
        

        logging.info('Loading test data from: {}'.format(sys_config.project_data_root))
        data_ts = data_bern_numpy_to_preprocessed_hdf5.load_masked_data_sliced(basepath=sys_config.project_data_root,
                                                                                idx_start=idx_start_ts,
                                                                                idx_end=idx_end_ts,
                                                                                train_test='test',
                                                                                suffix = suffix,
                                                                                include_compressed_sensing=include_compressed_sensing,
                                                                                only_compressed_sensing=only_compressed_sensing,
                                                                                )
        
        images_ts = data_ts['sliced_images_test']
        rotation_ts = data_ts['rotation_matrix']
        if with_test_labels:
                labels_ts = data_ts['labels_test']

        logging.info('Shape of test images: %s' %str(images_ts.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('Shape of test rotation matrix dataset: %s' %str(rotation_ts.shape))
        logging.info('=============================================================================')


    
    # ================================================
    # ==== if sliced full aorta preprocessing is selected
    # ================================================
    elif config['preprocess_method'] == 'sliced_full_aorta':

        logging.info('=============================================================================')
        logging.info(f"Preprocessing method: {config['preprocess_method']}")
        logging.info('Loading training data from: {}'.format(sys_config.project_data_root))

        data_tr = data_bern_numpy_to_preprocessed_hdf5.load_cropped_data_sliced_full_aorta(basepath=sys_config.project_data_root,
                                                                                            idx_start=idx_start_tr,
                                                                                            idx_end=idx_end_tr,
                                                                                            train_test='train',
                                                                                            )
        
        images_tr = data_tr['sliced_images_train']
        logging.info(type(images_tr))
        logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]

        logging.info('=============================================================================')
        logging.info('Loading validation data from: {}'.format(sys_config.project_data_root))

        data_vl = data_bern_numpy_to_preprocessed_hdf5.load_cropped_data_sliced_full_aorta(basepath=sys_config.project_data_root,
                                                                                            idx_start=idx_start_vl,
                                                                                            idx_end=idx_end_vl,
                                                                                            train_test='val',
                                                                                            )
        
        images_vl = data_vl['sliced_images_val']
        logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('=============================================================================')

        data_ts = data_bern_numpy_to_preprocessed_hdf5.load_cropped_data_sliced_full_aorta(basepath=sys_config.project_data_root,
                                                                                            idx_start=idx_start_ts,
                                                                                            idx_end=idx_end_ts,
                                                                                            train_test='test',
                                                                                            )
        
        images_ts = data_ts['sliced_images_test']
        if with_test_labels:
                labels_ts = data_ts['labels_test']


    # ================================================
    # ==== if masked sliced full aorta preprocessing is selected
    # ================================================

    elif config['preprocess_method'] == 'masked_sliced_full_aorta':

        logging.info('=============================================================================')
        logging.info(f"Preprocessing method: {config['preprocess_method']}")
        logging.info('Loading training data from: {}'.format(sys_config.project_data_root))

        data_tr = data_bern_numpy_to_preprocessed_hdf5.load_masked_data_sliced_full_aorta(basepath=sys_config.project_data_root,
                                                                                            idx_start=idx_start_tr,
                                                                                            idx_end=idx_end_tr,
                                                                                            train_test='train',
                                                                                            )
        
        images_tr = data_tr['sliced_images_train']
        logging.info(type(images_tr))
        logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]

        logging.info('=============================================================================')
        logging.info('Loading validation data from: {}'.format(sys_config.project_data_root))

        data_vl = data_bern_numpy_to_preprocessed_hdf5.load_masked_data_sliced_full_aorta(basepath=sys_config.project_data_root,
                                                                                            idx_start=idx_start_vl,
                                                                                            idx_end=idx_end_vl,
                                                                                            train_test='val',
                                                                                            )
        
        images_vl = data_vl['sliced_images_val']
        logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('=============================================================================')

        data_ts = data_bern_numpy_to_preprocessed_hdf5.load_masked_data_sliced_full_aorta(basepath=sys_config.project_data_root,
                                                                                            idx_start=idx_start_ts,
                                                                                            idx_end=idx_end_ts,
                                                                                            train_test='test',
                                                                                            )
        
        images_ts = data_ts['sliced_images_test']
        if with_test_labels:
                labels_ts = data_ts['labels_test']



    # ================================================
    # ==== if mock_square preprocessing is selected
    # ================================================
    elif config['preprocess_method'] == 'mock_square':
         mock_square_path = os.path.join(sys_config.project_code_root, 'data/mock_square')
         images_tr = np.load(os.path.join(mock_square_path, 'square_0.npy'))[:,:,:,:24,:]
         images_vl = np.load(os.path.join(mock_square_path, 'square_1.npy'))[:,:,:,:24,:]
         images_ts = None
         
         

    else:
        raise ValueError(f"Preprocessing method {config['preprocess_method']} not implemented.")
    

    # Make a return dictionary
    return_dic = {'images_tr': images_tr, 'images_vl': images_vl, 'images_test': images_ts, 'labels_test': labels_ts,
                  'rotation_tr': rotation_tr, 'rotation_vl': rotation_vl, 'rotation_test': rotation_ts}
    
    return return_dic
   

def load_syntetic_data(preprocess_method,
                        idx_start,
                        idx_end,
                        sys_config,
                        note = '',
                        ):
    savepath= sys_config.project_code_root + "data"
    if len(note) > 0:
        dataset_filepath = savepath + f'/{preprocess_method}_anomalies_images_from_' + str(idx_start) + '_to_' + str(idx_end) + '_' + note + '.hdf5'
    else:
        dataset_filepath = savepath + f'/{preprocess_method}_anomalies_images_from_' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'
    
    if not os.path.exists(dataset_filepath):
        raise ValueError(f"Dataset {dataset_filepath} does not exist. Go to syntehtic_anomalies.py in helpers to generate the file")
    else:
        print('Already preprocessed this configuration. Loading now...')
        # Name of file
        logging.info('Loading data from: {}'.format(dataset_filepath))
    
    return h5py.File(dataset_filepath, 'r')