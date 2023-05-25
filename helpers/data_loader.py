import data_bern_numpy_to_hdf5
import data_bern_numpy_to_preprocessed_hdf5
import logging
import os
import numpy as np



# ==================================================================
# Load the data
# ==================================================================
def load_data(config, config_sys, idx_start_tr = 0, idx_end_tr = 5, idx_start_vl = 5, idx_end_vl = 8):
    
    if config['preprocess_method'] == 'none':

            logging.info('=============================================================================')
            logging.info(f"Preprocessing method: {config['preprocess_method']}")
            logging.info('Loading training data from: {}'.format(config_sys.project_data_root))

            data_tr = data_bern_numpy_to_hdf5.load_data(basepath=config_sys.project_data_root, 
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
            logging.info('Loading validation data from: {}'.format(config_sys.project_data_root))

            data_vl = data_bern_numpy_to_hdf5.load_data(basepath=config_sys.project_data_root,
                                                        idx_start=idx_start_vl,
                                                        idx_end=idx_end_vl,
                                                        train_test='val',
                                                        )
            
            images_vl = data_vl['images_val']
            labels_vl = data_vl['labels_val']
            logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
            logging.info('Shape of validation labels: %s' %str(labels_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t]

    # ================================================
    # === If mask preprocessing is selected ==========
    # ================================================
    elif config['preprocess_method'] == 'mask':

        logging.info('=============================================================================')
        logging.info(f"Preprocessing method: {config['preprocess_method']}")
        logging.info('Loading training data from: {}'.format(config_sys.project_data_root))

        data_tr = data_bern_numpy_to_preprocessed_hdf5.load_masked_data(basepath=config_sys.project_data_root,
                                                                        idx_start=idx_start_tr,
                                                                        idx_end=idx_end_tr,
                                                                        train_test='train')
        
        images_tr = data_tr['masked_images_train']
        logging.info(type(images_tr))
        logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]

        logging.info('=============================================================================')
        logging.info('Loading validation data from: {}'.format(config_sys.project_data_root))

        data_vl = data_bern_numpy_to_preprocessed_hdf5.load_masked_data(basepath=config_sys.project_data_root,
                                                                        idx_start=idx_start_vl,
                                                                        idx_end=idx_end_vl,
                                                                        train_test='val',
                                                                        )
        
        images_vl = data_vl['masked_images_val']
        logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('=============================================================================')
    
    # ================================================
    # === If slicing preprocessing is selected ==========

    elif config['preprocess_method'] == 'slice':

        logging.info('=============================================================================')
        logging.info(f"Preprocessing method: {config['preprocess_method']}")
        logging.info('Loading training data from: {}'.format(config_sys.project_data_root))

        data_tr = data_bern_numpy_to_preprocessed_hdf5.load_cropped_data_sliced(basepath=config_sys.project_data_root,
                                                                                idx_start=idx_start_tr,
                                                                                idx_end=idx_end_tr,
                                                                                train_test='train',
                                                                                )
        
        images_tr = data_tr['sliced_images_train']
        logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('=============================================================================')

        logging.info('=============================================================================')
        logging.info('Loading validation data from: {}'.format(config_sys.project_data_root))

        data_vl = data_bern_numpy_to_preprocessed_hdf5.load_cropped_data_sliced(basepath=config_sys.project_data_root,
                                                                                idx_start=idx_start_vl,
                                                                                idx_end=idx_end_vl,
                                                                                train_test='val',
                                                                                )
        
        images_vl = data_vl['sliced_images_val']
        logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('=============================================================================')

    # ================================================
    # ==== If masked slicing preprocessing is selected
    # ================================================

    elif config['preprocess_method'] == 'masked_slice':

        logging.info('=============================================================================')
        logging.info(f"Preprocessing method: {config['preprocess_method']}")
        logging.info('Loading training data from: {}'.format(config_sys.project_data_root))

        data_tr = data_bern_numpy_to_preprocessed_hdf5.load_masked_data_sliced(basepath=config_sys.project_data_root,
                                                                                idx_start=idx_start_tr,
                                                                                idx_end=idx_end_tr,
                                                                                train_test='train',
                                                                                )
        
        images_tr = data_tr['sliced_images_train']
        logging.info(type(images_tr))
        logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        
        logging.info('=============================================================================')
        logging.info('Loading validation data from: {}'.format(config_sys.project_data_root))

        data_vl = data_bern_numpy_to_preprocessed_hdf5.load_masked_data_sliced(basepath=config_sys.project_data_root,
                                                                                idx_start=idx_start_vl,
                                                                                idx_end=idx_end_vl,
                                                                                train_test='val',
                                                                                )
        
        images_vl = data_vl['sliced_images_val']
        logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('=============================================================================')

    
    # ================================================
    # ==== if sliced full aorta preprocessing is selected
    # ================================================
    elif config['preprocess_method'] == 'sliced_full_aorta':

        logging.info('=============================================================================')
        logging.info(f"Preprocessing method: {config['preprocess_method']}")
        logging.info('Loading training data from: {}'.format(config_sys.project_data_root))

        data_tr = data_bern_numpy_to_preprocessed_hdf5.load_cropped_data_sliced_full_aorta(basepath=config_sys.project_data_root,
                                                                                            idx_start=idx_start_tr,
                                                                                            idx_end=idx_end_tr,
                                                                                            train_test='train',
                                                                                            )
        
        images_tr = data_tr['sliced_images_train']
        logging.info(type(images_tr))
        logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]

        logging.info('=============================================================================')
        logging.info('Loading validation data from: {}'.format(config_sys.project_data_root))

        data_vl = data_bern_numpy_to_preprocessed_hdf5.load_cropped_data_sliced_full_aorta(basepath=config_sys.project_data_root,
                                                                                            idx_start=idx_start_vl,
                                                                                            idx_end=idx_end_vl,
                                                                                            train_test='val',
                                                                                            )
        
        images_vl = data_vl['sliced_images_val']
        logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('=============================================================================')

    # ================================================
    # ==== if masked sliced full aorta preprocessing is selected
    # ================================================

    elif config['preprocess_method'] == 'masked_sliced_full_aorta':

        logging.info('=============================================================================')
        logging.info(f"Preprocessing method: {config['preprocess_method']}")
        logging.info('Loading training data from: {}'.format(config_sys.project_data_root))

        data_tr = data_bern_numpy_to_preprocessed_hdf5.load_masked_data_sliced_full_aorta(basepath=config_sys.project_data_root,
                                                                                            idx_start=idx_start_tr,
                                                                                            idx_end=idx_end_tr,
                                                                                            train_test='train',
                                                                                            )
        
        images_tr = data_tr['sliced_images_train']
        logging.info(type(images_tr))
        logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]

        logging.info('=============================================================================')
        logging.info('Loading validation data from: {}'.format(config_sys.project_data_root))

        data_vl = data_bern_numpy_to_preprocessed_hdf5.load_masked_data_sliced_full_aorta(basepath=config_sys.project_data_root,
                                                                                            idx_start=idx_start_vl,
                                                                                            idx_end=idx_end_vl,
                                                                                            train_test='val',
                                                                                            )
        
        images_vl = data_vl['sliced_images_val']
        logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('=============================================================================')

    # ================================================
    # ==== if mock_square preprocessing is selected
    # ================================================
    elif config['preprocess_method'] == 'mock_square':
         mock_square_path = os.path.join(config_sys.project_code_root, 'data/mock_square')
         images_tr = np.load(os.path.join(mock_square_path, 'square_0.npy'))
         images_vl = np.load(os.path.join(mock_square_path, 'square_1.npy'))
         

    else:
        raise ValueError(f"Preprocessing method {config['preprocess_method']} not implemented.")
    
    return images_tr, images_vl