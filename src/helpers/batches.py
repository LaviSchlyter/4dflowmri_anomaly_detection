import numpy as np
import sys
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/src/helpers')
from data_augmentation import do_data_augmentation

import math

from matplotlib import pyplot as plt
def iterate_minibatches(data,
                    config,
                    data_augmentation=False,
                    with_labels=False,
                    remove_indices = False,
                    indices_to_remove = [],
                    data_dict = None,
                    train_data = True,
                    ):
    '''
    Author: Neerav Kharani, extended by Pol Peiffer and further by Lavinia Schlyter
    Function to create mini batches from the dataset of a certain batch size
    Parameters:
    data: dictionary containing the images and masks
    config: configuration dictionary
    data_augmentation: boolean to indicate if data augmentation should be applied
    with_labels: boolean to indicate if the data contains labels
    remove_indices: boolean to indicate if we want to remove some indices from the data
    indices_to_remove: list of indices to remove from the data
    data_dict: dictionary containing the rotation matrices
    train_data: boolean to indicate if the data is training data
    Returns:
    return_dict: dictionary containing the images, rotation matrix and labels
    '''
    
    # ===========================
    # generate indices to randomly select slices in each minibatch
    # ===========================
    if with_labels:
        images = data['images']
        labels = data['masks']
        if train_data:
            data_rotation_matrix = data_dict['rotation_tr']
        else:
            #  We need to extend the rotaiton matrix for the shown anomalies
            # repeat the rotation matrix 7 times (because 7 deformation types)
            data_rotation_matrix = np.repeat(data_dict['rotation_tr'], 7, axis=0)
           

    else:
        images = data
        if train_data:
            data_rotation_matrix = data_dict['rotation_tr']
        else:
            data_rotation_matrix = data_dict['rotation_vl']
            data_rotation_matrix = np.repeat(data_dict['rotation_vl'], 7, axis=0)
        
    Y = None
    n_images = images.shape[0]
    if with_labels and remove_indices and len(indices_to_remove) == 2:
        # We remove the indices from the validation set
        # Because they contain the same form of anomalies as the training 
        # We remove the set indices within the list given by user
        random_indices = set(np.arange(n_images))
        indices_to_remove = set(np.arange(indices_to_remove[0], indices_to_remove[1]))
        diff = random_indices - indices_to_remove
        random_indices = np.array(list(diff))
        n_images = n_images - len(indices_to_remove)
        
    else:
        random_indices = np.arange(n_images)

    np.random.shuffle(random_indices)

    batch_size = config['batch_size']


    # add a new configuration check
    get_neighbours = config.get('get_neighbours', False)

    # ===========================
    # using only a fraction of the batches in each epoch
    # ===========================
    for b_i in range(0, n_images, batch_size):

        if b_i + batch_size > n_images:
            continue

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        if get_neighbours:
            extended_batch_indices = []
            for idx in batch_indices:
                patient_num = idx//64
                slice_num = idx % 64
                prev_idx = (patient_num * 64 + max(0, slice_num - 1)) if slice_num != 0 else None
                next_idx = (patient_num * 64 + min(63, slice_num + 1)) if slice_num != 63 else None

                extended_batch_indices.append((images[prev_idx] if prev_idx is not None else np.zeros_like(images[0]),
                                           images[idx], 
                                           images[next_idx] if next_idx is not None else np.zeros_like(images[0])))
        
                
                
            
            X = np.stack([idx_tuple for idx_tuple in extended_batch_indices], axis=0)
            
            
            if with_labels:
                #Y = np.stack([idx_tuple for idx_tuple in extended_batch_indices], axis=0)
                Y = labels[batch_indices, ...]

        else:
            X = images[batch_indices, ...]
            rotation_matrix = data_rotation_matrix[batch_indices, ...]
            if with_labels:
                Y = labels[batch_indices, ...]
                
        

        # ===========================
        # augment the batch
        # ===========================
        if data_augmentation:
            X = do_data_augmentation(images=X,
                                     data_aug_ratio=0.5, 
                                     trans_min=-10,
                                     trans_max=10,
                                     rot_min=-10,
                                     rot_max=10,
                                     scale_min=0.9,
                                     scale_max=1.1)
            if with_labels:
                # TODO: FIX: actually not because you don't do data augmentation on the validation if synthetic anomalies present
                Y = do_data_augmentation(images=Y,
                                        data_aug_ratio=0.5,
                                        trans_min=-10,
                                        trans_max=10,
                                        rot_min=-10,
                                        rot_max=10,
                                        scale_min=0.9,
                                        scale_max=1.1)

        

        return_dict = {'X': X, 'rotation_matrix': rotation_matrix, 'Y': Y,'batch_z_slice': batch_indices}                       
        
        yield return_dict


# ============================================
# Batch plotting helper functions
# ============================================
def tile_3d_complete(X, Out_Mu, rows, cols, every_x_time_step):
    """
    Tile images for display.

    Each patient slice in a batch has the following:
    ----------------------------------------------
    1. Row: original image for channel
    2. Out_Mu
    3. Difference
    ----------------------------------------------
    """
    row_separator_counter = 0
    row_seperator_width = 1

    tiling = np.zeros((rows * X.shape[1] * 3 * 4 + rows * row_seperator_width * 3 * 4, cols * X.shape[2] + cols), dtype = X.dtype)

    #Loop through all the channels
    i = 0
    subject = 0

    # Rows is the number of samples in a batch, 3 comes from the fact that we draw 3 rows per subject. We have 4 channels.
    while i < (rows*3*4-1):
        for channel in range(4):
            for j in range(cols):
                img = X[subject,:,:,j*every_x_time_step,channel]
                out_mu = Out_Mu[subject,:,:,j*every_x_time_step,channel]
                difference = np.absolute(img - out_mu)

                separator_offset= row_separator_counter*row_seperator_width

                # Original input image
                tiling[
                        i*X.shape[1] + separator_offset:(i+1)*X.shape[1] + separator_offset,
                        j*X.shape[2]:(j+1)*X.shape[2]] = img

                # Autoencoder prediction
                tiling[
                        (i+1)*X.shape[1]+ separator_offset:(i+2)*X.shape[1]+ separator_offset,
                        j*X.shape[2]:(j+1)*X.shape[2]] = out_mu

                # Difference of the images
                tiling[
                        (i+2)*X.shape[1]+ separator_offset:(i+3)*X.shape[1]+ separator_offset,
                        j*X.shape[2]:(j+1)*X.shape[2]] = difference

            # White line to separate this from the next channel
            tiling[
                    (i+3)*X.shape[1]+ separator_offset:(i+3)*X.shape[1]+ separator_offset + row_seperator_width,
                    0:(cols-1)*X.shape[2]] = 1

            # Increase row separator count
            row_separator_counter += 1

            #One channel is now complete, so increase i by 3 (three rows are done)
            i += 3

        #One subject is now complete, so move to the next subject in the batch
        subject += 1

    return tiling
def tile_3d_complete_1_chan(X, Out_Mu, rows, cols, every_x_time_step):
    """
    Tile images for display.

    Each patient slice in a batch has the following:
    ----------------------------------------------
    1. Row: original image for channel
    2. Out_Mu
    3. Difference
    ----------------------------------------------
    """
    row_separator_counter = 0
    row_seperator_width = 1
    c_ = 1

    tiling = np.zeros((rows * X.shape[1] * 3 * c_ + rows * row_seperator_width * 3 * c_, cols * X.shape[2] + cols), dtype = X.dtype)

    #Loop through all the channels
    i = 0
    subject = 0

    # Rows is the number of samples in a batch, 3 comes from the fact that we draw 3 rows per subject. We have c_ channels.
    while i < (rows*3*1-1):
        for channel in range(1):
            for j in range(cols):
                img = X[subject,:,:,j*every_x_time_step,channel]
                print(out_mu.shape)
                out_mu = Out_Mu[subject,:,:,j*every_x_time_step,channel]
                difference = np.absolute(img - out_mu)

                separator_offset= row_separator_counter*row_seperator_width

                # Original input image
                tiling[
                        i*X.shape[1] + separator_offset:(i+1)*X.shape[1] + separator_offset,
                        j*X.shape[2]:(j+1)*X.shape[2]] = img

                # Autoencoder prediction
                tiling[
                        (i+1)*X.shape[1]+ separator_offset:(i+2)*X.shape[1]+ separator_offset,
                        j*X.shape[2]:(j+1)*X.shape[2]] = out_mu

                # Difference of the images
                tiling[
                        (i+2)*X.shape[1]+ separator_offset:(i+3)*X.shape[1]+ separator_offset,
                        j*X.shape[2]:(j+1)*X.shape[2]] = difference

            # White line to separate this from the next channel
            tiling[
                    (i+3)*X.shape[1]+ separator_offset:(i+3)*X.shape[1]+ separator_offset + row_seperator_width,
                    0:(cols-1)*X.shape[2]] = 1

            # Increase row separator count
            row_separator_counter += 1

            #One channel is now complete, so increase i by 3 (three rows are done)
            i += 3

        #One subject is now complete, so move to the next subject in the batch
        subject += 1

    return tiling


def plot_batch_3d_complete(batch, Out_Mu, every_x_time_step, out_path):
    X = np.stack(batch)
    Out_Mu = np.stack(Out_Mu)
    rows = X.shape[0]
    cols = math.ceil(X.shape[3] // every_x_time_step)
    canvas = tile_3d_complete(X, Out_Mu, rows, cols, every_x_time_step)
    canvas = np.squeeze(canvas)
    plt.imsave(out_path, canvas, cmap='gray')

def plot_batch_3d_complete_1_chan(batch, Out_Mu, every_x_time_step, out_path):
    X = np.stack(batch)
    Out_Mu = np.stack(Out_Mu)
    rows = X.shape[0]
    cols = math.ceil(X.shape[3] // every_x_time_step)
    canvas = tile_3d_complete_1_chan(X, Out_Mu, rows, cols, every_x_time_step)
    canvas = np.squeeze(canvas)
    plt.imsave(out_path, canvas, cmap='gray')