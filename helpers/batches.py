import numpy as np

from data_augmentation import do_data_augmentation

import math

from matplotlib import pyplot as plt

def iterate_minibatches(images,
                    batch_size,
                    data_augmentation=False):
    '''
    Author: Neerav Kharani, extended by Pol Peiffer

    Function to create mini batches from the dataset of a certain batch size
    :param images: numpy dataset
    :param labels: numpy dataset (same as images/volumes)
    :param batch_size: batch size
    :return: mini batches
    '''

    # ===========================
    # generate indices to randomly select slices in each minibatch
    # ===========================
    n_images = images.shape[0]
    random_indices = np.arange(n_images)
    np.random.shuffle(random_indices)

    # ===========================
    # using only a fraction of the batches in each epoch
    # ===========================
    for b_i in range(0, n_images, batch_size):

        if b_i + batch_size > n_images:
            continue

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]

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
                                     
        yield X


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

def plot_batch_3d_complete(batch, Out_Mu, every_x_time_step, out_path):
    X = np.stack(batch)
    Out_Mu = np.stack(Out_Mu)
    rows = X.shape[0]
    cols = math.ceil(X.shape[3] // every_x_time_step)
    canvas = tile_3d_complete(X, Out_Mu, rows, cols, every_x_time_step)
    canvas = np.squeeze(canvas)
    plt.imsave(out_path, canvas, cmap='gray')