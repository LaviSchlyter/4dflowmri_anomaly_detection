import numpy as np

from data_augmentation import do_data_augmentation

import math

from matplotlib import pyplot as plt


def tile_3d_complete_SSL(X, Out_Mu, Y, rows, cols, every_x_time_step, channel_to_show = 1):
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
                img = X[subject,:,:,j*every_x_time_step,channel+channel_to_show]
                #print(Out_Mu.shape)
                out_mu = Out_Mu[subject,:,:,j*every_x_time_step,channel]
                label = Y[subject, :, :, j*every_x_time_step, channel + channel_to_show]
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
                        j*X.shape[2]:(j+1)*X.shape[2]] = label

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




def plot_batches_SSL(batch, Out_Mu, Y, channel_to_show, every_x_time_step, out_path):
    X = np.stack(batch)
    Out_Mu = np.stack(Out_Mu)
    rows = X.shape[0]
    cols = math.ceil(X.shape[3] // every_x_time_step)
    canvas = tile_3d_complete_SSL(X, Out_Mu, Y,  rows, cols, every_x_time_step, channel_to_show = channel_to_show)
    canvas = np.squeeze(canvas)
    if channel_to_show == 0:
        plt.imsave(out_path, canvas, cmap = 'viridis')
    else:
        plt.imsave(out_path, canvas, cmap = 'gray')


def tile_3d_in_out(X, Out_Mu, rows, cols, every_x_time_step, channel_to_show = 1):
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
    n_rows_per_subject = 2

    tiling = np.zeros((rows * X.shape[1] * n_rows_per_subject * c_ + rows * row_seperator_width * n_rows_per_subject * c_, cols * X.shape[2] + cols), dtype = X.dtype)

    #Loop through all the channels
    i = 0
    subject = 0

    # Rows is the number of samples in a batch, 3 comes from the fact that we draw 3 rows per subject. We have c_ channels.
    while i < (rows*n_rows_per_subject*1-1):
        for channel in range(1):
            for j in range(cols):
                img = X[subject,:,:,j*every_x_time_step,channel+channel_to_show]
                #print(Out_Mu.shape)
                out_mu = Out_Mu[subject,:,:,j*every_x_time_step,channel]
                #label = Y[subject, :, :, j*every_x_time_step, channel + channel_to_show]
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

                ## Difference of the images
                #tiling[
                #        (i+2)*X.shape[1]+ separator_offset:(i+3)*X.shape[1]+ separator_offset,
                #        j*X.shape[2]:(j+1)*X.shape[2]] = label
#
            # White line to separate this from the next channel
            tiling[
                    (i+2)*X.shape[1]+ separator_offset:(i+2)*X.shape[1]+ separator_offset + row_seperator_width,
                    0:(cols-1)*X.shape[2]] = 1

            # Increase row separator count
            row_separator_counter += 1

            #One channel is now complete, so increase i by 2 (2 rows are done)
            i += n_rows_per_subject

        #One subject is now complete, so move to the next subject in the batch
        subject += 1

    return tiling




def plot_batches_SSL_in_out(batch, Out_Mu,channel_to_show, every_x_time_step, out_path):
    X = np.stack(batch)
    Out_Mu = np.stack(Out_Mu)
    rows = X.shape[0]
    cols = math.ceil(X.shape[3] // every_x_time_step)
    canvas = tile_3d_in_out(X, Out_Mu, rows, cols, every_x_time_step, channel_to_show = channel_to_show)
    canvas = np.squeeze(canvas)
    if channel_to_show == 0:
        plt.imsave(out_path, canvas, cmap = 'viridis')
    else:
        plt.imsave(out_path, canvas, cmap = 'gray')

