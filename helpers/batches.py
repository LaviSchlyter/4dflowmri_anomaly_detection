import numpy as np

from data_augmentation import do_data_augmentation



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