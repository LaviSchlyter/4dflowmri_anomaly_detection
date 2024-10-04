"""
Quadrant Masks Generation for Anomaly Detection

This script generates and saves quadrant masks for gradient images used in anomaly detection.
The masks are created in two types:
1. Masks with quadrants: Posterior, Right, Anterior, Left. (Quadrants are based on 45-degree angles from main axes)
2. Masks with quadrants: Posterior-right, Anterior-right, Posterior-left, Anterior-left. (Quadrants are based on main axes)

All masks have sizes [# of slices, 4 (quadrants), 32, 32].

Note: We assume that the center of the aorta is in the center of the image, (a.k.a
where the centerline is drawn).

Helper functions are found in utils.py

Author: Lavinia Schlyter
Date: 2024-05-30
"""
import os
import copy
import torch
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from scipy import ndimage
from utils import make_dir_safely, create_all_quadrant_masks_main_axes, compute_all_quadrant_masks_between_axes



def get_quandrant_masks():
    """
    Generate and save quadrant masks for gradient images.

    This function processes gradient images that have undergone centerline preprocessing
    to create quadrant masks based on main axes and angles between axes. The masks are
    then saved to specified directories.
    """

    logging.info('Generating and saving quadrant masks for gradient images...')

    # Define paths for input and output data
    gradient_matching_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/data/gradient_matching'
    main_axes_save_basepath = '/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/data/quadrants_main_axes'
    between_axes_save_basepath = '/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/data/quadrants_between_axes'
    
    # Create output directories if they do not exist
    make_dir_safely(main_axes_save_basepath)
    make_dir_safely(between_axes_save_basepath)
    
    # List and sort all gradient matching files
    gradient_matching_files = os.listdir(gradient_matching_path)
    gradient_matching_files.sort()

    # Process each gradient image
    for gradient_matching_file in gradient_matching_files:  # Adjust the slicing as needed
        logging.info(f'Processing {gradient_matching_file}')
        # Extract subject ID from the filename
        subject_id = gradient_matching_file.split('.npy')[0]

        # Define paths to save the masks
        main_axes_save_path = os.path.join(main_axes_save_basepath, f'{subject_id}_main_axes_masks.npy')
        between_axes_save_path = os.path.join(between_axes_save_basepath, f'{subject_id}_between_axes_masks.npy')
        
        # Check if both mask files already exist
        if os.path.exists(main_axes_save_path) and os.path.exists(between_axes_save_path):
            logging.info(f'Skipping {gradient_matching_file}, masks already exist.')
            continue  # Skip this file and move on to the next
        
        # Load the gradient image
        gradient_image = np.load(os.path.join(gradient_matching_path, gradient_matching_file))
        # Shape: [32, 32, 64, 24, 4]

        # Crop the image to remove parts that may be outside the body
        gradient_image_cropped = gradient_image[2:-2, 2:-2]

        # Create a deep copy of the cropped gradient image
        gradient_image = copy.deepcopy(gradient_image_cropped)

        # Define Sobel kernels for edge detection
        k1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # Kernel for x differences
        k0 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Kernel for y differences

        # Channels for anterior-posterior and left-right directions
        ax_chans = [1, 2]
        all_masks_main_axes = []
        all_masks_between_axes = []

        # Randomly selected time frame, assuming centerline does not change much over time
        time_frame = 3

        # Process each slice in the z-direction
        for z_slice in range(gradient_image.shape[2]):
            ax_angles = []
            ax_mags = []
            for ax_chan in ax_chans:
                # Extract the slice for the current channel and time frame
                oi_sl = gradient_image[:, :, z_slice, time_frame, ax_chan]

                # Apply Sobel filters to compute gradients
                sob1 = ndimage.convolve(oi_sl, k1)
                sob0 = ndimage.convolve(oi_sl, k0)

                # Set small gradient magnitudes to zero
                tol = 1E-8
                sob0[np.abs(sob0) < tol] = 0.
                sob1[np.abs(sob1) < tol] = 0.

                # Compute gradient magnitude and orientation
                sob_mag = np.sqrt((sob1 ** 2) + (sob0 ** 2))
                sob_ori = np.arctan2(-sob0, sob1)  # radians

                # Store the mean gradient magnitude and orientation
                ax_mags.append(np.mean(sob_mag))
                ax_angles.append(np.mean(sob_ori))

            # Create masks for main axes and between axes
            masks_main_axes = create_all_quadrant_masks_main_axes(ax_angles) # Order: Posterior, Right, Anterior, Left
            masks_between_axes = compute_all_quadrant_masks_between_axes(ax_angles) # Order: Posterior-right, Anterior-right, Posterior-left, Anterior-left

            all_masks_main_axes.append(masks_main_axes)
            all_masks_between_axes.append(masks_between_axes)

        # Stack the masks for all slices
        all_masks_main_axes = torch.cat(all_masks_main_axes, dim=0)
        all_masks_between_axes = torch.cat(all_masks_between_axes, dim=0)


        # Save the masks as numpy arrays
        np.save(main_axes_save_path, all_masks_main_axes.numpy())
        np.save(between_axes_save_path, all_masks_between_axes.numpy())
    
    logging.info('Quadrant masks have been generated and saved successfully.')



if __name__ == '__main__':
    # Run the function to generate and save quadrant masks
    get_quandrant_masks()