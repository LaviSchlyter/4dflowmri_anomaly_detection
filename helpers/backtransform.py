# In this script we take the anomaly detection results and backtransform them to the original image space
from utils import resample_back, convert_to_vtk, expand_normal_slices, \
                make_dir_safely

import os
import numpy as np
from config import system_eval as config_sys

import SimpleITK as sitk


list_of_best_experiments_run1 = ["vae_convT/masked_slice/20230719-1823_vae_convT_masked_slice_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse-f100",
                                     "vae_convT/masked_slice/20230719-1840_vae_convT_masked_slice_SSL_lr1.500e-03_scheduler-e1500-bs8-gf_dim8-daFalse_2Dslice_decreased_interpolation_factor_cube_3",
                                     "cond_vae/masked_slice/20230721-1002_cond_vae_masked_slice_SSL_lr1.800e-03_scheduler-e1500-bs8-gf_dim8-daFalse-n_experts3_2Dslice_decreased_interpolation_factor_cube_3"]

list_of_experiments = list_of_best_experiments_run1

# Main
if __name__ == '__main__':
    project_code_root = config_sys.project_code_root
    project_data_root = config_sys.project_data_root

    # Path to the raw images
    img_path = os.path.join(project_data_root, 'preprocessed/patients/numpy')

    # Path to the geometry information
    geometry_path = os.path.join(project_code_root, 'data/geometry_for_backtransformation')


    for experiment in list_of_experiments:
        results_dir = os.path.join(project_code_root,'Results/Evaluation/' + experiment)
        
        # Anomaly scores
        anomaly_scores_paths = os.path.join(results_dir, 'test/outputs')

        # Save path
        save_path = os.path.join(results_dir, 'test/outputs_backtransformed')
        make_dir_safely(save_path)


        # ubjects
        subjects = os.listdir(anomaly_scores_paths)
        
        # For each subject we need to backtransform the anomaly scores
        for i, subject in enumerate(subjects):

            subject_name = subject.replace("_anomaly_scores", "")
            print("Backtransforming subject: ", subject_name)
            # Load the anomaly scores
            anomaly_scores = np.load(os.path.join(anomaly_scores_paths, subject))
            
            # Form (z,c,x,y,t) to (x,y,z,t,c)
            anomaly_scores = anomaly_scores.transpose(2,3,0,4,1)

            # The original slice size had a (x,y) (36,36)
            # Reduce to (x,y) (32,32) for network, now we pad back to (36,36)
            anomaly_scores = expand_normal_slices(anomaly_scores, [36,36,64,24,4])

            # If the channel dimension is 1, we repeat it 4 times
            if anomaly_scores.shape[-1] == 1:
                anomaly_scores = np.repeat(anomaly_scores, 4, axis=-1)

            # Load the raw image
            try:
                # With patients
                image = np.load(os.path.join(img_path, subject_name))
                
            except:
                # With controls
                image = np.load(os.path.join(img_path.replace("patients", "controls"), subject_name))

            # The geometric information we need from the initial image does not change with channel or time 
            sitk_image = sitk.GetImageFromArray(image[:,:,:,0,0])

            # Load the geometry information of the slices
            geometry_dict = np.load(os.path.join(geometry_path, subject_name), allow_pickle=True).item()

            # Resample back to the original image space
            anomaly_scores_original_frame = resample_back(anomaly_scores, sitk_image, geometry_dict)

            # Save the backtransformed anomaly scores
            np.save(os.path.join(save_path, subject_name), anomaly_scores_original_frame)

            # Convert to vtk
            save_dir_vtk = os.path.join(save_path, "vtk", subject_name.replace(".npy", ""))
            make_dir_safely(save_dir_vtk)
            convert_to_vtk(anomaly_scores_original_frame, subject_name.replace(".npy", ""), save_dir_vtk)


                
                





            