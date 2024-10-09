"""
Backtransform Anomaly Scores from 2D Slice Space + Time to Original 3D Space + Time (by combining the slices)

This script backtransforms the anomaly scores from the 2D slice space and time to the original 3D space and time.

This can be done in inference_test.py, but can be done here as well if outputs are already saved.

Author: Lavinia Schlyter
Date: 2024-05-30
"""
from utils import resample_back, convert_to_vtk, expand_normal_slices, make_dir_safely
import os
import numpy as np
from config import system_eval as config_sys
import SimpleITK as sitk

# List of experiments to process
list_of_experiments = [
    "deeper_bn_conv_enc_dec_aux/masked_slice/20240519-2101_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3"
]

def main():
    project_code_root = config_sys.project_code_root
    project_data_root = config_sys.project_data_root

    # Pathd to the 'raw' images (already converted from dicom to numpy)
    img_path_controls = os.path.join(project_data_root, 'preprocessed/controls/numpy')
    img_path_patients = os.path.join(project_data_root, 'preprocessed/patients/numpy')
    img_path_controls_cs = os.path.join(project_data_root, 'preprocessed/controls_compressed_sensing/numpy')
    img_path_patients_cs = os.path.join(project_data_root, 'preprocessed/patients_compressed_sensing/numpy')
    

    # Path to the geometry information
    geometry_path = os.path.join(project_code_root, f'data/geometry_for_backtransformation')


    for experiment in list_of_experiments:

        print('---------------------------- Expirment Name -----------------------------------')
        print(experiment)
        results_dir = os.path.join(project_code_root,'Results/Evaluation/' + experiment)
        
        # Anomaly scores path
        anomaly_scores_paths = os.path.join(results_dir, f'test/outputs')

        # Save path for backtransformed outputs
        save_path = os.path.join(results_dir, f'test/outputs_backtransformed')
        make_dir_safely(save_path)


        # Get list of subjects
        subjects = os.listdir(anomaly_scores_paths)
        subjects.sort()
        
        # Backtransform anomaly scores for each subject
        for i, subject in enumerate(subjects):
            subject_name = subject.replace("_anomaly_scores", "")

            # Skip if already backtransformed
            if os.path.exists(os.path.join(save_path, subject_name)):
                continue
            
            print("Backtransforming subject: ", subject_name)

            #Load the anomaly scores
            anomaly_scores = np.load(os.path.join(anomaly_scores_paths, subject))
            
            # Transform anomaly scores from (z,c,x,y,t) to (x,y,z,t,c)
            anomaly_scores = anomaly_scores.transpose(2,3,0,4,1)

            
            # Expand to original slice size (x,y) (36,36)
            anomaly_scores = expand_normal_slices(anomaly_scores, [36,36,64,24,4])

            # If the channel dimension is 1, we repeat it 4 times
            if anomaly_scores.shape[-1] == 1:
                anomaly_scores = np.repeat(anomaly_scores, 4, axis=-1)
            # Define possible image paths
            possible_paths = [
                os.path.join(img_path_controls, subject_name),
                os.path.join(img_path_controls_cs, subject_name),
                os.path.join(img_path_patients, subject_name),
                os.path.join(img_path_patients_cs, subject_name)
            ]

            # Try loading the image from one of the possible paths
            image = None
            for path in possible_paths:
                if os.path.exists(path):
                    image = np.load(path)
                    break
            
            if image is None:
                print(f"Image not found for subject: {subject_name}")
                continue
            # Get SimpleITK image for geometric information
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


                
if __name__ == '__main__':
    main()