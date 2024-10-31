import os
import argparse
import logging
import pandas as pd
import glob
import numpy as np
import random
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection')
from config import system_eval as config_sys
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/src/helpers')
from utils import make_dir_safely
from data_loader import load_data

from utils_inference import (
    descriptive_stats, correlation_analysis, visualize_data,  
    filter_subjects, backtransform_anomaly_scores,
    compute_metrics, compute_metrics_for_matches, permutation_test_auc_roc, 
    evaluate_predictions, perform_permutation_tests_duplicates, initialize_result_keys,
    get_model_type, load_subject_dict, matches_scenarios, load_config, gather_experiment_paths,
    get_start_end_indices, setup_directories, load_model, infer_batch, is_compressed_sensing, compute_scores_metrics,
    plot_all_scores, compute_region_metrics

)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Set up argument parsing
parser = argparse.ArgumentParser(description="Run inference with specific settings.")
parser.add_argument('--config_path', type=str, help='Path to the configuration YAML file')

# Parse arguments
args = parser.parse_args()

# Load the configuration

config = load_config(args.config_path)




# Subjects that don't have a match between Inselspital and the original dataset 
mismatch_subjects = ['MACDAVD_131', 'MACDAVD_159_', 'MACDAVD_202_']
name_pre_extension = ['']


model_name = config['filters']['model_name']
preprocess_method = config['filters']['preprocess_method']
specific_times = config['filters']['specific_times']

model_dir = os.path.join(config['models_dir'], model_name, preprocess_method)
if not os.path.exists(model_dir):
    logging.error(f"Model directory does not exist: {model_dir}")
    raise FileNotFoundError(f"Model directory does not exist: {model_dir}")


if __name__ == '__main__':
    adjacent_batch_slices = None
    batch_size = 32 

    results_summary = {
        'Model': [],
        'Preprocess Method': [],
        'SEED': [],
        'Rotation': [],
        'p-value AUC-ROC (Permutations)': [],
        'p-value AUC-ROC region based(Permutations)': [],
        'p-value Subset AUC-ROC (Permutations)': [],
        'p-value CS AUC-ROC (Permutations)': [],
        'p-value non-CS AUC-ROC (Permutations)': [],
    }

    list_of_experiments_paths = gather_experiment_paths(model_dir, config['filters']['specific_times'])

    # Initialize keys
    initialize_result_keys(results_summary, ['Original', 'Removed Duplicates', 'Region', 'Region without dup.', 'Region without mismatch', 'Region without mismatch, without dup.', 'CS', 'non-CS' ] + list(matches_scenarios.keys()))
    logging.info('List of experiments paths: {}'.format(list_of_experiments_paths))
        
    for i, exp_path in enumerate(list_of_experiments_paths):
        idx_start_ts, idx_end_ts = get_start_end_indices(exp_path)
        suffix_data_path = ''

    
        # Set up experiment paths
        pattern = os.path.join(exp_path, "*best*")
        best_exp_path = glob.glob(pattern)[0]
        experiment_name = exp_path.split("/")[-1]
        results_dir = os.path.join(config['project_code_root'],'Results/Evaluation/' + model_name + '/' + preprocess_method + '/' + experiment_name)
        

        
        results_dir_test, results_dir_subject, inputs_dir_subject = setup_directories(results_dir)

        # Set up logging
        log_file = os.path.join(results_dir, f'log_test.txt')
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        # Determine model type
        model_type, in_channels, out_channels = get_model_type(experiment_name)
        
        # Check if name_pre_extension is non empty
        if len(name_pre_extension) > 1:
            suffix_data = name_pre_extension[i]
        else:
            name_pre = name_pre_extension[0]
            suffix_data = name_pre + experiment_name.split('2Dslice_')[1].split('_decreased_interpolation_factor_cube_3')[0]
        

        logging.info('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        logging.info('Model being processed: {}'.format(model_name))
        logging.info('Experiment name: {}'.format(experiment_name))
        logging.info('Data suffic: {}'.format(suffix_data))
        logging.info('Name pre-extension: {}'.format(name_pre_extension))
        logging.info('Preprocess method: {}'.format(preprocess_method))
        logging.info('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

    
        # Initialize model
        model, device = load_model(experiment_name, best_exp_path, in_channels, out_channels)
        

        healthy_scores = []
        anomalous_scores = []
        # We keep the region based levels (by quadrants)
        region_based_scores = []
        # We keep the region based levels (by quadrants) from Inselspital
        region_based_labels = []

        
        # Initialize counters
        healthy_idx, anomalous_idx = 0, 0
        spatial_size_z = 64
        
        
        # Load data
        config['preprocess_method'] = preprocess_method
        data_dict = load_data(
            sys_config=config_sys, config=config, idx_start_tr=0, idx_end_tr=1, 
            idx_start_vl=0, idx_end_vl=1, idx_start_ts=idx_start_ts, idx_end_ts=idx_end_ts, 
            with_test_labels=True, suffix=suffix_data
        )
        images_test = data_dict['images_test']
        labels_test = data_dict['labels_test']
        rotation_matrix = data_dict.get('rotation_test', None)
        
        # Get the names of the test subjects
        if experiment_name.__contains__('balanced'):
            suffix_data_path = '_balanced'
        test_names = filter_subjects(config['seg_data_path'], experiment_name, suffix=suffix_data_path)

        logging.info('============================================================')
        logging.info('Test subjects: {}'.format(test_names))
        logging.info('============================================================')

        # Load the dictionary of the subjects, containing labels, sex, age
        subject_dict = load_subject_dict(os.path.join(config['subject_dict_path'], 'subject_dict.pkl'))
        # Load the dictionary of the Inselspital validation
        inselspital_validation_dict = load_subject_dict(os.path.join(config['subject_dict_path'], 'inselspital_validation_dict.pkl'))

        
        # Keep counter for subjects that have region based validations
        counter_region_based_validation = 0
        
        subject_indexes = range(np.int16(images_test.shape[0]/spatial_size_z))
        
        for subject_idx in subject_indexes:

            subject_name = test_names[subject_idx] # includes underscore
            logging.info('Processing subject {}'.format(subject_name))
            start_idx = 0
            end_idx = batch_size

            # Load mask quadrants
            mask_quadrants = np.load(os.path.join(config['quadrants_between_axes_path'], f'{subject_name}_between_axes_masks.npy'))

            # The metadata may it be compressed sensing or not will be the same 
            if subject_name[-1] == '_':
                subject_dict[subject_name] = subject_dict[subject_name[:-1]].copy()


            # Figure out if comrpessed sensing or not
            compressed_sensing = is_compressed_sensing(subject_name, config=config)

            # Add entry to dictionary about type of acquistion 
            subject_dict[subject_name]['Compressed_sensing'] = compressed_sensing

            subject_anomaly_score = []
            subject_reconstruction = []
            subject_sliced = images_test[subject_idx*spatial_size_z:(subject_idx+1)*spatial_size_z]
            subject_labels = labels_test[subject_idx*spatial_size_z:(subject_idx+1)*spatial_size_z]
            if rotation_matrix is not None:
                rotation_matrix_subject = rotation_matrix[subject_idx*spatial_size_z:(subject_idx+1)*spatial_size_z]
            while end_idx <= spatial_size_z:
                batch = subject_sliced[start_idx:end_idx]
                labels = subject_labels[start_idx:end_idx]
                batch_z_slice = torch.from_numpy(np.arange(start_idx, end_idx)).float().to(device)
                batch = torch.from_numpy(batch).transpose(1,4).transpose(2,4).transpose(3,4).float().to(device)
                rotation_matrix_batch = None
                if rotation_matrix is not None:
                    rotation_matrix_batch = rotation_matrix_subject[start_idx:end_idx]
                    rotation_matrix_batch = torch.from_numpy(rotation_matrix_batch).to(device).float()

                input_dict = {'input_images': batch, 'batch_z_slice':batch_z_slice, 'adjacent_batch_slices':adjacent_batch_slices,'rotation_matrix': rotation_matrix_batch}
                
                output_images, subject_reconstruction = infer_batch(input_dict, model=model, model_type=model_type, subject_reconstruction=subject_reconstruction)
                # Compute anomaly score
                subject_anomaly_score.append(output_images.cpu().detach().numpy())
                # Check if all labels are anomalous
                if np.all(labels == 0):
                    legend = "healthy"
                elif np.all(labels == 1):
                    legend = "anomalous"
                else:
                    raise ValueError("Labels are not all healthy or all anomalous, change batch size")
                start_idx += batch_size
                end_idx += batch_size

            
            file_path_input = os.path.join(inputs_dir_subject, f'{subject_name}_inputs.npy')
            file_path = os.path.join(results_dir_subject, f'{subject_name}_anomaly_scores.npy')
            np.save(file_path, np.concatenate(subject_anomaly_score))
            np.save(file_path_input, subject_sliced)
            if model_type == 'reconstruction-based':
                resconstruction_dir_subject = os.path.join(results_dir_test, "reconstruction")
                make_dir_safely(resconstruction_dir_subject)
                file_path_reconstruction = os.path.join(resconstruction_dir_subject, f'{subject_name}_reconstruction.npy')
                np.save(file_path_reconstruction, np.concatenate(subject_reconstruction))
            
            if config['backtransform_anomaly_scores_bool']:
                if config['backtransform_all'] or subject_name in config['backtransform_list']:
                    backtransform_anomaly_scores(subject_name, subject_anomaly_score, subject_dict, config['img_path'], geometry_path, config['seg_data_path'], results_dir_test, suffix_data_path)
            
            if legend == "healthy":
                healthy_scores.append(np.concatenate(subject_anomaly_score))
                healthy_idx += 1
                    
            else:
                anomalous_scores.append(np.concatenate(subject_anomaly_score))
                anomalous_idx += 1

            logging.info('{}_subject {} anomaly_score: {:.4e} +/- {:.4e}'.format(legend, subject_idx, np.mean(subject_anomaly_score), np.std(subject_anomaly_score)))
            # Save the anomaly scores and standard deviation of the subject in the subject_dict for further analysis
            subject_dict[subject_name]['anomaly_score'] = np.mean(subject_anomaly_score)
            subject_dict[subject_name]['std_anomaly_score'] = np.std(subject_anomaly_score)

            # Evaluate the predictions at the region based level if the subject is in the Inselspital validation set (self-supervised models only)
            if subject_name in inselspital_validation_dict and model_type == 'self-supervised':
                logging.info('Also evaluating predictions at the region based level...')
                region_based_label,region_based_score = evaluate_predictions(mask_quadrants, np.concatenate(subject_anomaly_score), inselspital_validation_dict, subject_name)
                region_based_scores.append(region_based_score)
                region_based_labels.append(region_based_label)
                # Save the labels and scores in the subject_dict for further analysis
                subject_dict[subject_name]['region_based_label'] = region_based_label
                subject_dict[subject_name]['region_based_score'] = region_based_score
                counter_region_based_validation += 1


        # Convert list to set for faster lookup
        test_names_set = set(test_names)

        # Filter the dictionary to only keep individuals in the test set
        filtered_subject_dict = {k: v for k, v in subject_dict.items() if k in test_names_set}

        # Compute some statistics and visulization over the age, sex and group distribution
        # First convert dictionary to pandas dataframe
        subject_df = pd.DataFrame.from_dict(filtered_subject_dict, orient='index')
        seed_experiment = experiment_name.split('_SEED_')[1].split('_')[0]
        subject_df['SEED'] = int(seed_experiment)

        # Compute mean and std for the anomaly scores based on the labels (controls vs cases)
        healthy_scores, anomalous_scores = compute_scores_metrics(healthy_scores, anomalous_scores)

        

        if config['visualize_inference_plots']:
            # Plotting scores
            plot_all_scores(healthy_scores, anomalous_scores, results_dir_test)


        # Compute metrics for original data
        y_true_original, y_scores_original, auc_roc_original = compute_metrics(subject_df, 'Original', results_summary)

        # Handling duplicates: Identify similar subjects and keep the one with 'Compressed_sensing' = True
        def get_base_name(name):
            return name.rstrip('_')
        
        subject_df['Base_Name'] = subject_df.index.map(get_base_name)
        subject_df_removed_duplicates = subject_df.sort_values(by='Compressed_sensing', ascending=False).drop_duplicates(subset='Base_Name', keep='first').drop(columns='Base_Name')
        subject_df_removed_duplicates.to_csv(os.path.join(results_dir, 'subject_df_with_anomaly_scores_removed_duplicated.csv'), index_label='ID')
        subject_df.to_csv(os.path.join(results_dir, 'subject_df_with_anomaly_scores_simple.csv'), index_label='ID')

        y_true_deduplicated, y_scores_deduplicated, auc_roc_deduplicated = compute_metrics(subject_df_removed_duplicates, 'Removed Duplicates', results_summary)

        # Compute permutation test for AUC-ROC
        p_value = permutation_test_auc_roc(y_true_deduplicated, y_scores_deduplicated, n_permutations=1000, random_seed=seed)
        results_summary['p-value AUC-ROC (Permutations)'].append(p_value)
        
        # Compute region based metrics
        if counter_region_based_validation > 0:
            logging.info('Compute region-based metrics:')
            # Metrics for all subjects
            compute_region_metrics(subject_df, 'Region', results_summary)
            # Metrics without duplicates
            compute_region_metrics(subject_df_removed_duplicates, 'Region without dup.', results_summary)

            # Handle subjects without mismatches
            subject_df_without_mismatch = subject_df[~subject_df.index.isin(mismatch_subjects)]
            compute_region_metrics(subject_df_without_mismatch, 'Region without mismatch', results_summary)

            # Metrics without mismatches and duplicates
            subject_df_without_mismatch_deduplicated = subject_df_removed_duplicates[~subject_df_removed_duplicates.index.isin(mismatch_subjects)]
            compute_region_metrics(subject_df_without_mismatch_deduplicated, 'Region without mismatch, without dup.', results_summary)

            # Aggregate region-based scores and labels, compute average precision
            region_based_scores = np.array(region_based_scores)
            region_based_labels = np.array(region_based_labels)
            random_average_precision = np.mean(region_based_labels)
            logging.info(f'Random Average Precision: {random_average_precision:.4f}')
            logging.info(f'Number of subjects with region-based validation: {counter_region_based_validation}/{len(test_names)}')

            # Compute permutation test for AUC-ROC
            p_value_region_based = permutation_test_auc_roc(region_based_labels.flatten(), region_based_scores.flatten(), n_permutations=1000, random_seed=seed)
            results_summary['p-value AUC-ROC region based(Permutations)'].append(p_value_region_based)



        # Compute metrics for each matching scenario
        
        for match_type, matches in matches_scenarios.items():
            matched_pairs, y_true_matched, y_scores_matched, auc_roc_matched = compute_metrics_for_matches(matches, subject_df_removed_duplicates, match_type, results_summary)
            if match_type == "Sex Equal, Age Gap <= 5":
                logging.info('Do permutation test for AUC-ROC')
                p_value_subset = permutation_test_auc_roc(
                    y_true_matched, y_scores_matched, n_permutations=1000, random_seed=seed
                )

                results_summary['p-value Subset AUC-ROC (Permutations)'].append(p_value_subset)


        # Logging that information in the results_summary
        results_summary['Model'].append(model_name)
        results_summary['Preprocess Method'].append(preprocess_method)
        results_summary['SEED'].append(seed_experiment)
        results_summary['Rotation'].append('with_rotation' if experiment_name.__contains__('with_rotation') else 'without_rotation')

        # Check metrics for duplicates (compare performance on same subjects in their CS and non-CS acquisitions)
        
        subject_df = perform_permutation_tests_duplicates(subject_df)

        # Check metrics for compressed sensing and non-compressed sensing acquisitions
        compressed_sensing_df = subject_df[subject_df['Compressed_sensing'] == True]
        non_compressed_sensing_df = subject_df[subject_df['Compressed_sensing'] == False]

        logging.info('\n \n Analysis across compressed sensing acquisitions:')
        # Compute metrics for compressed sensing acquisitions
        y_true_compressed_sensing, y_scores_compressed_sensing, auc_roc_compressed_sensing = compute_metrics(compressed_sensing_df, 'CS', results_summary)

        # Permutation test for AUC-ROC
        p_value_compressed_sensing = permutation_test_auc_roc(y_true_compressed_sensing, y_scores_compressed_sensing, n_permutations=1000, random_seed=seed)
        results_summary['p-value CS AUC-ROC (Permutations)'].append(p_value_compressed_sensing)

        # Compute metrics for non-compressed sensing acquisitions
        logging.info('\n\nAnalysis across non-compressed sensing acquisitions:')
        y_true_non_compressed_sensing, y_scores_non_compressed_sensing, auc_roc_non_compressed_sensing = compute_metrics(non_compressed_sensing_df, 'non-CS', results_summary)

        # Permutation test for AUC-ROC
        p_value_non_compressed_sensing = permutation_test_auc_roc(y_true_non_compressed_sensing, y_scores_non_compressed_sensing, n_permutations=1000, random_seed=seed)
        results_summary['p-value non-CS AUC-ROC (Permutations)'].append(p_value_non_compressed_sensing)
        
        case_subjects = subject_df_removed_duplicates[subject_df_removed_duplicates['Label'] == 'Cases']
        
        # Perform descriptive statistics
        desc_stats = descriptive_stats(subject_df_removed_duplicates)

        # Perform correlation analysis
        logging.info('Analysis across whole test set:')
        _, _,_,_ = correlation_analysis(subject_df_removed_duplicates)

        logging.info('\n Analysis across cases only in test set:')
        pearson_corr, pearson_p_value,spearman_corr,spearman_p_value = correlation_analysis(case_subjects)

        if config['visualize_inference_plots']:
        # Visualize data
            visualize_data(subject_df, results_dir_test)
            subject_df.sort_index(inplace=True)
            # Remove dummy and Age Group columns
            subject_df.drop(columns=['dummy', 'Age Group'], inplace=True)

        # Save the subject_df and name index as ID
        subject_df.to_csv(os.path.join(results_dir, f'subject_df_with_anomaly_scores_{int(seed_experiment)}.csv'), index_label='ID')
        logger.removeHandler(handler)
    
    # Create summary table
    summary_df = pd.DataFrame(results_summary)

    # Log summary table at the end of the log file
    logging.info('Summary:')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        
        logging.info('\n{}'.format(summary_df))
