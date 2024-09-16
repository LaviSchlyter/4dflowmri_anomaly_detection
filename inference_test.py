import os
import logging
import pandas as pd
import glob
import numpy as np
import random
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from config import system_eval as config_sys
from helpers.utils import make_dir_safely
from helpers.data_loader import load_data

from inference_experiments_list import (
    experiments_with_cs, experiments_without_cs, experiments_only_cs,
    experiments_only_cs_reconstruction_based, experiments_with_cs_reconstruction_based,
    experiments_without_cs_reconstruction_based, short_experiments_without_cs,
    short_experiments_with_cs, short_experiments_only_cs, matches_scenarios
)
from helpers.utils_inference import (
    descriptive_stats, correlation_analysis, visualize_data, plot_scores, 
    filter_subjects, plot_heatmap, plot_heatmap_comparaison, backtransform_anomaly_scores,
    compute_metrics, compute_metrics_for_matches, permutation_test_auc_roc, 
    evaluate_predictions, perform_permutation_tests_duplicates, initialize_result_keys,
    get_model_type, initialize_model, load_subject_dict
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


# Define paths and constants
models_dir = "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/Saved_models"
data_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/final_segmentations'
subject_dict_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/preprocessed'
quadrants_between_axes_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/data/quadrants_between_axes'
project_data_root = config_sys.project_data_root
project_code_root = config_sys.project_code_root
# Path to the raw images
img_path = os.path.join(project_data_root, f'preprocessed/patients/numpy')
# Path to the geometry information
geometry_path = os.path.join(project_code_root, f'data/geometry_for_backtransformation')

#==============================================================================
# Define experimental settings
#==============================================================================
chosen_dictionary = short_experiments_with_cs # experiments_only_cs, experiments_without_cs, experiments_with_cs, experiments_only_cs_reconstruction_based, experiments_without_cs_reconstruction_based, experiments_with_cs_reconstruction_based
backtransform_anomaly_scores_bool = False
backtransform_all = False
backtransform_list = [
    'MACDAVD_137_', 'MACDAVD_137', 'MACDAVD_131', 'MACDAVD_131_', 'MACDAVD_135_',
    'MACDAVD_135', 'MACDAVD_133_', 'MACDAVD_133', 'MACDAVD_143_', 'MACDAVD_143', 
    'MACDAVD_206_', 'MACDAVD_206'
]

# Subjects that don't have a match between Inselspital and the original dataset 
mismatch_subjects = ['MACDAVD_131', 'MACDAVD_159_', 'MACDAVD_202_']
name_pre_extension = ['']
visualize_inference_plots = False

# Define time slice indices based on the chosen dictionary
if chosen_dictionary in [experiments_only_cs, short_experiments_only_cs, experiments_only_cs_reconstruction_based]:
    idx_start_ts, idx_end_ts = 0, 17
elif chosen_dictionary in [experiments_without_cs, short_experiments_without_cs, experiments_without_cs_reconstruction_based]:
    idx_start_ts, idx_end_ts = 0, 34
elif chosen_dictionary in [experiments_with_cs, short_experiments_with_cs, experiments_with_cs_reconstruction_based]:
    idx_start_ts, idx_end_ts = 0, 54
else:
    raise ValueError("Invalid dictionary chosen")


if __name__ == '__main__':
    adjacent_batch_slices = None
    batch_size = 32 

    
    for i, (list_of_experiments_without_rotation, list_of_experiments_with_rotation) in enumerate(zip(chosen_dictionary.get('without_rotation', [[]]), chosen_dictionary.get('with_rotation', [[]]))):
        # Progress bar

        list_of_experiments_paths = list_of_experiments_without_rotation + list_of_experiments_with_rotation
        logging.info('Processing experiment {}/{}'.format(i+1, len(list_of_experiments_paths)))
        
        # Initialize dictionary to store results

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
        
        # Initialize keys
        initialize_result_keys(results_summary, ['Original', 'Removed Duplicates', 'Region', 'Region without dup.', 'Region without mismatch', 'Region without mismatch, without dup.', 'CS', 'non-CS' ] + list(matches_scenarios.keys()))
        logging.info('List of experiments paths: {}'.format(list_of_experiments_paths))
     
        for i, exp_rel_path in enumerate(list_of_experiments_paths):
            suffix_data_path = ''

            if not exp_rel_path:
                continue
            
            # Set up experiment paths
            exp_path = os.path.join(models_dir, exp_rel_path)
            pattern = os.path.join(exp_path, "*best*")
            best_exp_path = glob.glob(pattern)[0]
            model_str = exp_rel_path.split("/")[0]
            preprocess_method = exp_rel_path.split("/")[1]
            exp_name = exp_rel_path.split("/")[-1]
            results_dir = os.path.join(project_code_root,'Results/Evaluation/' + model_str + '/' + preprocess_method + '/' + exp_name)
            
            make_dir_safely(results_dir)        
            results_dir_test = results_dir + '/' + f'test'
            make_dir_safely(results_dir_test)

            # Set up logging
            log_file = os.path.join(results_dir, f'log_test.txt')
            logger = logging.getLogger()
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)

            # Determine model type
            model_type, in_channels, out_channels = get_model_type(exp_rel_path)
            
            # Check if name_pre_extension is non empty
            if len(name_pre_extension) > 1:
                suffix_data = name_pre_extension[i]
            else:
                name_pre = name_pre_extension[0]
                suffix_data = name_pre + exp_name.split('2Dslice_')[1].split('_decreased_interpolation_factor_cube_3')[0]
            

            logging.info('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            logging.info('Model being processed: {}'.format(exp_rel_path.split('/')[0]))
            logging.info('Experiment name: {}'.format(exp_rel_path.split('/')[-1]))
            logging.info('Data suffic: {}'.format(suffix_data))
            logging.info('Name pre-extension: {}'.format(name_pre_extension))
            logging.info('Preprocess method: {}'.format(preprocess_method))
            logging.info('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

            # Initialize model
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = initialize_model(exp_name, in_channels, out_channels)
            model.load_state_dict(torch.load(best_exp_path, map_location=device))
            model.to(device)
            model.eval()
            

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
            config = {'preprocess_method': preprocess_method}
            data_dict = load_data(
                sys_config=config_sys, config=config, idx_start_tr=0, idx_end_tr=1, 
                idx_start_vl=0, idx_end_vl=1, idx_start_ts=idx_start_ts, idx_end_ts=idx_end_ts, 
                with_test_labels=True, suffix=suffix_data
            )
            images_test = data_dict['images_test']
            labels_test = data_dict['labels_test']
            rotation_matrix = data_dict.get('rotation_test', None)
            
            # Get the names of the test subjects
            if exp_rel_path.__contains__('balanced'):
                suffix_data_path = '_balanced'
            test_names = filter_subjects(data_path, exp_rel_path, suffix=suffix_data_path)

            logging.info('============================================================')
            logging.info('Test subjects: {}'.format(test_names))
            logging.info('============================================================')

            # Load the dictionary of the subjects, containing labels, sex, age
            subject_dict = load_subject_dict(os.path.join(subject_dict_path, 'subject_dict.pkl'))
            # Load the dictionary of the Inselspital validation
            inselspital_validation_dict = load_subject_dict(os.path.join(subject_dict_path, 'inselspital_validation_dict.pkl'))

            
            # Keep counter for subjects that have region based validations
            counter_region_based_validation = 0
            
            subject_indexes = range(np.int16(images_test.shape[0]/spatial_size_z))
            
            for subject_idx in subject_indexes:

                subject_name = test_names[subject_idx] # includes underscore
                logging.info('Processing subject {}'.format(subject_name))
                start_idx = 0
                end_idx = batch_size

                # Load mask quadrants
                mask_quadrants = np.load(os.path.join(quadrants_between_axes_path, f'{subject_name}_between_axes_masks.npy'))

                # The metadata may it be compressed sensing or not will be the same 
                if subject_name[-1] == '_':
                    subject_dict[subject_name] = subject_dict[subject_name[:-1]].copy()

                # Figure out if comrpessed sensing or not
                subject_name_npy = subject_name+'.npy'
                if subject_name.__contains__('MACDAVD_1'):
                    
                    # Control 
                    if subject_name_npy in os.listdir(os.path.join(img_path.replace("patients", "controls"))):
                        compressed_sensing = False
                    elif subject_name_npy in os.listdir(os.path.join(img_path.replace("patients", "controls")+'_compressed_sensing')):
                        compressed_sensing = True
                    else:
                        raise ValueError("Cannot find the individual")
                else:
                    if subject_name_npy in os.listdir(os.path.join(img_path)):
                        compressed_sensing = False
                    elif subject_name_npy in os.listdir(os.path.join(img_path+'_compressed_sensing')):
                        compressed_sensing = True
                    else:
                        raise ValueError("Cannot find the individual")

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
                    with torch.no_grad():
                        model.eval()
                        
                        input_dict = {'input_images': batch, 'batch_z_slice':batch_z_slice, 'adjacent_batch_slices':adjacent_batch_slices,'rotation_matrix': rotation_matrix_batch}
                        output_dict = model(input_dict)

                        if model_type == 'self-supervised':
                            
                            output_images = torch.sigmoid(output_dict['decoder_output'])
                        else:
                            logging.info('Reconstruction based model')
                            # Reconstruction based
                            model_output = output_dict['decoder_output']
                            output_images = torch.abs(model_output - batch)
                            subject_reconstruction.append(output_images.cpu().detach().numpy())

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

                results_dir_subject = os.path.join(results_dir_test, "outputs")
                inputs_dir_subject = os.path.join(results_dir_test, "inputs")
                make_dir_safely(inputs_dir_subject)
                make_dir_safely(results_dir_subject)
                file_path_input = os.path.join(inputs_dir_subject, f'{subject_name}_inputs.npy')
                file_path = os.path.join(results_dir_subject, f'{subject_name}_anomaly_scores.npy')
                np.save(file_path, np.concatenate(subject_anomaly_score))
                np.save(file_path_input, subject_sliced)
                if model_type == 'reconstruction-based':
                    resconstruction_dir_subject = os.path.join(results_dir_test, "reconstruction")
                    make_dir_safely(resconstruction_dir_subject)
                    file_path_reconstruction = os.path.join(resconstruction_dir_subject, f'{subject_name}_reconstruction.npy')
                    np.save(file_path_reconstruction, np.concatenate(subject_reconstruction))
                
                if backtransform_anomaly_scores_bool:
                    if backtransform_all or subject_name in backtransform_list:
                        backtransform_anomaly_scores(subject_name, subject_anomaly_score, subject_dict, img_path, geometry_path, data_path, results_dir_test, suffix_data_path)
                
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
            seed_experiment = exp_rel_path.split('_SEED_')[1].split('_')[0]
            subject_df['SEED'] = int(seed_experiment)

            healthy_scores = np.array(healthy_scores)
            # healthy scores [#subjects, z slices, c, x,y,t]
            healthy_mean_anomaly_score = np.mean(healthy_scores)
            healthy_std_anomaly_score = np.std(healthy_scores)

            anomalous_scores = np.array(anomalous_scores)
            anomalous_mean_anomaly_score = np.mean(anomalous_scores)
            anomalous_std_anomaly_score = np.std(anomalous_scores)

            logging.info('============================================================')
            logging.info('Control subjects anomaly_score: {} +/- {:.4e}'.format(healthy_mean_anomaly_score, healthy_std_anomaly_score))
            logging.info('Anomalous subjects anomaly_score: {} +/- {:.4e}'.format(anomalous_mean_anomaly_score, anomalous_std_anomaly_score))
            logging.info('============================================================')

            if visualize_inference_plots:
                # Plotting scores
                logging.info('Plotting scores through slices...')
                
                plot_scores(healthy_scores, anomalous_scores, results_dir_test, level = 'patient')
                plot_scores(healthy_scores, anomalous_scores, results_dir_test, level = 'imagewise')
                logging.info('Plotting scores through time...')
                plot_scores(healthy_scores, anomalous_scores, results_dir_test, level = 'imagewise', through_time= True)

                #logging.info('Plotting scores through slices and time...')
                #
                #plot_heatmap(healthy_scores, results_dir_test, suffix_data = 'healthy')
                #plot_heatmap(anomalous_scores, results_dir_test, suffix_data = 'anomalous')
                #plot_heatmap(healthy_scores, results_dir_test, suffix_data = 'healthy_scale_colors', scale_colors= True)
                #plot_heatmap(anomalous_scores, results_dir_test, suffix_data = 'anomalous_scale_colors', scale_colors= True)

                #plot_heatmap_comparaison(healthy_scores, anomalous_scores, results_dir_test)

            

            # Compute metrics for original data
            y_true_original, y_scores_original, auc_roc_original = compute_metrics(subject_df, 'Original', results_summary)

            # Identify similar subjects and keep the one with 'Compressed_sensing' = True
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
                logging.info('\n Compute region based metrics:')
                # Compute region based metrics
                y_true_region_based, y_scores_region_based, auc_roc_region_based = compute_metrics(subject_df, 'Region', results_summary)
                # Compute region based metrics without duplicates
                y_true_region_based_deduplicated, y_scores_region_based_deduplicated, auc_roc_region_based_deduplicated = compute_metrics(subject_df_removed_duplicates, 'Region without dup.', results_summary)
                # Compute region based metrics without mismatch (subjects that have a mismatch of prognosis between Inselspital and the original dataset)
                subject_df_without_mismatch = subject_df[~subject_df.index.isin(mismatch_subjects)]
                y_true_region_based_without_mismatch, y_scores_region_based_without_mismatch, auc_roc_region_based_without_mismatch = compute_metrics(subject_df_without_mismatch, 'Region without mismatch', results_summary)
                # Compute region based metrics without mismatch and duplicates
                subject_df_without_mismatch_deduplicated = subject_df_removed_duplicates[~subject_df_removed_duplicates.index.isin(mismatch_subjects)]
                y_true_region_based_without_mismatch_deduplicated, y_scores_region_based_without_mismatch_deduplicated, auc_roc_region_based_without_mismatch_deduplicated = compute_metrics(subject_df_without_mismatch_deduplicated, 'Region without mismatch, without dup.', results_summary)
                region_based_scores = np.array(region_based_scores)
                region_based_labels = np.array(region_based_labels)
                random_average_precision = np.sum(region_based_labels.flatten())/len(region_based_labels.flatten())
                logging.info('Random Average Precision: {:.4f}'.format(random_average_precision))
                logging.info('Number of subjects with region based validation: {}/{}'.format(counter_region_based_validation, len(test_names)))
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
            results_summary['Model'].append(exp_rel_path.split('/')[0])
            results_summary['Preprocess Method'].append(exp_rel_path.split('/')[1])
            results_summary['SEED'].append(seed_experiment)
            results_summary['Rotation'].append('with_rotation' if exp_rel_path.__contains__('with_rotation') else 'without_rotation')

            # Check metrics for duplicates (compare performance on same subjects in their CS and non-CS acquisitions)
            
            subject_df = perform_permutation_tests_duplicates(subject_df)

            # Check metrics for compressed sensing and non-compressed sensing acquisitions
            compressed_sensing_df = subject_df[subject_df['Compressed_sensing'] == True]
            non_compressed_sensing_df = subject_df[subject_df['Compressed_sensing'] == False]

            logging.info('Analysis across compressed sensing acquisitions:')
            # Compute metrics for compressed sensing acquisitions
            y_true_compressed_sensing, y_scores_compressed_sensing, auc_roc_compressed_sensing = compute_metrics(compressed_sensing_df, 'CS', results_summary)

            # Permutation test for AUC-ROC
            p_value_compressed_sensing = permutation_test_auc_roc(y_true_compressed_sensing, y_scores_compressed_sensing, n_permutations=1000, random_seed=seed)
            results_summary['p-value CS AUC-ROC (Permutations)'].append(p_value_compressed_sensing)

            # Compute metrics for non-compressed sensing acquisitions
            logging.info('Analysis across non-compressed sensing acquisitions:')
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

            if visualize_inference_plots:
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
