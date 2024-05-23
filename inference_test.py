# In this we evaluate the model on the test set and save the results
import os
import pickle
import pandas as pd
import glob
import numpy as np
import random
from config import system_eval as config_sys
from helpers.utils import make_dir_safely
from helpers.data_loader import load_data

from models.vae import VAE_convT, ConvWithAux, ConvWithEncDecAux, ConvWithDeepAux, ConvWithDeepEncDecAux, ConvWithDeeperBNEncDecAux, ConvWithDeeperEncDecAux


from inference_experiments_list import experiments_with_cs, experiments_without_cs, experiments_only_cs \
,experiments_only_cs_reconstruction_based, experiments_with_cs_reconstruction_based, experiments_without_cs_reconstruction_based\
,short_experiments_without_cs, short_experiments_with_cs, short_experiments_only_cs \
,matches_scenarios

from helpers.utils_inference import descriptive_stats, compare_groups, correlation_analysis\
    , regression_analysis, visualize_data, plot_scores, filter_subjects, plot_heatmap\
        , plot_heatmap_comparaison, adjust_anomaly_scores,backtransform_anomaly_scores, compute_metrics, compute_metrics_for_matches

import torch

import logging
logging.basicConfig(level=logging.INFO)

from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
# Ignore FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

seed = 42  # you can set to your preferred seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def permutation_test_auc_roc(y_true_deduplicated, y_scores_deduplicated, y_true_matched, y_scores_matched, n_permutations=1000, random_seed=42):
    np.random.seed(random_seed)

    original_auc_roc = roc_auc_score(y_true_deduplicated, y_scores_deduplicated)
    matched_auc_roc = roc_auc_score(y_true_matched, y_scores_matched)

    # Permutation test for original AUC-ROC
    permuted_aucs_orig = np.array([roc_auc_score(y_true_deduplicated, np.random.permutation(y_scores_deduplicated)) for _ in range(n_permutations -1)])
    p_value_orig = (np.sum(permuted_aucs_orig >= original_auc_roc) + 1) / (n_permutations -1)
    p_value_orig_vs_matched = (np.sum(permuted_aucs_orig >= matched_auc_roc) + 1) / (n_permutations -1)

    # Permutation test for matched AUC-ROC
    permuted_aucs_matched = np.array([roc_auc_score(y_true_matched, np.random.permutation(y_scores_matched)) for _ in range(n_permutations -1)])
    p_value_matched = (np.sum(permuted_aucs_matched >= matched_auc_roc) + 1) / (n_permutations -1)
    p_value_matched_vs_orig = (np.sum(permuted_aucs_matched >= original_auc_roc) + 1) / (n_permutations -1)

    logging.info(f"Removed Duplicate Model AUC-ROC: {original_auc_roc:.4f}, p-value: {p_value_orig:.4f}")
    logging.info(f"Matching Subset Model AUC-ROC: {matched_auc_roc:.4f}, p-value (vs permuted original): {p_value_orig_vs_matched:.4f}")
    logging.info(f"Matching Subset Model AUC-ROC: {matched_auc_roc:.4f}, p-value (vs permuted matched): {p_value_matched:.4f}")
    logging.info(f"Removed Duplicate Model AUC-ROC: {original_auc_roc:.4f}, p-value (vs permuted matched): {p_value_matched_vs_orig:.4f}")

    return p_value_orig, p_value_orig_vs_matched, p_value_matched, p_value_matched_vs_orig




models_dir = "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/logs"
data_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/final_segmentations'
subject_dict_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/preprocessed'

project_data_root = config_sys.project_data_root
project_code_root = config_sys.project_code_root
# Path to the raw images
img_path = os.path.join(project_data_root, f'preprocessed/patients/numpy')
# Path to the geometry information
geometry_path = os.path.join(project_code_root, f'data/geometry_for_backtransformation')


# Choose the appropriate dictionary
# [experiments_only_cs, experiments_without_cs, experiments_with_cs, experiments_only_cs_reconstruction_based, experiments_without_cs_reconstruction_based, experiments_with_cs_reconstruction_based]
chosen_dictionary = short_experiments_with_cs

# Whether to backtransform the anomaly scores
backtransform_anomaly_scores_bool = False 
backtransform_all = False
backtransform_list = ['MACDAVD_204_', 'MACDAVD_101_', 'MACDAVD_311_']

name_pre_extension = ['']

# Define idx_start_ts and idx_end_ts based on the chosen dictionary
if chosen_dictionary == experiments_only_cs or chosen_dictionary == short_experiments_only_cs or chosen_dictionary == experiments_only_cs_reconstruction_based:
    idx_start_ts = 0
    idx_end_ts = 17 # 17 originally
elif chosen_dictionary == experiments_without_cs or chosen_dictionary == short_experiments_without_cs or chosen_dictionary == experiments_without_cs_reconstruction_based:
    idx_start_ts = 0
    idx_end_ts = 34 # 34 originally
elif chosen_dictionary == experiments_with_cs or chosen_dictionary == short_experiments_with_cs or chosen_dictionary == experiments_with_cs_reconstruction_based:
    idx_start_ts = 0
    idx_end_ts = 54
else:
    raise ValueError("Invalid dictionary chosen")

if __name__ == '__main__':
    adjacent_batch_slices = None
    batch_size = 32 
    
    
    for i, (list_of_experiments_with_rotation, list_of_experiments_without_rotation) in enumerate(zip(chosen_dictionary['with_rotation'], chosen_dictionary['without_rotation'])):
        # Progress bar
        logging.info('Processing experiment {}/{}'.format(i+1, len(chosen_dictionary['with_rotation'])))
        list_of_experiments_paths = list_of_experiments_with_rotation + list_of_experiments_without_rotation
        # Initialize dictionary to store results
        #results_summary = {'Model': [], 'Preprocess Method': [], 'SEED': [], 'AUC-ROC': [], 'Average Precision': [], 'Rotation': []}

        results_summary = {
        'Model': [],
        'Preprocess Method': [],
        'SEED': [],
        'Rotation': [],
        'p-value Removed Duplicate AUC-ROC (Permutations)': [],
        'p-value Removed Duplicate AUC-ROC (Compared to subset)': [],
        'p-value Subset AUC-ROC (Permutations)': [],
        'p-value Subset AUC-ROC (Compared to original)': []
    }
     # Function to ensure keys are initialized
        def initialize_result_keys(results_summary, labels):
            for label in labels:
                if f'{label} AUC-ROC' not in results_summary:
                    results_summary[f'{label} AUC-ROC'] = []
                if f'{label} Average Precision' not in results_summary:
                    results_summary[f'{label} Average Precision'] = []

        # Initialize keys
        initialize_result_keys(results_summary, ['Original', 'Removed Duplicates'] + list(matches_scenarios.keys()))
        logging.info('List of experiments paths: {}'.format(list_of_experiments_paths))
     
        for i, exp_rel_path in enumerate(list_of_experiments_paths):
            suffix_data_path = ''

            if not exp_rel_path:
                continue
            
            exp_path = os.path.join(models_dir, exp_rel_path)
            pattern = os.path.join(exp_path, "*best*")
            best_exp_path = glob.glob(pattern)[0]
            model_str = exp_rel_path.split("/")[0]
            
            preprocess_method = exp_rel_path.split("/")[1]
            exp_name = exp_rel_path.split("/")[-1]
            project_code_root = config_sys.project_code_root
            results_dir = os.path.join(project_code_root,'Results/Evaluation/' + model_str + '/' + preprocess_method + '/' + exp_name)
            
            make_dir_safely(results_dir)        
            results_dir_test = results_dir + '/' + f'test'
            make_dir_safely(results_dir_test)
            log_file = os.path.join(results_dir, f'log_test.txt')
            logger = logging.getLogger()
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            logging.info('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            # Logging the model being processed
            logging.info('Model being processed: {}'.format(exp_rel_path.split('/')[0]))
            logging.info('Experiment name: {}'.format(exp_rel_path.split('/')[-1]))
            logging.info('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

            logging.info('name pre extension: {}'.format(name_pre_extension))


            # Check if self-supervised or reconstruction based
            if exp_rel_path.__contains__('SSL'):
                model_type = 'self-supervised'
                in_channels = 4
                out_channels =1 
            else:
                model_type = 'reconstruction-based'
                in_channels = 4
                out_channels = 4

            

            # Check if name_pre_extension is non empty
            if len(name_pre_extension) > 1:
                name_extension = name_pre_extension[i]
            else:
                name_pre = name_pre_extension[0]
                name_extension = name_pre + exp_name.split('2Dslice_')[1].split('_decreased_interpolation_factor_cube_3')[0]
            
            
            
            # Excepetionally else one above
            #name_extension = name_pre_extension[i]
            logging.info('name_extension: {}'.format(name_extension))

            if exp_name.__contains__('vae_convT'):
                model = VAE_convT(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
            elif exp_name.__contains__('deeper_conv_enc_dec'):
                model = ConvWithDeeperEncDecAux(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
            elif exp_name.__contains__('deeper_bn_conv_enc_dec'):
                model = ConvWithDeeperBNEncDecAux(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
            elif exp_name.__contains__('deep_conv_with_aux'):
                model = ConvWithDeepAux(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
            elif exp_name.__contains__('deep_conv_enc_dec'):
                model = ConvWithDeepEncDecAux(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
            elif exp_name.__contains__('conv_with_aux'):
                model = ConvWithAux(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
            elif exp_name.__contains__('conv_enc_dec'):
                model = ConvWithEncDecAux(in_channels=in_channels, out_channels=out_channels, gf_dim=8)

            else:
                raise ValueError('Exp name {} has no model recognized'.format(exp_name))
            # Load the model onto device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(best_exp_path, map_location=device))
            model.to(device)
            model.eval()
            

            healthy_scores = []
            anomalous_scores = []
            healthy_idx = 0
            anomalous_idx = 0
            spatial_size_z = 64
            preprocess_method = exp_rel_path.split("/")[1]
            config = {'preprocess_method': preprocess_method}

            
            data_dict = load_data(sys_config=config_sys, config=config, idx_start_tr=0, idx_end_tr=1, idx_start_vl=0, idx_end_vl=1,idx_start_ts=idx_start_ts, idx_end_ts=idx_end_ts, with_test_labels= True, suffix = name_extension)
            images_test = data_dict['images_test']
            labels_test = data_dict['labels_test']
            rotation_matrix = data_dict.get('rotation_test', None)
            
            # Get the names of the test subjects
            if exp_rel_path.__contains__('balanced'):
                suffix_data_path = '_balanced'
            test_names = filter_subjects(data_path, exp_rel_path, suffix=suffix_data_path)


            logging.info('============================================================')
            # Print test names
            logging.info('Test subjects: {}'.format(test_names))
            logging.info('============================================================')

            # Load the dictionary of the subjects, containing labels, sex, age

            with open(os.path.join(subject_dict_path, 'subject_dict.pkl'), 'rb') as f:
                subject_dict = pickle.load(f)
            
            
            subject_indexes = range(np.int16(images_test.shape[0]/spatial_size_z))
            
            for subject_idx in subject_indexes:

                subject_name = test_names[subject_idx] # includes underscore
                logging.info('Processing subject {}'.format(subject_name))
                start_idx = 0
                end_idx = batch_size

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
                        backtransform_anomaly_scores(subject_name, subject_anomaly_score, subject_dict, img_path, geometry_path, backtransform_list, results_dir_test)
                
                
                
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

            # Plotting scores
            logging.info('Plotting scores through slices...')
            
            plot_scores(healthy_scores, anomalous_scores, results_dir_test, level = 'patient')
            plot_scores(healthy_scores, anomalous_scores, results_dir_test, level = 'imagewise')
            logging.info('Plotting scores through time...')
            plot_scores(healthy_scores, anomalous_scores, results_dir_test, level = 'imagewise', through_time= True)

            logging.info('Plotting scores through slices and time...')
            
            plot_heatmap(healthy_scores, results_dir_test, name_extension = 'healthy')
            plot_heatmap(anomalous_scores, results_dir_test, name_extension = 'anomalous')
            plot_heatmap(healthy_scores, results_dir_test, name_extension = 'healthy_scale_colors', scale_colors= True)
            plot_heatmap(anomalous_scores, results_dir_test, name_extension = 'anomalous_scale_colors', scale_colors= True)

            plot_heatmap_comparaison(healthy_scores, anomalous_scores, results_dir_test)

            

            # Convert list to set for faster lookup
            test_names_set = set(test_names)

            # Filter the dictionary to only keep individuals in the test set
            filtered_subject_dict = {k: v for k, v in subject_dict.items() if k in test_names_set}

            # Compute some statistics and visulization over the age, sex and group distribution
            # First convert dictionary to pandas dataframe
            subject_df = pd.DataFrame.from_dict(filtered_subject_dict, orient='index')
            
            # Save the subject_df and name index as ID
            subject_df.to_csv(os.path.join(results_dir, 'subject_df_with_anomaly_scores.csv'), index_label='ID')

            # Compute metrics for original data
            y_true_original, y_scores_original, auc_roc_original = compute_metrics(subject_df, 'Original', results_summary)

            # Identify similar subjects and keep the one with 'Compressed_sensing' = True
            def get_base_name(name):
                return name.rstrip('_')

            subject_df['Base_Name'] = subject_df.index.map(get_base_name)
            subject_df_removed_duplicates = subject_df.sort_values(by='Compressed_sensing', ascending=False).drop_duplicates(subset='Base_Name', keep='first').drop(columns='Base_Name')
            subject_df_removed_duplicates.to_csv(os.path.join(results_dir, 'subject_df_with_anomaly_scores_removed_duplicated.csv'), index_label='ID')

            y_true_deduplicated, y_scores_deduplicated, auc_roc_deduplicated = compute_metrics(subject_df_removed_duplicates, 'Removed Duplicates', results_summary)
            # Compute metrics for each matching scenario
            
            for match_type, matches in matches_scenarios.items():
                matched_pairs, y_true_matched, y_scores_matched, auc_roc_matched = compute_metrics_for_matches(matches, subject_df_removed_duplicates, match_type, results_summary)
                if match_type == "Sex Equal, Age Gap <= 5":
                    logging.info('Do permutation test for AUC-ROC')
                    p_value_orig, p_value_orig_vs_matched, p_value_matched, p_value_matched_vs_orig = permutation_test_auc_roc(
                        y_true_deduplicated, y_scores_deduplicated, y_true_matched, y_scores_matched, n_permutations=1000, random_seed=seed
                    )

                    results_summary['p-value Removed Duplicate AUC-ROC (Permutations)'].append(p_value_orig)
                    results_summary['p-value Removed Duplicate AUC-ROC (Compared to subset)'].append(p_value_orig_vs_matched)
                    results_summary['p-value Subset AUC-ROC (Permutations)'].append(p_value_matched)
                    results_summary['p-value Subset AUC-ROC (Compared to original)'].append(p_value_matched_vs_orig)


                
            results_summary['Model'].append(exp_rel_path.split('/')[0])
            results_summary['Preprocess Method'].append(exp_rel_path.split('/')[1])
            results_summary['SEED'].append(exp_rel_path.split('_SEED_')[1].split('_')[0])
            results_summary['Rotation'].append('with_rotation' if exp_rel_path.__contains__('with_rotation') else 'without_rotation')

            
            case_subjects = subject_df_removed_duplicates[subject_df_removed_duplicates['Label'] == 'Cases']
            

            
            
            # Perform descriptive statistics
            desc_stats = descriptive_stats(subject_df_removed_duplicates)


            # Compare anomaly scores between groups
            p_val = compare_groups(subject_df_removed_duplicates)
            

            # Perform correlation analysis
            logging.info('Analysis across whole test set:')
            _, _,_,_ = correlation_analysis(subject_df_removed_duplicates)

            logging.info('\n Analysis across cases only in test set:')
            pearson_corr, pearson_p_value,spearman_corr,spearman_p_value = correlation_analysis(case_subjects)

    
            # Perform regression analysis
            mse, r2, intercept, coefficients = regression_analysis(subject_df_removed_duplicates)

        
            # Visualize data
            visualize_data(subject_df_removed_duplicates, results_dir_test)


            logger.removeHandler(handler)
        
        # Create summary table
        summary_df = pd.DataFrame(results_summary)

        # Log summary table at the end of the log file
        logging.info('Summary:')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            
            logging.info('\n{}'.format(summary_df))
