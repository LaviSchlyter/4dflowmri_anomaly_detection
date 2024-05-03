# In this we evaluate the model on the test set and save the results
import os
import torch
import warnings
# Ignore FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO)
import pickle
import pandas as pd
import glob
import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from config import system_eval as config_sys
import config.system as sys_config
from helpers.utils import make_dir_safely
from helpers.data_loader import load_data
from models.vae import VAE_convT, ConvWithAux, ConvWithEncDecAux, ConvWithDeepAux, ConvWithDeepEncDecAux, ConvWithDeeperBNEncDecAux, ConvWithDeeperEncDecAux
from inference_experiments_list import experiments_with_cs, experiments_without_cs, experiments_only_cs
from helpers.utils_inference import descriptive_stats, compare_groups, correlation_analysis, regression_analysis, visualize_data, plot_scores, filter_subjects

seed = 42  # you can set to your preferred seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


models_dir = "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/logs"
data_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/final_segmentations'
subject_dict_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/preprocessed'


# Choose the appropriate dictionary
chosen_dictionary = experiments_with_cs  # [experiments_only_cs, experiments_without_cs, experiments_with_cs]
name_pre_extension = ['']

# Define idx_start_ts and idx_end_ts based on the chosen dictionary
if chosen_dictionary == experiments_only_cs:
    idx_start_ts = 0
    idx_end_ts = 17
elif chosen_dictionary == experiments_without_cs:
    idx_start_ts = 0
    idx_end_ts = 34
elif chosen_dictionary == experiments_with_cs:
    idx_start_ts = 0
    idx_end_ts = 54
else:
    raise ValueError("Invalid dictionary chosen")


if __name__ == '__main__':
    adjacent_batch_slices = None
    batch_size = 32 
    # Initialize dictionary to store results
    

    for i, (list_of_experiments_with_rotation, list_of_experiments_without_rotation) in enumerate(zip(chosen_dictionary['with_rotation'], chosen_dictionary['without_rotation'])):

        # Progress bar
        logging.info('Processing experiment {}/{}'.format(i+1, len(chosen_dictionary['with_rotation'])))
        results_summary = {'Model': [], 'Preprocess Method': [], 'SEED': [], 'AUC-ROC': [], 'Average Precision': [], 'Rotation': []}
        list_of_experiments_paths = list_of_experiments_with_rotation + list_of_experiments_without_rotation
        logging.info('List of experiments paths: {}'.format(list_of_experiments_paths))
        

    
        for i, exp_rel_path in enumerate(list_of_experiments_paths):

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
            test_names = filter_subjects(data_path, exp_rel_path)


            logging.info('============================================================')
            # Print test names
            logging.info('Test subjects: {}'.format(test_names))
            logging.info('============================================================')

            # Load the dictionary of the subjects, containing labels, sex, age

            with open(os.path.join(subject_dict_path, 'subject_dict.pkl'), 'rb') as f:
                subject_dict = pickle.load(f)
            
            
            subject_indexes = range(np.int16(images_test.shape[0]/spatial_size_z))
            
            for subject_idx in subject_indexes:

                subject_name = test_names[subject_idx]
                logging.info('Processing subject {}'.format(subject_name))
                start_idx = 0
                end_idx = batch_size

                

                # Stip the last underscore of subject name if existing
                if subject_name.endswith('_'):
                    subject_name = subject_name.rstrip('_')
                
                
                
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

                        output_images = torch.sigmoid(output_dict['decoder_output'])
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
            #plot_scores(healthy_scores, anomalous_scores, results_dir_test, level = 'patient', through_time= True)
            plot_scores(healthy_scores, anomalous_scores, results_dir_test, level = 'imagewise', through_time= True)



            # Compute AUC-ROC
            anomalous_scores_patient = np.mean(anomalous_scores, axis=(1,2,3,4,5))
            healthy_scores_patient = np.mean(healthy_scores, axis=(1,2,3,4,5))
            
            y_true = np.concatenate((np.zeros(len(healthy_scores_patient.flatten())), np.ones(len(anomalous_scores_patient.flatten()))))
            y_scores = np.concatenate((healthy_scores_patient.flatten(), anomalous_scores_patient.flatten()))
            auc_roc = roc_auc_score(y_true, y_scores)
            logging.info('AUC-ROC: {:.3f}'.format(auc_roc))

            # After AUC-ROC is computed, compute the average precision score
            # Compute AUC-PR
            auc_pr = average_precision_score(y_true, y_scores)
            logging.info('AUC-PR: {:.3f} \n\n'.format(auc_pr))

            results_summary['Model'].append(exp_rel_path.split('/')[0])
            results_summary['Preprocess Method'].append(exp_rel_path.split('/')[1])
            results_summary['SEED'].append(exp_rel_path.split('_SEED_')[1].split('_')[0])
            results_summary['AUC-ROC'].append(auc_roc)
            results_summary['Average Precision'].append(auc_pr)
            # Add column with or without rotation
            results_summary['Rotation'].append('with_rotation' if exp_rel_path.__contains__('with_rotation') else 'without_rotation')

            # Compute some statistics and visulization over the age, sex and group distribution
            # First convert dictionary to pandas dataframe
            subject_df = pd.DataFrame.from_dict(subject_dict, orient='index')

            # Drop all rows without anomaly score
            subject_df = subject_df.dropna(subset=['anomaly_score'])
            
            # Perform descriptive statistics
            desc_stats = descriptive_stats(subject_df)


            # Compare anomaly scores between groups
            p_val = compare_groups(subject_df)
            

            # Perform correlation analysis
            pearson_corr_age, pearson_p_val_age, point_biserial_corr, point_biserial_p_val = correlation_analysis(subject_df)
    
            # Perform regression analysis
            mse, r2, intercept, coefficients = regression_analysis(subject_df)
        
            # Visualize data
            visualize_data(subject_df, results_dir_test)


            logger.removeHandler(handler)
        
        # Create summary table
        summary_df = pd.DataFrame(results_summary)

        # Log summary table at the end of the log file
        logging.info('Summary:')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            
            logging.info('\n{}'.format(summary_df))
        
        
