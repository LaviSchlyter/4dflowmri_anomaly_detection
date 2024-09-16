import os
import pickle
import numpy as np
import random
import torch
import logging
import pandas as pd
from config import system_eval as config_sys

from helpers.data_loader import load_data
from helpers.utils import make_dir_safely
import glob
from models.model_zoo import SimpleConvNet, VAE_convT, ConvWithAux, ConvWithEncDecAux, ConvWithDeepAux, ConvWithDeepEncDecAux, ConvWithDeeperBNEncDecAux, ConvWithDeeperEncDecAux
from inference_experiments_list import short_experiments_with_cs

from helpers.utils_inference import filter_subjects

logging.basicConfig(level=logging.INFO)


seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def evaluate_training_set(models_dir, data_path, subject_dict_path, project_data_root, project_code_root, chosen_dictionary, idx_start_tr, idx_end_tr, name_pre_extension, batch_size=32, spatial_size_z=64):
    adjacent_batch_slices = None
    
    for i, (list_of_experiments_with_rotation, list_of_experiments_without_rotation) in enumerate(zip(chosen_dictionary['with_rotation'], chosen_dictionary['without_rotation'])):
        logging.info('Processing experiment {}/{}'.format(i+1, len(chosen_dictionary['with_rotation'])))
        list_of_experiments_paths = list_of_experiments_with_rotation + list_of_experiments_without_rotation

        logging.info('List of experiments paths: {}'.format(list_of_experiments_paths))
     
        for exp_rel_path in list_of_experiments_paths:
            suffix_data_path = ''
            if exp_rel_path.__contains__('balanced'):
                suffix_data_path = '_balanced'

            if not exp_rel_path:
                continue
            
            exp_path = os.path.join(models_dir, exp_rel_path)
            pattern = os.path.join(exp_path, "*best*")
            best_exp_path = glob.glob(pattern)[0]
            model_str = exp_rel_path.split("/")[0]
            
            preprocess_method = exp_rel_path.split("/")[1]
            exp_name = exp_rel_path.split("/")[-1]
            project_code_root = config_sys.project_code_root
            results_dir = os.path.join(project_code_root, 'Results/Evaluation/' + model_str + '/' + preprocess_method + '/' + exp_name)
            
            make_dir_safely(results_dir)        
            results_dir_train = results_dir + '/' + f'train'
            make_dir_safely(results_dir_train)
            log_file = os.path.join(results_dir_train, f'log_train.txt')
            logger = logging.getLogger()
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            logging.info('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            logging.info('Model being processed: {}'.format(exp_rel_path.split('/')[0]))
            logging.info('Experiment name: {}'.format(exp_rel_path.split('/')[-1]))
            logging.info('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

            logging.info('name pre extension: {}'.format(name_pre_extension))

            if exp_rel_path.__contains__('SSL'):
                model_type = 'self-supervised'
                in_channels = 4
                out_channels = 1
            else:
                model_type = 'reconstruction-based'
                in_channels = 4
                out_channels = 4

            if len(name_pre_extension) > 1:
                name_extension = name_pre_extension[i]
            else:
                name_pre = name_pre_extension[0]
                name_extension = name_pre + exp_name.split('2Dslice_')[1].split('_decreased_interpolation_factor_cube_3')[0]

            logging.info('name_extension: {}'.format(name_extension))

            if exp_name.__contains__('simple_conv'):
                model = SimpleConvNet(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
            elif exp_name.__contains__('vae_convT'):
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
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(best_exp_path, map_location=device))
            model.to(device)
            model.eval()
            
            preprocess_method = exp_rel_path.split("/")[1]
            config = {'preprocess_method': preprocess_method}

            data_dict = load_data(sys_config=config_sys, config=config, idx_start_tr=idx_start_tr, idx_end_tr=idx_end_tr, idx_start_vl=0, idx_end_vl=1, idx_start_ts=0, idx_end_ts=1, with_test_labels=False, suffix=name_extension)
            images_train = data_dict['images_tr']
            
            rotation_matrix = data_dict.get('rotation_train', None)
            
            train_names = filter_subjects(data_path, exp_rel_path, suffix=suffix_data_path, train=True)

            logging.info('============================================================')
            logging.info('Train subjects: {}'.format(train_names))
            logging.info('============================================================')

            with open(os.path.join(subject_dict_path, 'subject_dict.pkl'), 'rb') as f:
                subject_dict = pickle.load(f)
            
            subject_indexes = range(np.int16(images_train.shape[0] / spatial_size_z))

            # Keep track for mean and std of anomaly scores
            mean_anomaly_scores = []
            
            
            for subject_idx in subject_indexes:
                subject_name = train_names[subject_idx]
                logging.info('Processing subject {}'.format(subject_name))
                start_idx = 0
                end_idx = batch_size

                if subject_name[-1] == '_':
                    subject_dict[subject_name] = subject_dict[subject_name[:-1]].copy()

                subject_name_npy = subject_name + '.npy'
                
                if subject_name_npy in os.listdir(os.path.join(img_path)):
                    compressed_sensing = False
                elif subject_name_npy in os.listdir(os.path.join(img_path+'_compressed_sensing')):
                    compressed_sensing = True
                else:
                    raise ValueError("Cannot find the individual")
               

                subject_dict[subject_name]['Compressed_sensing'] = compressed_sensing

                subject_anomaly_score = []
                subject_reconstruction = []
                subject_sliced = images_train[subject_idx * spatial_size_z:(subject_idx + 1) * spatial_size_z]
                
                if rotation_matrix is not None:
                    rotation_matrix_subject = rotation_matrix[subject_idx * spatial_size_z:(subject_idx + 1) * spatial_size_z]
                while end_idx <= spatial_size_z:
                    batch = subject_sliced[start_idx:end_idx]
                
                    batch_z_slice = torch.from_numpy(np.arange(start_idx, end_idx)).float().to(device)
                    batch = torch.from_numpy(batch).transpose(1, 4).transpose(2, 4).transpose(3, 4).float().to(device)
                    rotation_matrix_batch = None
                    if rotation_matrix is not None:
                        rotation_matrix_batch = rotation_matrix_subject[start_idx:end_idx]
                        rotation_matrix_batch = torch.from_numpy(rotation_matrix_batch).to(device).float()
                    with torch.no_grad():
                        model.eval()
                        
                        input_dict = {'input_images': batch, 'batch_z_slice': batch_z_slice, 'adjacent_batch_slices': adjacent_batch_slices, 'rotation_matrix': rotation_matrix_batch}
                        output_dict = model(input_dict)

                        if model_type == 'self-supervised':
                            output_images = torch.sigmoid(output_dict['decoder_output'])
                        else:
                            logging.info('Reconstruction based model')
                            model_output = output_dict['decoder_output']
                            output_images = torch.abs(model_output - batch)
                            subject_reconstruction.append(output_images.cpu().detach().numpy())

                        subject_anomaly_score.append(output_images.cpu().detach().numpy())

                    start_idx += batch_size
                    end_idx += batch_size

                results_dir_subject = os.path.join(results_dir_train, "outputs")
                inputs_dir_subject = os.path.join(results_dir_train, "inputs")
                make_dir_safely(inputs_dir_subject)
                make_dir_safely(results_dir_subject)
                file_path_input = os.path.join(inputs_dir_subject, f'{subject_name}_inputs.npy')
                file_path = os.path.join(results_dir_subject, f'{subject_name}_anomaly_scores.npy')
                np.save(file_path, np.concatenate(subject_anomaly_score))
                np.save(file_path_input, subject_sliced)
                if model_type == 'reconstruction-based':
                    reconstruction_dir_subject = os.path.join(results_dir_train, "reconstruction")
                    make_dir_safely(reconstruction_dir_subject)
                    file_path_reconstruction = os.path.join(reconstruction_dir_subject, f'{subject_name}_reconstruction.npy')
                    np.save(file_path_reconstruction, np.concatenate(subject_reconstruction))
                
                
                # Save the anomaly scores and standard deviation of the subject in the subject_dict for further analysis
                subject_dict[subject_name]['anomaly_score'] = np.mean(subject_anomaly_score)
                subject_dict[subject_name]['std_anomaly_score'] = np.std(subject_anomaly_score)
                
                logging.info('{}_subject {} anomaly_score: {:.4e} +/- {:.4e}'.format('control', subject_idx, np.mean(subject_anomaly_score), np.std(subject_anomaly_score)))
                mean_anomaly_scores.append(np.mean(subject_anomaly_score))
                
        # Logging the mean of all anomaly scores
        mean_anomaly_scores = np.array(mean_anomaly_scores)
        logging.info('Healthy mean anomaly score: {:.4e}'.format(np.mean(mean_anomaly_scores)))


        # Convert list to set for faster lookup
        train_names_set = set(train_names)

        # Filter the dictionary to only keep individuals in the train set
        filtered_subject_dict = {k: v for k, v in subject_dict.items() if k in train_names_set}

        # Compute some statistics and visulization over the age, sex and group distribution
        subject_df = pd.DataFrame.from_dict(filtered_subject_dict, orient='index')
        
        # Save the subject_df and name index as ID
        subject_df.to_csv(os.path.join(results_dir, 'train_subject_df_with_anomaly_scores.csv'), index_label='ID')

        

        logger.removeHandler(handler)

if __name__ == '__main__':
    
    #models_dir = "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/logs"
    models_dir = "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/Saved_models"
    data_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/final_segmentations'
    subject_dict_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/preprocessed'

    project_data_root = config_sys.project_data_root
    project_code_root = config_sys.project_code_root
    img_path = os.path.join(project_data_root, f'preprocessed/controls/numpy')
    

    chosen_dictionary = short_experiments_with_cs
    name_pre_extension = ['']
    
    idx_start_tr = 0
    idx_end_tr = 41  # Or 51 if including validation set
    


    evaluate_training_set(models_dir, data_path, subject_dict_path, project_data_root, project_code_root, chosen_dictionary, idx_start_tr, idx_end_tr, name_pre_extension)

