# =================================================================================
# ============== GENERAL PACKAGE IMPORTS ==========================================
# =================================================================================
import os
import re
import torch 
import yaml
import h5py
import logging
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import RegularGridInterpolator

from config import system_eval as config_sys



# =================================================================================
# ============== HELPERS PACKAGE IMPORTS ===========================================
# =================================================================================

from helpers import data_bern_numpy_to_preprocessed_hdf5
from helpers import data_bern_numpy_to_hdf5
from helpers.utils import make_dir_safely
from helpers.run import load_model
from helpers.data_loader import load_data
from helpers.metrics import RMSE
from helpers.batches import plot_batch_3d_complete

# =================================================================================
# ============== IMPORT MODELS ====================================================
# =================================================================================
from models.vae import VAE

"""
https://github.com/jemtan/FPI/blob/master/synthetic/example_synthesizing_outliers_mood.ipynb
"""
def calc_distance(xyz0 = [], xyz1 = []):
    delta_OX = (xyz0[0] - xyz1[0])**2
    delta_OY = (xyz0[1] - xyz1[1])**2
    delta_OZ = (xyz0[2] - xyz1[2])**2
    return (delta_OX+delta_OY+delta_OZ)**0.5 

def create_mask(im,center,width):
    dims = np.shape(im)
    mask = np.zeros_like(im)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                dist_i = calc_distance([i,j,k],center)
                if dist_i<width:
                    mask[i,j,k]=1
    return mask


def create_deformation(im,center,width,polarity=1):
    dims = np.array(np.shape(im))
    mask = np.zeros_like(im)
    
    center = np.array(center)
    xv,yv,zv = np.arange(dims[0]),np.arange(dims[1]),np.arange(dims[2])
    interp_samp = RegularGridInterpolator((xv, yv, zv), im)
    
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                dist_i = calc_distance([i,j,k],center)
                displacement_i = (dist_i/width)**2
                
                if displacement_i < 1.:
                    #within width
                    if polarity > 0:
                        #push outward
                        diff_i = np.array([i,j,k])-center
                        new_coor =  center + diff_i*displacement_i
                        new_coor = np.clip(new_coor,(0,0,0),dims-1)
                        mask[i,j,k]= interp_samp(new_coor)
                        
                    else:
                        #pull inward
                        cur_coor = np.array([i,j,k])
                        diff_i = cur_coor-center
                        new_coor = cur_coor + diff_i*(1-displacement_i)
                        new_coor = np.clip(new_coor,(0,0,0),dims-1)
                        mask[i,j,k]= interp_samp(new_coor)
                else:
                    mask[i,j,k] = im[i,j,k]
    return mask

def generate_deformation(im, save_viz=False, subject_idx=0):
    # Very inneficient way of doing this, but it works
    for batch_i in range(im.shape[0]):
        for channel in range(im.shape[-1]):

            if channel ==1 and batch_i == 0 and subject_idx ==0:
                fig, axs = plt.subplots(3, im.shape[0], figsize=(17, 7))
                im_original = im[batch_i,:,:,:,channel].copy()
            im_in = im[batch_i,:,:,:,channel]
            dims = np.array(np.shape(im_in))
            core = dims/4 #width of core region
            offset = (dims-core)/2#offset to center core
            #print('dims: ',dims, ' core: ',core, ' offset: ',offset)

            min_width = np.round(0.05*dims[0])
            max_width = np.round(0.2*dims[0])

            sphere_center = []
            sphere_width = []

            for i,_ in enumerate(dims[:3]):
                sphere_center.append(np.random.randint(offset[i],offset[i]+core[i]))
            sphere_width = np.random.randint(min_width,max_width)
            mask_i = create_mask(im_in,sphere_center,sphere_width)
            sphere_polarity = 1
            if np.random.randint(2):#random sign
                sphere_polarity *= -1
            t = sphere_center[2]
            # Apply deformation
            im[batch_i,:,:,:,channel] = create_deformation(im_in,sphere_center,sphere_width,sphere_polarity)
            if channel == 1 and save_viz and subject_idx == 0:
                
                axs[0, batch_i].imshow(im_original[:,:,t], vmin=-0.5, vmax=0.5)
                axs[0, batch_i].set_title('Original')
                axs[1, batch_i].imshow(mask_i[:,:,t], vmin=-0.5, vmax=0.5)
                axs[1, batch_i].set_title('Mask')
                axs[2, batch_i].imshow(im[batch_i,:,:,:,channel][:,:,t], vmin=-0.5, vmax=0.5)
                axs[2, batch_i].set_title('Deformed')
                
    if save_viz and subject_idx == 0:
        plt.savefig(os.path.join(project_code_root, 'Results/Evaluation/' + config['model'] + '/' + config['preprocess_method'] + '/' + config['model_name'] + '/' + 'deformation_example.png'))
        plt.close()
                

    return im

            

def generate_noise(config):
    noise_parameters = [(0.0, 0.02), (0.00, 0.1) ,(0.2, 0.4), (0.3, 0.1)] # additive noise to be tested, each tuple has mean and stdev for the normal distribution

    for (mean, std) in noise_parameters:

        # Add some random noise to the image
        part_noise = np.random.normal(mean, std, (config['batch_size'], 5, 5, config['spatial_size_t'], 4))
        full_noise = np.zeros((config['batch_size'], config['spatial_size_x'], config['spatial_size_y'], config['spatial_size_t'], 4))
        full_noise[:, 14:19, 14:19,:, :] = part_noise

        yield mean, std, full_noise
# =================================================================================
# ============== MAIN FUNCTION ==================================================
# =================================================================================

if __name__ ==  "__main__":
    config_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/config/evaluation/eval_config_vae.yaml'
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--config_path', type=str, default=config_path, help='Path to the config file.')
    # The rest of the arguments are passed to the config file
    args = parser.parse_args()

    # Load config file
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Combine arguments passed to config file and command line
    for arg, value in vars(args).items():
        config[arg] = value
    """
    # Load config file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)


    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    # ================================================
    # ======= LOGGING CONFIGURATION ==================
    # ================================================
    project_data_root = config_sys.project_data_root
    project_code_root = config_sys.project_code_root

    log_dir = os.path.join(config_sys.log_root, config['model'],config['preprocess_method'], config['model_name'], 'EVAL' + '_' + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M"))

    make_dir_safely(log_dir)
    
    log_file = os.path.join(log_dir, 'log.txt')
    if not os.path.exists(log_file):
        open(log_file, 'a').close()
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(log_file))
    logging.info('============================================================')
    logging.info(f"Logging to {log_dir}")
    logging.info('============================================================')
    
    # ================================================
    # ======= LOAD DATA ==============================
    # ================================================
    logging.info('Loading data...')
    images_tr, images_vl, images_ts = load_data(config, config_sys, idx_start_tr=0, idx_end_tr=5, idx_start_vl=42, idx_end_vl=44, idx_start_ts=0, idx_end_ts=2)
    _, images_syn, _ = load_data(config, config_sys, idx_start_tr=0, idx_end_tr=5, idx_start_vl=42, idx_end_vl=44, idx_start_ts=0, idx_end_ts=2)


    # ================================================
    # ======= LOAD MODEL ==============================
    # ================================================
    
    
    if config['model'] == 'vae':
            # Find the z_dim from the model name
            match = re.search(r'zdim(\d+)', config['model_name'])
            model = VAE(z_dim=match.group(1), in_channels=4, gf_dim=8).to(device)
    else:
        raise ValueError(f"Unknown model: {config['model']}")
    model_path = os.path.join(project_code_root, config["model_directory"])
    model = load_model(model, model_path, config, device=device)

    # ================================================
    # ======= EVALUATE MODEL =========================
    # ================================================
    # Create a directory for the results
    results_dir_train =     'Results/Evaluation/' + config['model'] + '/' + config['preprocess_method'] + '/' + config['model_name'] + '/' + 'train'
    results_dir_val = 'Results/Evaluation/' + config['model'] + '/' + config['preprocess_method'] + '/' + config['model_name'] + '/' + 'validation'
    results_dir_test = 'Results/Evaluation/' + config['model'] + '/' + config['preprocess_method'] + '/' + config['model_name'] + '/' + 'test'
    results_dir_noisy = 'Results/Evaluation/' + config['model'] + '/' + config['preprocess_method'] + '/' + config['model_name'] + '/' + 'noisy'
    results_dir_deformation = 'Results/Evaluation/' + config['model'] + '/' + config['preprocess_method'] + '/' + config['model_name'] + '/' + 'deformation'
    make_dir_safely(os.path.join(project_code_root, results_dir_train))
    make_dir_safely(os.path.join(project_code_root, results_dir_val))
    make_dir_safely(os.path.join(project_code_root, results_dir_test))
    make_dir_safely(os.path.join(project_code_root, results_dir_noisy))
    make_dir_safely(os.path.join(project_code_root, results_dir_deformation))

    # Write summary of important parameters to log file
    logging.info('============================================================')
    logging.info('SUMMARY OF IMPORTANT PARAMETERS')
    logging.info('============================================================')

    logging.info(f"Model: {config['model']}")
    logging.info(f"Preprocessing method: {config['preprocess_method']}")
    logging.info(f"Model name: {config['model_name']}")
    logging.info('Evaluation on {} subjects and {} slices'.format(config['subject_mode'],config['slice_mode']))
    logging.info('Number of epochs: {}'.format(config['latest_model_epoch']))
    logging.info('Which datasets: {}'.format(str(config['which_datasets'])))

    if config['subject_mode'] != 'all':
        logging.info('Custom train subjects: {}'.format(str(config['subjects_train'])))
        logging.info('Custom validation subjects: {}'.format(str(config['subjects_validation'])))
        logging.info('Custom test subjects: {}'.format(str(config['subjects_test'])))
    if config['slice_mode'] != 'all':
         logging.info('Custom slices selected: {}'.format(str(config['which_slices'])))

    
    logging.info('============================================================')
    logging.info('============================================================')

    number_slices_per_patient = config['spatial_size_z']

    healthy_mean_rmse = []
    anomalous_mean_rmse = []

    mean = np.inf
    std = np.inf
    # Loop over the different datasets and evaluate 
    for which_dataset in config['which_datasets']:
         
        logging.info('============= DATASET: {} ============='.format(which_dataset))
        

        # Store array for all RMSE in dataset
        all_dataset_mean_rmse = []

        # Select subjects 
        if config['subject_mode'] == 'all':
            if which_dataset == 'train':
                subject_indexes = range(config['train_data_end_idx'])
            elif which_dataset == 'validation':
                subject_indexes = range(config['validation_data_end_idx'] - config['validation_data_start_idx'])
            elif which_dataset == 'test':
                subject_indexes = range(config['test_data_end_idx'])
            else:
                # This will be for noisy and deformation
                subject_indexes = range(config['synthetic_data_end_idx'] - config['synthetic_data_start_idx'])
        else:
            if which_dataset == 'train':
                subject_indexes = config['subjects_train']
            elif which_dataset == 'validation':
                subject_indexes = config['subjects_validation']
            elif which_dataset == 'test':
                subject_indexes = config['subjects_test']
            else:
                # This will be for noisy and deformation
                subject_indexes = range(config['synthetic_data_end_idx'] - config['synthetic_data_start'])


        # If we create a hdf5
        if config['save_hdf5']:

            if which_dataset == 'train':
                
                #hdf5_path = os.path.join(project_code_root, results_dir_train)
                
                filepath_output = os.path.join(project_code_root, 'Results/model_reconstructions/' + config['model'] + '/' + config['preprocess_method'] + '/' + config['model_name'] + '_' + which_dataset + '_' + str(config['train_data_start_idx']) + 'to' + str(config['train_data_end_idx']) + '.hdf5')
                
            elif which_dataset == 'validation':
                filepath_output = os.path.join(project_code_root, 'Results/model_reconstructions/' + config['model'] + '/' + config['preprocess_method'] + '/' + config['model_name'] + '_' + which_dataset + '_' + str(config['validation_data_start_idx']) + 'to' + str(config['validation_data_end_idx']) + '.hdf5')
                
            elif which_dataset == 'test':
                filepath_output = os.path.join(project_code_root, 'Results/model_reconstructions/' + config['model'] + '/' + config['preprocess_method'] + '/' + config['model_name'] + '_' + which_dataset + '_' + str(config['test_data_start_idx']) + 'to' + str(config['test_data_end_idx']) + '.hdf5')

            elif which_dataset == 'noisy':
                filepath_output = os.path.join(project_code_root, 'Results/model_reconstructions/' + config['model'] + '/' + config['preprocess_method'] + '/' + config['model_name'] + '_' + which_dataset + '_' + str(config['synthetic_data_start_idx']) + 'to' + str(config['synthetic_data_end_idx']) + '.hdf5')
                
            elif which_dataset == 'deformation':
                filepath_output = os.path.join(project_code_root, 'Results/model_reconstructions/' + config['model'] + '/' + config['preprocess_method'] + '/' + config['model_name'] + '_' + which_dataset + '_' + str(config['synthetic_data_start_idx']) + 'to' + str(config['synthetic_data_end_idx']) + '.hdf5')
            else:
                raise ValueError('Unknown dataset: {}'.format(which_dataset))
            
            # Create hdf5 file
            dataset = {}
            hdf5_file = h5py.File(filepath_output, "w")
            
            # Check if dataset is train, validation or test
            if ['train', 'validation', 'test'].__contains__(which_dataset):
                dataset['reconstruction'] = hdf5_file.create_dataset("reconstruction", images_tr.shape, dtype='float32')
            elif which_dataset == 'noisy':
                dataset['noisy'] = hdf5_file.create_dataset("noisy", images_syn.shape, dtype='float32')
                dataset['noisy_reconstruction'] = hdf5_file.create_dataset("noisy_reconstruction", images_syn.shape, dtype='float32')
            elif which_dataset == 'deformation':
                dataset['deformation'] = hdf5_file.create_dataset("deformation", images_syn.shape, dtype='float32')
                dataset['deformation_reconstruction'] = hdf5_file.create_dataset("deformation_reconstruction", images_syn.shape, dtype='float32')
            

        # Loop over subjects

        for subject_idx in subject_indexes:

            # Grad full data for subject
            if which_dataset == 'train':
                subject_sliced = images_tr[subject_idx*number_slices_per_patient:(subject_idx+1)*number_slices_per_patient,:,:,:,:]
            elif which_dataset == 'validation':
                subject_sliced = images_vl[subject_idx*number_slices_per_patient:(subject_idx+1)*number_slices_per_patient,:,:,:,:]
            elif which_dataset == 'test':
                subject_sliced = images_ts[subject_idx*number_slices_per_patient:(subject_idx+1)*number_slices_per_patient,:,:,:,:]
            elif which_dataset == 'noisy':
                subject_sliced = images_syn[subject_idx*number_slices_per_patient:(subject_idx+1)*number_slices_per_patient,:,:,:,:]
            elif which_dataset == 'deformation':
                subject_sliced = images_syn[subject_idx*number_slices_per_patient:(subject_idx+1)*number_slices_per_patient,:,:,:,:]

            
            # Initialize while loop
            start_idx = 0
            end_idx = config['batch_size']
            subject_rmse = []
            if which_dataset == 'noisy':

                generator_noise = generate_noise(config)
                mean, std, full_noise = next(generator_noise)
                print('Mean: {}, std: {}'.format(mean, std))

            while end_idx <= config['spatial_size_z']:

                # Select slices for batch and run model
                subject_sliced_batch = subject_sliced[start_idx:end_idx,:,:,:,:]
                # If deformation dataset
                if which_dataset == 'deformation':
                    subject_sliced_batch = generate_deformation(subject_sliced_batch, save_viz=True, subject_idx=subject_idx)
                    # Save it to the hdf5 file
                    if (config['save_hdf5']):
                        dataset['deformation'][start_idx+(subject_idx*config["spatial_size_z"]):end_idx+(subject_idx*config["spatial_size_z"]), :, :, :, :] = subject_sliced[start_idx:end_idx]

                # If noisy dataset
                if which_dataset == 'noisy':
                    subject_sliced_batch = subject_sliced_batch + full_noise
                    # Save it to the hdf5 file if the selected noise level is hit
                    if (mean == 0.1) and (std == 0.1) and (config['save_hdf5']):
                        dataset['noisy'][start_idx+(subject_idx*config["spatial_size_z"]):end_idx+(subject_idx*config["spatial_size_z"]), :, :, :, :] = subject_sliced[start_idx:end_idx]

                # Turn into tensor
                subject_sliced_batch = torch.from_numpy(subject_sliced_batch).permute(0,4,1,2,3).to(device)
                # Run model
                model.eval()
                with torch.no_grad():
                    reconstruction, _, _, _  = model(subject_sliced_batch.float())
                    
                    # Save deformation reconstruction to hdf5 file
                    if (config['save_hdf5'] & (which_dataset == 'deformation')):
                        dataset['deformation_reconstruction'][start_idx+(subject_idx*config["spatial_size_z"]):end_idx+(subject_idx*config["spatial_size_z"]), :, :, :, :] = reconstruction

                    # Save noisy reconstruction to hdf5 file for specific noise level
                    if (config['save_hdf5'] & (which_dataset == 'noisy') & (mean == 0.1) & (std == 0.1)):
                        dataset['noisy_reconstruction'][start_idx+(subject_idx*config["spatial_size_z"]):end_idx+(subject_idx*config["spatial_size_z"]), :, :, :, :] = reconstruction


                    error = RMSE(subject_sliced_batch, reconstruction)
                    subject_rmse.append(error.item())

                    # Visualize batch
                    if config['visualization_mode'] == 'all':
                        out_path = os.path.join(project_code_root,'Results/Evaluation/' + config['model'] + '/' + config['preprocess_method'] + '/' + config['model_name'] + '/' + which_dataset + '/' + 'Subject_' + str(subject_idx) + '_' + str(start_idx) + '_' + str(end_idx) + '.png')
                        subject_sliced_batch = subject_sliced_batch.transpose(1,2).transpose(2,3).transpose(3,4).cpu().numpy()
                        reconstruction = reconstruction.transpose(1,2).transpose(2,3).transpose(3,4).cpu().numpy()
                        plot_batch_3d_complete(subject_sliced_batch, reconstruction, every_x_time_step=2, out_path= out_path)
                    if config['save_hdf5'] and (which_dataset != 'noisy') and (which_dataset != 'deformation'):
                        # Save it to the hdf5 file
                        dataset['reconstruction'][start_idx+(subject_idx*config["spatial_size_z"]):end_idx+(subject_idx*config["spatial_size_z"]), :, :, :, :] = reconstruction

                # Update indices
                start_idx += config['batch_size']
                end_idx += config['batch_size']
            # We have the rmse of the different slices, compute total rmse and std for subject
            subject_rmse = np.array(subject_rmse)
            subject_rmse_mean = np.mean(subject_rmse)
            subject_rmse_std = np.std(subject_rmse)
            logging.info('Subject {} RMSE: {} +/- {}'.format(subject_idx, subject_rmse_mean, subject_rmse_std))
            all_dataset_mean_rmse.append(subject_rmse_mean)
        
        # End of dataset
        all_dataset_mean_rmse = np.array(all_dataset_mean_rmse)
        dataset_mean_rmse = np.mean(all_dataset_mean_rmse)
        dataset_std_rmse = np.std(all_dataset_mean_rmse)
        logging.info('Dataset {} RMSE: {} +/- {}'.format(which_dataset, dataset_mean_rmse, dataset_std_rmse))
        logging.info('============================================================')
        logging.info('============================================================')
        if ['validation'].__contains__(which_dataset):
            healthy_mean_rmse.append(dataset_mean_rmse)
        elif which_dataset == 'test':
            logging.info('CAREFUL: TEST DATA ALSO CONTAINS HEALTHY DATA.... FIND A BETTER WAY')
            anomalous_mean_rmse.append(dataset_mean_rmse)
        else:
            anomalous_mean_rmse.append(dataset_mean_rmse)
        if config['save_hdf5']:
            hdf5_file.close()
            logging.info('HDF5 file saved to {}'.format(filepath_output))
            logging.info('============================================================')
    if config['compute_roc_auc']:
        logging.info('Computing ROC AUC...')
        if (config['validation_data_start_idx'] == config['synthetic_data_start_idx']) and (config['validation_data_end_idx'] == config['synthetic_data_end_idx']):
            logging.info('NOTE: Synthetic data is created on the validation')
            
        
        healthy_mean_rmse = np.array(healthy_mean_rmse)
        anomalous_mean_rmse = np.array(anomalous_mean_rmse)
        y_true = np.concatenate((np.zeros(healthy_mean_rmse.shape), np.ones(anomalous_mean_rmse.shape)))
        y_score = np.concatenate((healthy_mean_rmse, anomalous_mean_rmse))
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc_score = auc(fpr, tpr)
        logging.info('AUC score: {}'.format(auc_score))
        logging.info('============================================================')
        logging.info('============================================================')
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_score:.2f}')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.savefig(os.path.join(project_code_root, 'Results/Evaluation/' + config['model'] + '/' + config['preprocess_method'] + '/' + config['model_name'] + '/' + 'ROC_curve.png'))
            





