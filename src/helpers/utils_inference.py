import numpy as np
import logging
import os
import yaml
import torch

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection')
from config import system_eval as config_sys
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/src')
from helpers.utils import make_dir_safely, resample_back, convert_to_vtk, expand_normal_slices
project_code_root = config_sys.project_code_root
import pickle
import SimpleITK as sitk

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from models.model_zoo import (
    SimpleConvNet, VAE_convT, ConvWithAux, ConvWithEncDecAux, ConvWithDeepAux, 
    ConvWithDeepEncDecAux, ConvWithDeeperBNEncDecAux, ConvWithDeeperEncDecAux
)

# Dictionary with matches_scenarios
matches_scenarios= {
  
    "Sex Equal, Age Gap <= 5": [
        ('MACDAVD_204_', 'MACDAVD_104_'),
        ('MACDAVD_308_', 'MACDAVD_101_'),
        ('MACDAVD_213', 'MACDAVD_102_'),
        ('MACDAVD_214', 'MACDAVD_107_'),
        ('MACDAVD_203_', 'MACDAVD_112_'),
        ('MACDAVD_206', 'MACDAVD_109_'),
        ('MACDAVD_305_', 'MACDAVD_106_'),
        ('MACDAVD_309_', 'MACDAVD_135'),
        ('MACDAVD_303_', 'MACDAVD_108_'),
        ('MACDAVD_307_', 'MACDAVD_131')
    ]}

# Custom palette
custom_palette = {'Male': '#ABC8E2', 'Female': '#F4BE9F', 'Cases': '#C9A9E5', 'Controls': '#A7D8A2'}
custom_cs_palette = {True: '#4C72B0', False: '#55A868'}
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.titlesize'] = 16

# =============================================================================
# Basic functions
# =============================================================================

# Function to ensure keys are initialized
def initialize_result_keys(results_summary, labels):
    for label in labels:
        if f'{label} AUC-ROC' not in results_summary:
            results_summary[f'{label} AUC-ROC'] = []
        if f'{label} Average Precision' not in results_summary:
            results_summary[f'{label} Average Precision'] = []

# Function to get the model type based on experiment path
def get_model_type(exp_rel_path):
    if 'SSL' in exp_rel_path:
        return 'self-supervised', 4, 1
    else:
        return 'reconstruction-based', 4, 4

# Function to initialize the model
def initialize_model(exp_name, in_channels, out_channels):
    if 'simple_conv' in exp_name:
        return SimpleConvNet(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
    elif 'vae_convT' in exp_name:
        return VAE_convT(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
    elif 'deeper_conv_enc_dec' in exp_name:
        return ConvWithDeeperEncDecAux(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
    elif 'deeper_bn_conv_enc_dec' in exp_name:
        return ConvWithDeeperBNEncDecAux(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
    elif 'deep_conv_with_aux' in exp_name:
        return ConvWithDeepAux(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
    elif 'deep_conv_enc_dec' in exp_name:
        return ConvWithDeepEncDecAux(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
    elif 'conv_with_aux' in exp_name:
        return ConvWithAux(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
    elif 'conv_enc_dec' in exp_name:
        return ConvWithEncDecAux(in_channels=in_channels, out_channels=out_channels, gf_dim=8)
    else:
        raise ValueError('Exp name {} has no model recognized'.format(exp_name))

# Function to load subject dictionary
def load_subject_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
# =============================================================================
# Functions for data analysis
# =============================================================================

def descriptive_stats(df):
    """
    Compute descriptive statistics for anomaly scores.
    """
    logging.info("Descriptive Statistics:")
    desc_stats = df.groupby('Label')['anomaly_score'].describe()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logging.info('\n{}'.format(desc_stats))
    return desc_stats

def correlation_analysis(df):
    """
    Perform correlation analysis between age, and anomaly scores.
    
    Args:
    - df: DataFrame containing the data
    
    Returns:
    - pearson_corr_age: Pearson correlation coefficient between age and anomaly scores
    - pearson_p_val_age: p-value for the correlation between age and anomaly scores
    
    """
    logging.info("\nCorrelation Analysis:")
    
    # Compute Pearson correlation coefficient between age and anomaly scores
    pearson_corr_age, pearson_p_val_age = pearsonr(df['Age'], df['anomaly_score'])
    # Compute Spearman correlation coefficient between age and anomaly scores
    spearman_corr_age, spearman_p_val_age = spearmanr(df['Age'], df['anomaly_score'])
    # Logging with 3 point precision
    logging.info("Pearson correlation coefficient (Age vs. Anomaly Score): {:.3f}".format(pearson_corr_age))
    logging.info("Pearson correlation p-value (Age vs. Anomaly Score): {:.3f}".format(pearson_p_val_age))
    logging.info("Spearman correlation coefficient (Age vs. Anomaly Score): {:.3f}".format(spearman_corr_age))
    logging.info("Spearman correlation p-value (Age vs. Anomaly Score): {:.3f}".format(spearman_p_val_age))
    
    return pearson_corr_age, pearson_p_val_age, spearman_corr_age, spearman_p_val_age
    
# =============================================================================
# Functions for backtransforming anomaly scores
# =============================================================================

def adjust_anomaly_scores(anomaly_scores, segmentation_mask):
    # Get the shape of the anomaly scores and segmentation mask
    anomaly_shape = anomaly_scores.shape
    seg_mask_shape = segmentation_mask.shape

    # Determine the size difference
    size_diff = seg_mask_shape[-1] - anomaly_shape[-1]

    if size_diff > 0:
        # If segmentation mask's time dimension is larger, pad the anomaly scores
        padding = [(0, 0)] * (anomaly_scores.ndim - 1) + [(0, size_diff)]
        adjusted_anomaly_scores = np.pad(anomaly_scores, padding, mode='constant', constant_values=0)
    else:
        # If segmentation mask's time dimension is smaller or equal, slice the anomaly scores
        adjusted_anomaly_scores = anomaly_scores[..., :seg_mask_shape[-1]]

    # Multiply the adjusted arrays
    result = adjusted_anomaly_scores * segmentation_mask

    return result

def backtransform_anomaly_scores(subject_name, subject_anomaly_scores, subject_dict, img_path, geometry_path,data_path, results_dir_test, suffix_data_path = ''):
    logging.info('Backtransforming anomaly scores...')
    save_path = os.path.join(results_dir_test, f'outputs_backtransformed')
    make_dir_safely(save_path)
    subject_name_npy = subject_name+'.npy'
    # Check if the anomaly scores have already been backtransformed
    if os.path.exists(os.path.join(save_path, subject_name_npy)):
        logging.info('Anomaly scores already backtransformed, skipping...')
        return
    else:
        # Load the anomaly scores
        anomaly_scores = np.concatenate(subject_anomaly_scores)
        # Form (z,c,x,y,t) to (x,y,z,t,c)
        anomaly_scores = anomaly_scores.transpose(2,3,0,4,1)
        # The original slice size had a (x,y) (36,36)
        # Reduce to (x,y) (32,32) for network, now we pad back to (36,36)
        anomaly_scores = expand_normal_slices(anomaly_scores, [36,36,64,24,4])
        # If the channel dimension is 1, we repeat it 4 times
        if anomaly_scores.shape[-1] == 1:
            anomaly_scores = np.repeat(anomaly_scores, 4, axis=-1)
        # Load the raw image
        # Print the subject name
        logging.info('Subject name: {}'.format(subject_name))
        if subject_dict[subject_name]['Label'] == 'Controls':
            if subject_dict[subject_name]['Compressed_sensing'] == False:
                image = np.load(os.path.join(img_path.replace("patients", "controls"), subject_name_npy))
            elif subject_dict[subject_name]['Compressed_sensing'] == True:
                # Compression sensing
                image = np.load(os.path.join(img_path.replace("patients", "controls")+'_compressed_sensing', subject_name_npy))
        elif subject_dict[subject_name]['Label'] == 'Cases':
            #Then we are dealing with a patient
            if subject_dict[subject_name]['Compressed_sensing'] == False:
                image = np.load(os.path.join(img_path, subject_name_npy))
            elif subject_dict[subject_name]['Compressed_sensing'] == True:
                # Compression sensing
                image = np.load(os.path.join(img_path+'_compressed_sensing', subject_name_npy))

    # The geometric information we need from the initial image does not change with channel or time 
    sitk_image = sitk.GetImageFromArray(image[:,:,:,0,0])
    # Load the geometry information of the slices
    geometry_dict = np.load(os.path.join(geometry_path, subject_name_npy), allow_pickle=True).item()

    # Resample back to the original image space
    anomaly_scores_original_frame = resample_back(anomaly_scores, sitk_image, geometry_dict)

    # Load segmentation from datapath
    seg_path = os.path.join(data_path,'test'+suffix_data_path, 'seg_'+subject_name_npy)
    segmentation_mask = np.load(seg_path)

    # Mask the anomaly scores (for visualization, since we dilated the segmentation a bit for preprocessing)
    anomaly_scores_original_frame = adjust_anomaly_scores(anomaly_scores_original_frame, segmentation_mask)
    anomaly_scores_original_frame = anomaly_scores_original_frame * segmentation_mask

    # Save the backtransformed anomaly scores
    np.save(os.path.join(save_path, subject_name_npy), anomaly_scores_original_frame)

    # Convert to vtk
    save_dir_vtk = os.path.join(save_path, "vtk", subject_name_npy.replace(".npy", ""))
    make_dir_safely(save_dir_vtk)
    convert_to_vtk(anomaly_scores_original_frame, subject_name_npy.replace(".npy", ""), save_dir_vtk)

# =============================================================================
# Functions for filtering subjects
# =============================================================================

def filter_subjects(data_path, experiment_name, suffix='', train=False):
    """
    Filter subjects based on experiment requirements. If we only use compressed sensing, we need the correct subject names

    Args:
        data_path (str): The path to the data directory.
        experiment_name (str): The name of the experiment.

    Returns:
        list: A list of filtered subject names.

    """
    # Paths to compressed sensing data directories
    cs_controls_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/preprocessed/controls/dicom_compressed_sensing'
    cs_patients_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/preprocessed/patients/dicom_compressed_sensing'
    
    if train:
        # Get all subject names and remove the .npy extension
        train_names = [os.path.splitext(name)[0].split('seg_')[1] for name in os.listdir(data_path + f'/train_val{suffix}')]
        train_names.sort()
        return train_names
    # If only compressed sensing data we have different distribution of subjects separated by train, val, test from the whole test
    if 'only_cs' in experiment_name:
        test_names = [os.path.splitext(name)[0].split('seg_')[1] for name in os.listdir(data_path + f'/test_compressed_sensing{suffix}')]

    else: # Without compressed sensing and all data share the same subjects across the split
    # Get all subject names and remove the .npy extension
        test_names = [os.path.splitext(name)[0].split('seg_')[1] for name in os.listdir(data_path + f'/test{suffix}')]
    test_names.sort()

    # Get the list of compressed sensing subjects and remove the .npy extension
    cs_subjects_controls = set(os.path.splitext(name)[0] for name in os.listdir(cs_controls_path))
    cs_subjects_patients = set(os.path.splitext(name)[0] for name in os.listdir(cs_patients_path))
    cs_subjects = cs_subjects_controls.union(cs_subjects_patients)

    # Filter subjects based on experiment requirements
    if 'without_cs' in experiment_name:
        # Remove compressed sensing subjects
        filtered_names = [name for name in test_names if name not in cs_subjects]
    elif 'only_cs' in experiment_name:
        # Only keep compressed sensing subjects
        filtered_names = [name for name in test_names if name in cs_subjects]
    else:
        # Keep all subjects
        filtered_names = test_names

    return filtered_names


# =============================================================================
# Functions for computing metrics
# =============================================================================

def compute_metrics(subject_df, type_data, results_summary):
    # Check if type_data is either Region, Region without dup., Region without mismatch, Region without mismatch, without dup.
    if 'Region' in type_data:
        # region_based_score is the column that contains the region based anomaly scores
        # region_based_label is the column that contains the region based labels
        # We need to extract the elements of the list and flatten
        y_scores_arrays = subject_df['region_based_score'].dropna().values
        y_true_arrays = subject_df['region_based_label'].dropna().values
        y_scores = np.concatenate([np.array(arr) for arr in y_scores_arrays])
        y_true = np.concatenate([np.array(arr) for arr in y_true_arrays])

        
    else:
        cases = subject_df[subject_df['Label'] == 'Cases']['anomaly_score'].values
        controls = subject_df[subject_df['Label'] == 'Controls']['anomaly_score'].values
    
        y_true = np.concatenate((np.zeros(len(controls)), np.ones(len(cases))))
        y_scores = np.concatenate((controls.flatten(), cases.flatten()))

    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)

    logging.info(f'{type_data} AUC-ROC: {auc_roc:.3f}')
    logging.info(f'{type_data} Average Precision: {auc_pr:.3f}\n')

    results_summary[f'{type_data} AUC-ROC'].append(auc_roc)
    results_summary[f'{type_data} Average Precision'].append(auc_pr)
    
    return y_true, y_scores, auc_roc


def compute_metrics_for_matches(matches, subject_df, match_type, results_summary):
    matched_pairs = {}
    for case_id, control_id in matches:
        matched_pairs[case_id] = {
            'case': subject_df.loc[case_id].to_dict(),
            'control': subject_df.loc[control_id].to_dict()
        }

    matched_anomaly_scores_cases = [matched_pairs[key]['case']['anomaly_score'] for key in matched_pairs]
    matched_anomaly_scores_controls = [matched_pairs[key]['control']['anomaly_score'] for key in matched_pairs]

    y_true_matched = np.concatenate((np.zeros(len(matched_anomaly_scores_controls)), np.ones(len(matched_anomaly_scores_cases))))
    y_scores_matched = np.concatenate((matched_anomaly_scores_controls, matched_anomaly_scores_cases))

    auc_roc_matched = roc_auc_score(y_true_matched, y_scores_matched)
    auc_pr_matched = average_precision_score(y_true_matched, y_scores_matched)

    logging.info(f"{match_type}:")
    logging.info(f"Number of Matches: {len(matched_pairs)}")
    logging.info(f"AUC-ROC: {auc_roc_matched:.3f}")
    logging.info(f"AUC-PR: {auc_pr_matched:.3f}\n")

    results_summary[f'{match_type} AUC-ROC'].append(auc_roc_matched)
    results_summary[f'{match_type} Average Precision'].append(auc_pr_matched)

    logging.info(f"Correlation statistics for {match_type} cases only:")
    # Compute Pearson and Spearman correlations for matched cases
    case_subjects = subject_df.loc[[pair[0] for pair in matches]]
    pearson_corr, pearson_p_value, spearman_corr, spearman_p_value = correlation_analysis(case_subjects)

    # Compute Pearson and Spearman correlations for matched pairs
    pairs_data = pd.DataFrame({
        'Age': [matched_pairs[key]['case']['Age'] for key in matched_pairs] + [matched_pairs[key]['control']['Age'] for key in matched_pairs],
        'anomaly_score': [matched_pairs[key]['case']['anomaly_score'] for key in matched_pairs] + [matched_pairs[key]['control']['anomaly_score'] for key in matched_pairs]
    })
    logging.info(f"Correlation statistics for matched pairs:")
    pearson_corr_pairs, pearson_p_value_pairs, spearman_corr_pairs, spearman_p_value_pairs = correlation_analysis(pairs_data)

  
    return matched_pairs, y_true_matched, y_scores_matched, auc_roc_matched

# =============================================================================
# Functions for evaluating predictions at a region level
# =============================================================================

def fill_anomaly_scores(mask_quadrants, anomaly_scores):
    mean_predictions = np.mean(anomaly_scores, axis=-1).squeeze()
    filled_scores = np.zeros_like(mask_quadrants)
    
    for s in range(mask_quadrants.shape[0]):
        for i in range(mask_quadrants.shape[1]):
            region_mask = mask_quadrants[s, i, :, :] == 1
            if np.any(region_mask):
                filled_scores[s, i, :, :] = np.where(region_mask, mean_predictions[s, :, :][region_mask].mean(), 0)
    
    return filled_scores

def aggregate_scores(filled_scores, mask_quadrants):
    slice_ranges = [range(22), range(22, 43), range(43, 64)]
    aggregated_predictions = []
    
    for zone_idx, slices in enumerate(slice_ranges):
        for quadrant in range(4):
            region_scores = []
            for s in slices:
                region_mask = mask_quadrants[s, quadrant, :, :] == 1
                if np.any(region_mask):
                    region_scores.append(filled_scores[s, quadrant, :, :][region_mask].mean())
            aggregated_predictions.append(np.mean(region_scores))
    
    return aggregated_predictions

def evaluate_predictions(mask_quadrants, anomaly_scores, nested_dict, subject_name):
    filled_anomaly_scores = fill_anomaly_scores(mask_quadrants, anomaly_scores)
    aggregated_predictions = aggregate_scores(filled_anomaly_scores, mask_quadrants)
    
    true_labels = nested_dict[subject_name]
    zones = ['Lower', 'Mid', 'Upper']
    channels = ['PR', 'AR', 'PL', 'AL']
    y_true = []
    
    for zone in zones:
        for channel in channels:
            y_true.append(true_labels[zone][channel])
    
    return y_true, aggregated_predictions

# =============================================================================
# Functions for permutation tests
# =============================================================================


def permutation_test_auc_roc(y_true_labels, y_scores, n_permutations=1000, random_seed=42):
    """
    Perform a permutation test for the AUC-ROC score.
    
    Parameters:
    - y_true_labels: True labels for the dataset.
    - y_scores: Predicted scores for the dataset.
    - n_permutations: Number of permutations to perform (default is 1000).
    - random_seed: Seed for the permutation randomness, kept constant across experiments 
                   to ensure consistent permutation results (this seed is independent of 
                   the experimental seed used for model training).
    
    Returns:
    - p_value: The p-value of the permutation test.
    """
    np.random.seed(random_seed)

    original_auc_roc = roc_auc_score(y_true_labels, y_scores)

    # Permutation test for original AUC-ROC
    permuted_aucs = np.array([roc_auc_score(y_true_labels, np.random.permutation(y_scores)) for _ in range(n_permutations -1)])
    p_value = (np.sum(permuted_aucs >= original_auc_roc) + 1) / (n_permutations -1)

    logging.info(f"AUC-ROC - permutation test: {original_auc_roc:.4f}, p-value: {p_value:.4f}")

    return p_value

# =============================================================================
# Functions for Test-Retest Analysis
# =============================================================================

def permutation_test_within_acquisition_method(cs_scores, seq_scores, df, swap_cs=True, num_permutations=2000):
    # Calculate the observed differences for each pair of duplicates
    observed_diffs = np.abs(np.array(cs_scores) - np.array(seq_scores))
    perm_diffs = np.zeros((num_permutations, len(cs_scores)))

    for perm in range(num_permutations):
        for i in range(len(cs_scores)):
            if swap_cs:
                # Swap CS versions within the respective acquisition method
                cs_group_scores = df[df['Compressed_sensing'] == True]['anomaly_score'].values
                #Should probably remove itself ? 
                perm_cs_score = np.random.choice(cs_group_scores)
                perm_diffs[perm, i] = np.abs(perm_cs_score - seq_scores[i])
            else:
                # Swap non-CS versions within the respective acquisition method
                seq_group_scores = df[df['Compressed_sensing'] == False]['anomaly_score'].values
                perm_seq_score = np.random.choice(seq_group_scores)
                perm_diffs[perm, i] = np.abs(cs_scores[i] - perm_seq_score)
    
    # Calculate p-values for each pair
    p_values = []
    for i in range(len(cs_scores)):
        observed_diff = observed_diffs[i]
        perm_diff_distribution = perm_diffs[:, i]
        p_value = np.mean(perm_diff_distribution >= observed_diff)
        #p_value = np.mean(perm_diff_distribution <= observed_diff) #Might use this as update 19.09
        p_values.append(p_value)
    
    return observed_diffs, p_values

def permutation_test_within_label_and_method(cs_scores, seq_scores, df, df_duplicates, swap_cs=True, num_permutations=2000):
    # Calculate the observed differences for each pair of duplicates
    observed_diffs = np.abs(np.array(cs_scores) - np.array(seq_scores))
    perm_diffs = np.zeros((num_permutations, len(cs_scores)))

    for perm in range(num_permutations):
        for i in range(len(cs_scores)):
            label = df_duplicates.iloc[i]['Label']
            if swap_cs:
                # Swap CS versions within the same label and method
                cs_label_scores = df[(df['Label'] == label) & (df['Compressed_sensing'] == True)]['anomaly_score'].values
                perm_cs_score = np.random.choice(cs_label_scores)
                perm_diffs[perm, i] = np.abs(perm_cs_score - seq_scores[i])
            else:
                # Swap non-CS versions within the same label and method
                seq_label_scores = df[(df['Label'] == label) & (df['Compressed_sensing'] == False)]['anomaly_score'].values
                perm_seq_score = np.random.choice(seq_label_scores)
                perm_diffs[perm, i] = np.abs(cs_scores[i] - perm_seq_score)
    
    # Calculate p-values for each pair
    p_values = []
    for i in range(len(cs_scores)):
        observed_diff = observed_diffs[i]
        perm_diff_distribution = perm_diffs[:, i]
        p_value = np.mean(perm_diff_distribution >= observed_diff)
        #p_value = np.mean(perm_diff_distribution <= observed_diff) #Might use this as update 19.09
        p_values.append(p_value)
    
    return observed_diffs, p_values
def count_lower_diff_within_label_and_acquisition(cs_scores, seq_scores, df, df_duplicates, swap_cs=True):
    observed_diffs = np.abs(np.array(cs_scores) - np.array(seq_scores))
    count_lower_diff = np.zeros(len(cs_scores))
    total_comparisons = np.zeros(len(cs_scores))

    for i in range(len(cs_scores)):
        label = df_duplicates.iloc[i]['Label']
        if swap_cs:
            label_scores = df[(df['Label'] == label) & (df['Compressed_sensing'] == True)]['anomaly_score'].values
        else:
            label_scores = df[(df['Label'] == label) & (df['Compressed_sensing'] == False)]['anomaly_score'].values
        
        for score in label_scores:
            if swap_cs:
                if np.abs(score - seq_scores[i]) < observed_diffs[i]:
                    count_lower_diff[i] += 1
            else:
                if np.abs(cs_scores[i] - score) < observed_diffs[i]:
                    count_lower_diff[i] += 1
            total_comparisons[i] += 1  # Include the score in the total comparisons

        # Subtract one to exclude the comparison with itself
        total_comparisons[i] -= 1
    
    return count_lower_diff, total_comparisons

def count_lower_diff_within_acquisition(cs_scores, seq_scores, df, swap_cs=True):
    observed_diffs = np.abs(np.array(cs_scores) - np.array(seq_scores))
    count_lower_diff = np.zeros(len(cs_scores))
    total_comparisons = np.zeros(len(cs_scores))

    if swap_cs:
        group_scores = df[df['Compressed_sensing'] == True]['anomaly_score'].values
    else:
        group_scores = df[df['Compressed_sensing'] == False]['anomaly_score'].values

    for i in range(len(cs_scores)):
        for score in group_scores:
            if swap_cs:
                if np.abs(score - seq_scores[i]) < observed_diffs[i]:
                    count_lower_diff[i] += 1
            else:
                if np.abs(cs_scores[i] - score) < observed_diffs[i]:
                    count_lower_diff[i] += 1
            total_comparisons[i] += 1  # Include the score in the total comparisons

        # Subtract one to exclude the comparison with itself
        total_comparisons[i] -= 1
    
    return count_lower_diff, total_comparisons


def perform_permutation_tests_duplicates(subject_df):
    df_duplicates = subject_df[subject_df.duplicated('Base_Name', keep=False)].sort_values('Base_Name')
    cs_scores = []
    non_cs_scores = []

    for base_name, group in df_duplicates.groupby('Base_Name'):
        if len(group) == 2:
            cs_score = group[group['Compressed_sensing'] == True]['anomaly_score'].values
            non_cs_score = group[group['Compressed_sensing'] == False]['anomaly_score'].values
            if len(cs_score) == 1 and len(non_cs_score) == 1:
                cs_scores.append(cs_score[0])
                non_cs_scores.append(non_cs_score[0])

    cs_scores = np.array(cs_scores)
    non_cs_scores = np.array(non_cs_scores)

    if len(cs_scores) == len(non_cs_scores):
        _, perm_p_values_non_cs_group = permutation_test_within_acquisition_method(cs_scores, non_cs_scores, subject_df, swap_cs=False)
        _, perm_p_values_cs_group = permutation_test_within_acquisition_method(cs_scores, non_cs_scores, subject_df, swap_cs=True)
        _, perm_p_values_non_cs_label = permutation_test_within_label_and_method(cs_scores, non_cs_scores, subject_df, df_duplicates, swap_cs=False)
        _, perm_p_values_cs_label = permutation_test_within_label_and_method(cs_scores, non_cs_scores, subject_df, df_duplicates, swap_cs=True)

        # Additional Analysis
        count_lower_diff_label_and_acquisition_non_cs, total_comparisons_label_and_acquisition_non_cs = count_lower_diff_within_label_and_acquisition(cs_scores, non_cs_scores, subject_df, df_duplicates, swap_cs=False)
        count_lower_diff_label_and_acquisition_cs, total_comparisons_label_and_acquisition_cs = count_lower_diff_within_label_and_acquisition(cs_scores, non_cs_scores, subject_df, df_duplicates, swap_cs=True)
        count_lower_diff_acquisition_non_cs, total_comparisons_acquisition_non_cs = count_lower_diff_within_acquisition(cs_scores, non_cs_scores, subject_df, swap_cs=False)
        count_lower_diff_acquisition_cs, total_comparisons_acquisition_cs = count_lower_diff_within_acquisition(cs_scores, non_cs_scores, subject_df, swap_cs=True)

        # Update the subject_df with the results
        for i, base_name in enumerate(df_duplicates['Base_Name'].unique()):
            subject_df.loc[subject_df['Base_Name'] == base_name, 'perm_p_values_non_cs_group'] = perm_p_values_non_cs_group[i]
            subject_df.loc[subject_df['Base_Name'] == base_name, 'perm_p_values_cs_group'] = perm_p_values_cs_group[i]
            subject_df.loc[subject_df['Base_Name'] == base_name, 'perm_p_values_non_cs_label'] = perm_p_values_non_cs_label[i]
            subject_df.loc[subject_df['Base_Name'] == base_name, 'perm_p_values_cs_label'] = perm_p_values_cs_label[i]
            subject_df.loc[subject_df['Base_Name'] == base_name, 'count_lower_diff_label_and_acquisition_non_cs'] = count_lower_diff_label_and_acquisition_non_cs[i]
            subject_df.loc[subject_df['Base_Name'] == base_name, 'total_comparisons_label_and_acquisition_non_cs'] = total_comparisons_label_and_acquisition_non_cs[i]
            subject_df.loc[subject_df['Base_Name'] == base_name, 'count_lower_diff_label_and_acquisition_cs'] = count_lower_diff_label_and_acquisition_cs[i]
            subject_df.loc[subject_df['Base_Name'] == base_name, 'total_comparisons_label_and_acquisition_cs'] = total_comparisons_label_and_acquisition_cs[i]
            subject_df.loc[subject_df['Base_Name'] == base_name, 'count_lower_diff_acquisition_non_cs'] = count_lower_diff_acquisition_non_cs[i]
            subject_df.loc[subject_df['Base_Name'] == base_name, 'total_comparisons_acquisition_non_cs'] = total_comparisons_acquisition_non_cs[i]
            subject_df.loc[subject_df['Base_Name'] == base_name, 'count_lower_diff_acquisition_cs'] = count_lower_diff_acquisition_cs[i]
            subject_df.loc[subject_df['Base_Name'] == base_name, 'total_comparisons_acquisition_cs'] = total_comparisons_acquisition_cs[i]
    else:
        logging.warning("Mismatch in number of CS and non-CS samples for permutation tests.")

    return subject_df



# =============================================================================
# Functions for visualizing data
# =============================================================================

def visualize_data(df, save_path):
    """
    Visualize data using plots and save them to a specified directory.
    
    Args:
    - df: DataFrame containing the data
    - save_path: Directory path to save the plots
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    
    # Boxplot of anomaly scores by label (healthy vs. anomalous)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Label', y='anomaly_score', data=df, palette=custom_palette)
    plt.title("Boxplot of Anomaly Scores by Label")
    plt.xlabel("Label (0: Healthy, 1: Anomalous)")
    plt.ylabel("Anomaly Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "boxplot_anomaly_scores.png"), dpi =400)  # Save the plot
    plt.close()  # Close the plot to clear memory
    
    # Violin plot of anomaly scores by label
    plt.figure(figsize=(8, 6))
    # Add a dummy column for plotting; all data points have the same x-axis value
    df['dummy'] = ' '
    #sns.violinplot(x='Label', y='anomaly_score', hue='Label', data=df, palette=custom_palette, cut=0, split=True)
    ax = sns.violinplot(x='dummy', y='anomaly_score', hue='Label', data=df, split=True, inner="quart", palette=custom_palette, cut=0)
    ax.set_xlabel('')  # Remove the dummy x-label
    plt.title("Violin Plot of Anomaly Scores by Label")
    plt.xlabel("Label (0: Healthy, 1: Anomalous)")
    plt.ylabel("Anomaly Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "violinplot_anomaly_scores.png"), dpi =400)  # Save the plot
    plt.close()  # Close the plot to clear memory
    
    # Bar plot of mean anomaly scores by label
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Label', y='anomaly_score', data=df, estimator=np.mean, palette=custom_palette)
    plt.title("Bar Plot of Mean Anomaly Scores by Label")
    plt.xlabel("Label (0: Healthy, 1: Anomalous)")
    plt.ylabel("Mean Anomaly Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "barplot_mean_anomaly_scores_ci.png"), dpi =400)  # Save the plot
    plt.close()  # Close the plot to clear memory


    # Categorical boxplot of anomaly scores by label and sex
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Label', y='anomaly_score', hue='Sex', data=df, palette=custom_palette)
    plt.title("Categorical Boxplot of Anomaly Scores by Label and Sex")
    plt.xlabel("Label (0: Healthy, 1: Anomalous)")
    plt.ylabel("Anomaly Score")
    plt.tight_layout()
    plt.legend(title='Sex')
    plt.savefig(os.path.join(save_path, "categorical_boxplot_anomaly_scores_sex.png"), dpi =400)  # Save the plot
    plt.close()  # Close the plot to clear memory
    
    # Create age groups based on age ranges
    bins = [0, 30, 60, float('inf')]
    labels = ['<30', '30-60', '60>']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    # Categorical boxplot of anomaly scores by label and age group
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Label', y='anomaly_score', hue='Age Group', data=df)
    plt.title("Categorical Boxplot of Anomaly Scores by Label and Age Group")
    plt.xlabel("Label (0: Healthy, 1: Anomalous)")
    plt.ylabel("Anomaly Score")
    plt.tight_layout()
    plt.legend(title='Age Group')
    plt.savefig(os.path.join(save_path, "categorical_boxplot_anomaly_scores_age_group.png"), dpi =400)  # Save the plot
    plt.close()  # Close the plot to clear memory
    
    # Pairplot with emphasis on age group
    plt.figure(figsize=(10, 6))
    sns.pairplot(df, hue='Age Group', palette='husl', vars=['Age', 'anomaly_score'])
    plt.title("Pairplot of Anomaly Scores and Age Group")
    plt.savefig(os.path.join(save_path, "pairplot_anomaly_scores_age_group.png"), dpi =400)  # Save the plot
    plt.close()  # Close the plot to clear memory
    
    # Scatter plot with hue, emphasis on age group
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Age', y='anomaly_score', hue='Age Group', palette='husl', s=130)
    plt.title("Scatter Plot of Anomaly Scores by Age, with Age Group Differentiation")
    plt.xlabel("Age")
    plt.ylabel("Anomaly Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "scatterplot_anomaly_scores_age_group.png"), dpi =400)  # Save the plot
    plt.close()  # Close the plot to clear memory
    
    # Bar plot with hue, emphasis on age group
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Label', y='anomaly_score', hue='Age Group', data=df, estimator=np.mean)
    plt.title("Bar Plot of Mean Anomaly Scores by Label and Age Group")
    plt.xlabel("Label (0: Healthy, 1: Anomalous)")
    plt.ylabel("Mean Anomaly Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "barplot_mean_anomaly_scores_age_group.png"), dpi =400)  # Save the plot
    plt.close()  # Close the plot to clear memory
    
    # Violin plot with hue, emphasis on age group
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Label', y='anomaly_score', hue='Age Group', data=df)
    plt.title("Violin Plot of Anomaly Scores by Label and Age Group")
    plt.xlabel("Label (0: Healthy, 1: Anomalous)")
    plt.ylabel("Anomaly Score")
    plt.tight_layout()
    plt.legend(title='Age Group')
    plt.savefig(os.path.join(save_path, "violinplot_anomaly_scores_age_group.png"), dpi =400)  # Save the plot
    plt.close()  # Close the plot to clear memory
    
    # Histogram with facet grid, emphasis on age group
    g = sns.FacetGrid(df, col="Label", hue="Age Group", palette='husl')
    g.map(sns.histplot, "anomaly_score", kde=True)
    g.set_axis_labels("Anomaly Score", "Frequency")
    g.add_legend(title='Age Group')
    plt.savefig(os.path.join(save_path, "histogram_anomaly_scores_age_group.png"), dpi =400)  # Save the plot
    plt.close()  # Close the plot to clear memory

    # Plot the distribution of anomaly scores by age, not age group (for cases only)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Age', y='anomaly_score', data=df[df['Label'] == 'Cases'], ci='sd')
    plt.title("Distribution of Anomaly Scores by Age (Cases)")
    plt.xlabel("Age")
    plt.ylabel("Anomaly Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "lineplot_anomaly_scores_age_cases.png"), dpi =400)  # Save the plot
    plt.close()  # Close the plot to clear memory

    # Also plot in scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='anomaly_score', data=df[df['Label'] == 'Cases'], s=130)
    plt.title("Scatter Plot of Anomaly Scores by Age (Cases)")
    plt.xlabel("Age")
    plt.ylabel("Anomaly Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "scatterplot_anomaly_scores_age_cases.png"), dpi =400)  # Save the plot
    plt.close()  # Close the plot to clear memory

    # Plot the distribution of anomaly scores by age, for all subjects
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Age', y='anomaly_score', data=df, hue='Label', ci='sd', palette=custom_palette)
    plt.title("Distribution of Anomaly Scores by Age")
    plt.xlabel("Age")
    plt.ylabel("Anomaly Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "lineplot_anomaly_scores_age.png"), dpi =400)  # Save the plot
    plt.close()  # Close the plot to clear memory

    # Also plot in scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='anomaly_score', data=df, hue='Label', style='Label', palette=custom_palette, s=130)
    plt.title("Scatter Plot of Anomaly Scores by Age")
    plt.xlabel("Age")
    plt.ylabel("Anomaly Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "scatterplot_anomaly_scores_age.png"), dpi =400)  # Save the plot
    plt.close()  # Close the plot to clear memory
    # Plot subject-wise anomaly scores in order and hue compressed sensing
    df_sorted = df.sort_index()
    # Identify the position to separate controls and cases
    separator_position = df_sorted[df_sorted['Label'] == 'Cases'].index[0]
    plt.figure(figsize=(15, 8))
    
    ax = sns.barplot(x=df_sorted.index, y='anomaly_score', data=df_sorted, hue='Compressed_sensing', palette=custom_cs_palette, dodge=False)
    plt.axvline(x=df_sorted.index.get_loc(separator_position) - 0.5, color='red', linestyle='--')
    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.legend(title='Compressed Sensing')
    plt.ylabel('Anomaly Score')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'subject_wise_anomaly_scores_by_cs.png'), dpi=400)
    # Same in log scale
    ax.set_yscale('log')
    plt.savefig(os.path.join(save_path, 'subject_wise_anomaly_scores_by_cs_log.png'), dpi=400)
    plt.close()

    # Extract duplicates
    df_duplicates = df_sorted[df_sorted.duplicated('Base_Name', keep=False)].sort_values('Base_Name')
    separator_position = df_duplicates[df_duplicates['Label'] == 'Cases'].index[0]
    plt.figure(figsize=(15, 8))
    
    ax = sns.barplot(x=df_duplicates.index, y='anomaly_score', data=df_duplicates, hue='Compressed_sensing', palette=custom_cs_palette, dodge=False)
    plt.axvline(x=df_duplicates.index.get_loc(separator_position) - 0.5, color='red', linestyle='--')
    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.legend(title='Compressed Sensing')
    plt.ylabel('Anomaly Score', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'duplicate_subjects_anomaly_scores.png'), dpi=400)
    # Same in log scale
    ax.set_yscale('log')
    plt.savefig(os.path.join(save_path, 'duplicate_subjects_anomaly_scores_log.png'), dpi=400)
    plt.close()


def plot_heatmap_comparaison(healthy_scores, anomalous_scores, save_dir=None):

    agg_healthy_scores = np.mean(healthy_scores, axis=0)
    agg_anomalous_scores = np.mean(anomalous_scores, axis=0)

    vmin = np.min(np.concatenate((agg_healthy_scores.mean(axis = (1,2,3)).flatten(), agg_anomalous_scores.mean(axis = (1,2,3)).flatten())))
    vmax = np.max(np.concatenate((agg_healthy_scores.mean(axis = (1,2,3)).flatten(), agg_anomalous_scores.mean(axis = (1,2,3)).flatten())))

        # Create heatmap for healthy scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 12), sharey=True)
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])

    sns.heatmap(np.mean(agg_healthy_scores, axis=(1, 2, 3)), cmap='viridis', xticklabels=np.arange(healthy_scores.shape[-1]),
                yticklabels=np.arange(healthy_scores.shape[1]), cbar=True,cbar_ax=cbar_ax, ax=ax1, vmin=vmin, vmax=vmax)

    # Adjusting the ticks
    ax1.set_xticks(np.arange(0, healthy_scores.shape[-1], 2))  # Skip every 2 time steps
    ax1.set_xticklabels(np.arange(0, healthy_scores.shape[-1], 2))
    ax1.set_yticks(np.arange(0, agg_healthy_scores.shape[0], 5))  # Skip every 5 z slices
    ax1.set_yticklabels(np.arange(0, agg_healthy_scores.shape[0], 5))

    ax1.set_xlabel('Time', fontsize=20)
    ax1.set_ylabel('Aortic slices', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    
    # Increase the colorbar tick labels size
    cbar_ax.tick_params(labelsize=16)

    # Create heatmap for anomalous scores
    sns.heatmap(np.mean(agg_anomalous_scores, axis=(1, 2, 3)), cmap='viridis', xticklabels=np.arange(anomalous_scores.shape[-1]),
                yticklabels=np.arange(anomalous_scores.shape[1]), cbar=False, ax=ax2, vmin=vmin, vmax=vmax)
    
    
    ax2.set_xticks(np.arange(0, healthy_scores.shape[-1], 2))  # Skip every 2 time steps
    ax2.set_xticklabels(np.arange(0, healthy_scores.shape[-1], 2))
    ax2.set_yticks(np.arange(0, agg_healthy_scores.shape[0], 5))  # Skip every 5 z slices
    ax2.set_yticklabels(np.arange(0, agg_healthy_scores.shape[0], 5))
    ax2.set_xlabel('Time', fontsize=20)
    ax2.set_ylabel('', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    # Save the plot
    if save_dir:
        plt.savefig(save_dir + f'/heatmap_combined_anomaly_scores_z_slices_vs_time.png', dpi=400)
        plt.close()
    else:
        plt.show()
    


def plot_heatmap(scores, save_dir=None, scale_colors = False, name_extension = ''):
    
    scores = np.mean(scores, axis=0)
    #(64, 1, 32, 32, 24)
    # Compute vmin and vmax for the given scores
    if scale_colors:
        vmin = np.percentile(scores.mean(axis=(1,2,3)), 2)
        vmax = np.percentile(scores.mean(axis=(1,2,3)), 98)

    plt.figure(figsize=(10, 14))

    ax = sns.heatmap(np.mean(scores, axis=(1, 2, 3)), cmap='viridis', xticklabels=np.arange(scores.shape[-1]),
                     yticklabels=np.arange(scores.shape[0]), vmin=vmin if scale_colors else None, vmax=vmax if scale_colors else None)

    # Adjusting the ticks
    ax.set_xticks(np.arange(0, scores.shape[-1], 2))  # Skip every 2 time steps
    ax.set_xticklabels(np.arange(0, scores.shape[-1], 2))
    ax.set_yticks(np.arange(0, scores.shape[0], 5))  # Skip every 5 z slices
    ax.set_yticklabels(np.arange(0, scores.shape[0], 5))
    plt.xlabel('Time')
    plt.ylabel('Aortic slices')
    plt.title('Anomaly scores heatmap (mean) - aortic slices vs Time')

    if save_dir:
        
        plt.savefig(save_dir + f'/{name_extension}_heatmap_z_slices_vs_time.png', dpi=400, bbox_inches='tight')  # Save high-quality image
        plt.close()
    else:
        plt.show()

def plot_scores(healthy_scores, sick_scores, results_dir, level,  data="test", deformation=None, note=None, through_time=False):
    save_dir = get_save_dir(data, results_dir, deformation)
    
    # axis: [#subjects, z_slice,c,x,y,t]

    if through_time:
        axis_values = (1, 2, 3, 4, 5) if level == 'patient' else (1, 2, 3, 4)
    else:
        axis_values = (1, 2, 3, 4, 5) if level == 'patient' else (2, 3, 4, 5)

    sick_means = np.mean(sick_scores, axis=axis_values)
    sick_stds = sick_scores.std(axis=axis_values)
    healthy_means = np.mean(healthy_scores, axis=axis_values)
    healthy_stds = healthy_scores.std(axis=axis_values)
    
    if level == 'patient':
        plot_patient_level(sick_means, sick_stds, healthy_means, healthy_stds, save_dir, data, note, through_time)
    elif level == 'imagewise':
        indexes =plot_imagewise_level(sick_means, sick_stds, healthy_means, healthy_stds, save_dir, data, note, through_time)
        return indexes
    else:
        print("Invalid level. Please enter 'patient' or 'imagewise'.")




def get_save_dir(data, results_dir, deformation, note = None):
    if deformation:
        return os.path.join(project_code_root, results_dir, data, deformation)
    elif data == 'test':
        dir_path = os.path.join(project_code_root, results_dir)
        if note:
            dir_path = os.path.join(dir_path, 'thesis_plots')
            make_dir_safely(dir_path)
        return dir_path
    else:
        return os.path.join(project_code_root, results_dir, data)
    
def plot_lineplot(df, save_dir, data, note, through_time=False, palette=None, errorbar=None):
    # Create the line plot
    plt.figure(figsize=(12, 6))
    if errorbar:
        line_plot = sns.lineplot(data=df, x='imagewise', y='Mean Score', hue='Subject Status', palette=palette, err_style="bars", errorbar=errorbar)
    else:
        line_plot = sns.lineplot(data=df, x='imagewise', y='Mean Score', hue='Subject Status', style="Subject", palette=palette)
    line_plot.legend_.remove()  # Remove the original legend
    plt.xlabel('Time' if through_time else 'Aortic slices')
    # Create a new legend only for Subject Status (Color)
    handles, labels = line_plot.get_legend_handles_labels()
    line_plot.legend(handles=handles[:3], labels=labels[:3], loc='upper left')
    plt.ylabel('Mean Anomaly Score')
    plt.tight_layout()
    if errorbar == None:
        save_name = f'{data}_mean_imagewise_scores_by_patient'
    else:
        save_name = f'{data}_mean_imagewise_scores_{errorbar}'
    if note:        
        plt.savefig(os.path.join(save_dir, f'{save_name}_{note}.png'), bbox_inches='tight', dpi=400)
        # Set y axis to log scale
        plt.yscale('log')
        plt.savefig(os.path.join(save_dir, f'{save_name}_{note}_log.png'), bbox_inches='tight', dpi=400)
    else:
        plt.savefig(os.path.join(save_dir, f'{save_name}{"_through_time" if through_time else ""}.png'), bbox_inches='tight', dpi=400)
        # Set y axis to log scale
        plt.yscale('log')
        plt.savefig(os.path.join(save_dir, f'{save_name}{"_through_time" if through_time else ""}_log.png'), bbox_inches='tight', dpi=400)
    plt.close()

def plot_imagewise_level(sick_means, sick_stds, healthy_means, healthy_stds, save_dir, data, note, through_time=False):
    # Prepare the data for seaborn
    # imagewise here refers to the z slice or time depending on through_time flag
    sick_df = pd.DataFrame({
        'Subject Status': np.repeat('Cases', sick_means.size),
        'Subject': np.repeat(np.arange(sick_means.shape[0]), sick_means.shape[1]),
        'imagewise': np.tile(np.arange(1, sick_means.shape[1] + 1), sick_means.shape[0]),
        'Mean Score': sick_means.flatten(),
        'Std Deviation': sick_stds.flatten()
    })

    healthy_df = pd.DataFrame({
        'Subject Status': np.repeat('Controls', healthy_means.size),
        'Subject': np.repeat(np.arange(healthy_means.shape[0]), healthy_means.shape[1]),
        'imagewise': np.tile(np.arange(1, healthy_means.shape[1] + 1), healthy_means.shape[0]),
        'Mean Score': healthy_means.flatten(),
        'Std Deviation': healthy_stds.flatten()
    })

    # Concatenate the two dataframes
    df = pd.concat([sick_df, healthy_df])


    plot_lineplot(df, save_dir, data, note, through_time=through_time, palette=custom_palette, errorbar=None)
    plot_lineplot(df, save_dir, data, note, through_time=through_time, palette=custom_palette, errorbar='sd')
    plot_lineplot(df, save_dir, data, note, through_time=through_time, palette=custom_palette, errorbar='se')
    plot_lineplot(df, save_dir, data, note, through_time=through_time, palette=custom_palette, errorbar='ci')
    
            
    # If note exists we want to return the indexes of the slices with the min nad max score for the healthy and sick patient (there is only one patient in the healthy group and one in the sick group)
    # Save them in dictionary to return
    if note:
        # Get the indexes of the slices with the min and max score for the healthy and sick patient
        sick_min_index = df.loc[df['Subject Status'] == 'Cases']['Mean Score'].idxmin()
        sick_max_index = df.loc[df['Subject Status'] == 'Cases']['Mean Score'].idxmax()
        healthy_min_index = df.loc[df['Subject Status'] == 'Controls']['Mean Score'].idxmin()
        healthy_max_index = df.loc[df['Subject Status'] == 'Controls']['Mean Score'].idxmax()
        # Get the imagewise indexes
        sick_min_index = df.loc[df['Subject Status'] == 'Cases']['imagewise'][sick_min_index]
        sick_max_index = df.loc[df['Subject Status'] == 'Cases']['imagewise'][sick_max_index]
        healthy_min_index = df.loc[df['Subject Status'] == 'Controls']['imagewise'][healthy_min_index]
        healthy_max_index = df.loc[df['Subject Status'] == 'Controls']['imagewise'][healthy_max_index]
        # Save them in dictionary
        indexes = {'sick_min_index': sick_min_index, 'sick_max_index': sick_max_index, 'healthy_min_index': healthy_min_index, 'healthy_max_index': healthy_max_index}
        return indexes
    plt.close()


def plot_barplot(df, save_dir, data, note=None, with_error_bars=False, palette=None):
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot with error bars if specified
    if with_error_bars:
        sns.barplot(x='Subject', y='Mean Score', hue='Status', data=df, yerr=df['Std Deviation'], capsize=.2, palette=palette)
        errorbar_label = '_with_std_bars'  
    else:
        sns.barplot(x='Subject', y='Mean Score', hue='Status', data=df, capsize=.2, palette=palette)
        errorbar_label = ''  
        
    plt.xlabel('')
    plt.xticks(rotation=90)  # Rotate x-axis labels to avoid overlap
    plt.ylabel('Mean Anomaly Score')
    plt.tight_layout()
    
    # Save plot
    if note:
        plt.savefig(os.path.join(save_dir, f'{data}_patient_mean_wise_scores_{note}{errorbar_label}.png'), bbox_inches='tight', dpi=400)
    else:
        plt.savefig(os.path.join(save_dir, f'{data}_patient_mean_wise_scores{errorbar_label}.png'), bbox_inches='tight', dpi=400)
        
        
    plt.close()

def plot_patient_level(sick_means, sick_stds, healthy_means, healthy_stds, save_dir, data, note, through_time=False):
    df_sick = pd.DataFrame({
            'Subject': ['Cases '+str(i) for i in range(len(sick_means))],
            'Mean Score': sick_means,
            'Std Deviation': sick_stds,
            'Status': ['Cases']*len(sick_means)
        })

    # Create a dataframe for healthy patients
    df_healthy = pd.DataFrame({
        'Subject': ['Controls '+str(i) for i in range(len(healthy_means))],
        'Mean Score': healthy_means,
        'Std Deviation': healthy_stds,
        'Status': ['Controls']*len(healthy_means)
    })

    # Concatenate the dataframes
    df = pd.concat([df_sick, df_healthy])

    # Plot the barplot
    plot_barplot(df, save_dir, data, note, with_error_bars=True, palette=custom_palette)
    plot_barplot(df, save_dir, data, note, with_error_bars=False, palette=custom_palette)
    


# =============================================================================
# Functions used with the conditioning network - used during experimentation phase
# ============================================================================= 

def get_random_and_neighbour_indices(indices, subject_length):    

    neighbour_indices = []
    for idx in indices:
        subject_num = idx // subject_length
        slice_num = idx % subject_length
        prev_idx = subject_num * subject_length + max(0, slice_num - 1)
        next_idx = subject_num * subject_length + min(subject_length - 1, slice_num + 1)
        neighbour_indices.append((prev_idx, idx, next_idx))
    
    return neighbour_indices
def get_images_from_indices(images, indices):
    images_with_neighbours = []
    for prev_idx, idx, next_idx in indices:
        prev_image = images[prev_idx] if prev_idx != idx else np.zeros_like(images[0])
        image = images[idx]
        next_image = images[next_idx] if next_idx != idx else np.zeros_like(images[0])
        images_with_neighbours.append([prev_image, image, next_image])

    return images_with_neighbours
def get_combined_images(images, indices):
    images_with_neighbours = get_images_from_indices(images, indices)
    combined_images = np.stack(images_with_neighbours, axis=0)
    
    return combined_images

# =============================================================================
# Functions for loading, saving data and running infenrece
# =============================================================================

def load_config(config_path):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): The file path to the configuration file.

    Returns:
        dict: A dictionary containing the loaded configuration.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_start_end_indices(experiment_name):
    """
    Determines the start and end indices for time slices based on the compressed sensing (CS) usage specified in the experiment name.

    Args:
        experiment_name (str): Name of the experiment, which contains indications of CS usage.

    Returns:
        tuple: A tuple containing the start and end indices for time slices.

    Raises:
        ValueError: If the experiment name does not conform to expected CS usage indicators.
    """
    if 'only_cs' in experiment_name:
        return 0, 17
    elif 'without_cs' in experiment_name:
        return 0, 34
    elif 'with_cs' in experiment_name:
        return 0, 54
    else:
        raise ValueError("Experiment name does not specify CS usage properly")

def load_model(experiment_name, best_exp_path, in_channels, out_channels):
    """
    Loads a model from a saved state dictionary, initializing it with specified parameters and settings it to evaluation mode.

    Args:
        experiment_name (str): The name of the experiment, used to initialize the model.
        best_exp_path (str): Path to the saved model state dictionary.
        in_channels (int): Number of input channels for the model.
        out_channels (int): Number of output channels for the model.

    Returns:
        tuple: A tuple containing the loaded model and the device it is loaded on.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize_model(experiment_name, in_channels, out_channels)
    model.load_state_dict(torch.load(best_exp_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def is_compressed_sensing(subject_name, config):
    """
    Determines whether a subject's data was acquired using compressed sensing (CS).

    Args:
        subject_name (str): The name of the subject.
        config (dict): Configuration dictionary containing paths and settings.

    Returns:
        bool: True if the subject's data was acquired using CS, False otherwise.

    Raises:
        ValueError: If the subject's data cannot be located.
    """
    subject_name_npy = subject_name + '.npy'
    if subject_name.__contains__('MACDAVD_1'):
        if subject_name_npy in os.listdir(os.path.join(config['img_path'].replace("patients", "controls"))):
            return False
        elif subject_name_npy in os.listdir(os.path.join(config['img_path'].replace("patients", "controls") + '_compressed_sensing')):
            return True
        else:
            raise ValueError("Cannot find the individual")
    else:
        if subject_name_npy in os.listdir(os.path.join(config['img_path'])):
            return False
        elif subject_name_npy in os.listdir(os.path.join(config['img_path'] + '_compressed_sensing')):
            return True
        else:
            raise ValueError("Cannot find the individual")

def infer_batch(input_dict, model, model_type, subject_reconstruction):
    """
    Processes a batch of data through a model, optionally capturing the reconstruction for analysis.

    Args:
        input_dict (dict): Dictionary containing input data for the model.
        model (torch.Module): The model to process the input.
        model_type (str): Type of the model ('self-supervised' or other types indicating the output handling).
        subject_reconstruction (list): A list to which reconstruction data will be appended if applicable.

    Returns:
        tuple: A tuple containing the output images from the model and possibly updated reconstruction data.
    """
    with torch.no_grad():
        model.eval()
        output_dict = model(input_dict)
        if model_type == 'self-supervised':
            output_images = torch.sigmoid(output_dict['decoder_output'])
        else:
            logging.info('Reconstruction based model')
            model_output = output_dict['decoder_output']
            output_images = torch.abs(model_output - input_dict['batch'])
            subject_reconstruction.append(output_images.cpu().detach().numpy())
        return output_images, subject_reconstruction

def setup_directories(results_dir):
    """
    Sets up directories for storing results, outputs, and inputs of tests.

    Args:
        results_dir (str): The base directory to set up the test results.

    Returns:
        tuple: A tuple containing paths to the test results directory, outputs directory, and inputs directory.
    """
    make_dir_safely(results_dir)
    results_dir_test = results_dir + '/' + 'test'
    make_dir_safely(results_dir_test)
    results_dir_subject = os.path.join(results_dir_test, "outputs")
    inputs_dir_subject = os.path.join(results_dir_test, "inputs")
    make_dir_safely(inputs_dir_subject)
    make_dir_safely(results_dir_subject)
    return results_dir_test, results_dir_subject, inputs_dir_subject

def compute_scores_metrics(healthy_scores, anomalous_scores):
    """
    Computes and logs the mean and standard deviation of anomaly scores for healthy and anomalous subjects.

    Args:
        healthy_scores (list): A list of anomaly scores for healthy subjects.
        anomalous_scores (list): A list of anomaly scores for anomalous subjects.

    Returns:
        tuple: A tuple containing arrays of healthy and anomalous scores.
    """
    healthy_scores = np.array(healthy_scores)
    healthy_mean_anomaly_score = np.mean(healthy_scores)
    healthy_std_anomaly_score = np.std(healthy_scores)

    anomalous_scores = np.array(anomalous_scores)
    anomalous_mean_anomaly_score = np.mean(anomalous_scores)
    anomalous_std_anomaly_score = np.std(anomalous_scores)

    logging.info('============================================================')
    logging.info('Control subjects anomaly_score: {} +/- {:.4e}'.format(healthy_mean_anomaly_score, healthy_std_anomaly_score))
    logging.info('Anomalous subjects anomaly_score: {} +/- {:.4e}'.format(anomalous_mean_anomaly_score, anomalous_std_anomaly_score))
    logging.info('============================================================')
    return healthy_scores, anomalous_scores

def plot_all_scores(healthy_scores, anomalous_scores, results_dir):
    """
    Plots and logs the anomaly scores for healthy and anomalous subjects.

    Args:
        healthy_scores (np.array): An array of healthy subject scores.
        anomalous_scores (np.array): An array of anomalous subject scores.
        results_dir (str): The directory where plots should be saved.

    Performs:
        Plotting operations that generate visual representations of the scores.
    """
    logging.info('Plotting scores through slices...')
    plot_scores(healthy_scores, anomalous_scores, results_dir, level='patient')
    plot_scores(healthy_scores, anomalous_scores, results_dir, level='imagewise')
    logging.info('Plotting scores through time...')
    plot_scores(healthy_scores, anomalous_scores, results_dir, level='imagewise', through_time=True)

def compute_region_metrics(subject_df, label, results_summary):
    """
    Computes and updates the results summary with metrics for a specified label based on the data in a subject DataFrame.

    Args:
        subject_df (pandas.DataFrame): The DataFrame containing subject data.
        label (str): The label under which metrics are calculated.
        results_summary (dict): A dictionary where the results will be updated.

    Logs:
        AUC-ROC values and related metrics for the specified label.
    """
    y_true, y_scores, auc_roc = compute_metrics(subject_df, label, results_summary)
    logging.info(f"{label} AUC-ROC: {auc_roc:.4f}")


def gather_experiment_paths(model_dir, specific_times):
    """
    Gather all valid experiment paths based on the specific times listed in the configuration.

    Args:
        model_dir (str): The directory containing all model experiments.
        specific_times (list): A list of specific times to filter the experiments.

    Returns:
        list: A list of paths to the experiments that match the specific times.
        set: A set of found times that matched the directories.
    """
    list_of_experiments_paths = []
    found_times = set()

    for exp_dir in os.listdir(model_dir):
        for time in specific_times:
            if time in exp_dir:
                exp_path = os.path.join(model_dir, exp_dir)
                list_of_experiments_paths.append(exp_path)
                found_times.add(time)
                break

    # Check for specific times that were not found in any directory
    missing_times = set(specific_times) - found_times
    if missing_times:
        logging.warning(f"No directories found for specified times: {', '.join(missing_times)}")
    else:
        logging.info("All specified times matched with directories.")

    return list_of_experiments_paths
