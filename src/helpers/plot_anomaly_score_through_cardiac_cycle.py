"""
Plotting Mean Anomaly Scores Through the Cardiac Cycle

This script generates plots for the mean anomaly scores of subjects through the cardiac cycle
from anomaly detection experiments. It creates individual plots for each subject and a combined
plot comparing conditions (CAD and Arrhythmia).

Functions:
- plot_subject_mean_anomaly_scores: Generates and saves plots for anomaly scores of subjects.
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import make_dir_safely

# Define your custom palette
custom_palette = {'CAD': '#1f77b4', 'Arrhythmia': '#ff7f0e'}

# Update matplotlib parameters
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.titlesize'] = 16

def plot_subject_mean_anomaly_scores(experiment_paths, palette=custom_palette):
    """
    Generates and saves plots for the mean anomaly scores of subjects through the cardiac cycle.

    Parameters:
    - experiment_paths (list of str): List of paths to the experiment directories.
    - palette (dict): Custom color palette for the conditions.

    Returns:
    None
    """
    for experiment_path in experiment_paths:
        output_dir = os.path.join(experiment_path, 'test', 'outputs')
        save_dir = os.path.join(experiment_path, 'test', 'anomaly_score_through_cardiac_cycle')
        # Make save directory
        make_dir_safely(save_dir)

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            print(f"Output directory {output_dir} does not exist. Skipping...")
            continue

        cad_data = []
        arrhythmia_data = []

        for file_name in os.listdir(output_dir):
            if file_name.endswith('_anomaly_scores.npy'):
                subject_id = file_name.split('_anomaly_scores')[0]
                file_path = os.path.join(output_dir, file_name)

                # Load anomaly scores
                anomaly_scores = np.load(file_path)

                # Calculate mean anomaly score over time (last dimension)
                mean_scores = anomaly_scores.mean(axis=(0, 1, 2, 3))

                # Create a DataFrame for the subject
                df = pd.DataFrame({
                    'Time': np.arange(1, mean_scores.size + 1),
                    'Mean Anomaly Score': mean_scores,
                    'Subject': subject_id
                })

                if subject_id.startswith('MACDAVD_2'):
                    df['Condition'] = 'CAD'
                    cad_data.append(df)
                elif subject_id.startswith('MACDAVD_3'):
                    df['Condition'] = 'Arrhythmia'
                    arrhythmia_data.append(df)

                # Create plot for individual subject
                plt.figure(figsize=(12, 6))
                sns.lineplot(data=df, x='Time', y='Mean Anomaly Score', hue='Subject')
                plt.xlabel('Time')
                plt.ylabel('Mean Anomaly Score')
                plt.tight_layout()

                # Save the individual plot
                save_path = os.path.join(save_dir, f'{subject_id}_mean_anomaly_score.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=400)
                plt.close()

        # Combine and plot CAD and Arrhythmia subjects
        if cad_data or arrhythmia_data:
            combined_df = pd.concat(cad_data + arrhythmia_data)

            plt.figure(figsize=(12, 6))
            sns.lineplot(data=combined_df, x='Time', y='Mean Anomaly Score', hue='Condition', palette=palette, linewidth=2, alpha=0.8)
            plt.xlabel('Time')
            plt.ylabel('Mean Anomaly Score')
            plt.legend(title='Condition')
            plt.tight_layout()

            # Save the combined plot
            combined_save_path = os.path.join(save_dir, 'CAD_vs_Arrhythmia_mean_anomaly_score.png')
            plt.savefig(combined_save_path, bbox_inches='tight', dpi=400)
            plt.close()

if __name__ == '__main__':
    # List of experiment paths
    experiment_paths = [
        "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/Results/Evaluation/vae_convT/masked_slice/20240518-2136_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3",
    ]

    # Generate and save the plots
    plot_subject_mean_anomaly_scores(experiment_paths, custom_palette)