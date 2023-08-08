import numpy as np
import logging
from helpers.metrics import RMSE, compute_auc_roc_score, compute_average_precision_score
from sklearn.metrics import roc_curve, auc, average_precision_score
import os
from scipy import stats

from config import system_eval as config_sys
project_code_root = config_sys.project_code_root

from utils import make_dir_safely

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt



def plot_scores(healthy_scores, sick_scores, results_dir, level,  data="test", deformation=None, note=None):
    save_dir = get_save_dir(data, results_dir, deformation)
    
    # Calculate common metrics
    axis_values = (1, 2, 3, 4, 5) if level == 'patient' else (2, 3, 4, 5)
    sick_means = np.mean(sick_scores, axis=axis_values)
    sick_stds = sick_scores.std(axis=axis_values)
    healthy_means = np.mean(healthy_scores, axis=axis_values)
    healthy_stds = healthy_scores.std(axis=axis_values)
    
    if level == 'patient':
        plot_patient_level(sick_means, sick_stds, healthy_means, healthy_stds, save_dir, data, note)
    elif level == 'imagewise':
        indexes =plot_imagewise_level(sick_means, sick_stds, healthy_means, healthy_stds, save_dir, data, note)
        return indexes
    else:
        print("Invalid level. Please enter 'patient' or 'imagewise'.")
def compute_auc(healthy_scores, anomalous_scores, results_dir, format_wise= "2Dslice", agg_function = np.mean, data = "test", deformation = None):

                
                if data == "test":

                    if format_wise == "patient_wise":
                        # Take mean
                        healthy_scores = agg_function(healthy_scores, axis=(1,2,3,4,5))
                        anomalous_scores = agg_function(anomalous_scores, axis=(1,2,3,4,5))
                    elif format_wise == "imagewise":
                        # Take mean
                        healthy_scores = agg_function(healthy_scores, axis=(2,3,4,5))
                        anomalous_scores = agg_function(anomalous_scores, axis=(2,3,4,5))
                    elif format_wise == "2Dslice":
                        # Take mean
                        healthy_scores = agg_function(healthy_scores, axis=(3,4))
                        anomalous_scores = agg_function(anomalous_scores, axis=(3,4))
                else:
                    if format_wise == "patient_wise":
                        healthy_scores = agg_function(healthy_scores, axis=(1,2,3,4,5))
                        anomalous_scores = agg_function(anomalous_scores, axis=(1,2,3,4,5))
                    elif format_wise == "imagewise":
                        healthy_scores = agg_function(healthy_scores, axis=(1,2,3,4))
                        anomalous_scores = agg_function(anomalous_scores, axis=(1,2,3,4))
                    elif format_wise == "2Dslice":
                        healthy_scores = agg_function(healthy_scores, axis=(1,2))
                        anomalous_scores = agg_function(anomalous_scores, axis=(1,2))

                

                y_true = np.concatenate((np.zeros(len(healthy_scores.flatten())), np.ones(len(anomalous_scores.flatten()))))
                y_scores = np.concatenate((healthy_scores.flatten(), anomalous_scores.flatten()))
                if len(np.unique(y_true)) == 1:
                    return logging.info('ROC - Same class for all - cannot compute - all {}'.format(np.unique(y_true)))
                else:
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                    auc_roc = auc(fpr, tpr)
                    logging.info('============================================================')
                    logging.info('AUC-ROC score {}: {:.2f}'.format(format_wise, auc_roc))
                    # Plot the ROC curve
                    logging.info('============================================================')
                    logging.info('Plotting ROC curve...')
                    logging.info('============================================================')
                    plt.figure()
                    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_roc:.2f}')
                    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC) Curve - {}'.format(format_wise))
                    plt.legend()

                    # save_dir
                    if deformation:
                        save_dir = os.path.join(project_code_root, results_dir, data, deformation)
                    else:
                        if data == 'test':
                            save_dir = os.path.join(project_code_root, results_dir)
                        else:
                            save_dir = os.path.join(project_code_root, results_dir, data)

                    if agg_function == np.mean:
                        plt.savefig(os.path.join(save_dir + '/' + f'{data}_mean_{format_wise}_ROC_curve.png'))
                    elif agg_function == np.sum:
                        plt.savefig(os.path.join(save_dir + '/' + f'{data}_sum_{format_wise}_ROC_curve.png'))
                    plt.close()
def compute_average_precision(healthy_scores, anomalous_scores, format_wise= False, agg_function = np.mean, data = "test"):
    if data == "test":

        if format_wise == "patient_wise":
            # Take mean
            healthy_scores = agg_function(healthy_scores, axis=(1,2,3,4,5))
            anomalous_scores = agg_function(anomalous_scores, axis=(1,2,3,4,5))
        elif format_wise == "imagewise":
            # Take mean
            healthy_scores = agg_function(healthy_scores, axis=(2,3,4,5))
            anomalous_scores = agg_function(anomalous_scores, axis=(2,3,4,5))
        elif format_wise == "2Dslice":
            # Take mean
            healthy_scores = agg_function(healthy_scores, axis=(3,4))
            anomalous_scores = agg_function(anomalous_scores, axis=(3,4))
    else:
        if format_wise == "patient_wise":
            healthy_scores = agg_function(healthy_scores, axis=(1,2,3,4,5))
            anomalous_scores = agg_function(anomalous_scores, axis=(1,2,3,4,5))
        elif format_wise == "imagewise":
            healthy_scores = agg_function(healthy_scores, axis=(1,2,3,4))
            anomalous_scores = agg_function(anomalous_scores, axis=(1,2,3,4))
        elif format_wise == "2Dslice":
            healthy_scores = agg_function(healthy_scores, axis=(1,2))
            anomalous_scores = agg_function(anomalous_scores, axis=(1,2))

    
    y_true = np.concatenate((np.zeros(len(healthy_scores.flatten())), np.ones(len(anomalous_scores.flatten()))))
    y_scores = np.concatenate((healthy_scores.flatten(), anomalous_scores.flatten()))
    ap = average_precision_score(y_true, y_scores)
    if agg_function == np.mean:
        logging.info('Average precision score {}: {:.2f}'.format(format_wise, ap))
    elif agg_function == np.sum:
        logging.info('Sum precision score {}: {:.2f}'.format(format_wise, ap))
    logging.info('============================================================')


def validation_metrics(anomaly_scores, masks, deformation = None):
                ## Patient wise
                anomalous_subjects_indexes = np.max(masks, axis=(1,2,3,4,5)).astype(bool)
                healthy_subjects_indexes = np.logical_not(np.max(masks, axis=(1,2,3,4,5)).astype(bool))
                healthy_scores = anomaly_scores[healthy_subjects_indexes]
                anomalous_scores = anomaly_scores[anomalous_subjects_indexes]
                if (len(healthy_scores) !=0) and (len(healthy_scores) !=0):
                    compute_auc(healthy_scores, anomalous_scores, format_wise= "patient_wise", agg_function=np.mean, data = "validation",deformation = deformation)
                    compute_average_precision(healthy_scores, anomalous_scores, format_wise= "patient_wise", agg_function=np.mean, data = "validation")
                else:
                    logging.info('Patient Wise: Same class for all - cannot compute')
                plot_scores(healthy_scores, anomalous_scores, level = 'patient', agg_function= np.mean, data="validation",deformation = deformation)
                plot_scores(healthy_scores, anomalous_scores, level = 'imagewise', agg_function= np.mean, data="validation",deformation = deformation)

                ## Image wise
                anomalous_subjects_indexes = np.max(masks, axis=(2,3,4,5)).astype(bool)
                healthy_subjects_indexes = np.logical_not(np.max(masks, axis=(2,3,4,5)).astype(bool))
                healthy_scores = anomaly_scores[healthy_subjects_indexes]
                anomalous_scores = anomaly_scores[anomalous_subjects_indexes]
                if (len(healthy_scores) !=0) and (len(healthy_scores) !=0):
                    compute_auc(healthy_scores, anomalous_scores, format_wise= "imagewise", agg_function=np.mean, data = "validation",deformation = deformation)
                    compute_average_precision(healthy_scores, anomalous_scores, format_wise= "imagewise", agg_function=np.mean, data = "validation")
                else:
                    logging.info('Image Wise: Same class for all - cannot compute')

                ## 2D slice wise
                anomalous_subjects_indexes = np.max(masks, axis=(3,4)).astype(bool)
                healthy_subjects_indexes = np.logical_not(np.max(masks, axis=(3,4)).astype(bool))
                healthy_scores = anomaly_scores.reshape(-1,32,32)[healthy_subjects_indexes.flatten()]
                anomalous_scores =anomaly_scores.reshape(-1,32,32)[anomalous_subjects_indexes.flatten()]
                if (len(healthy_scores) !=0) and (len(healthy_scores) !=0):
                    compute_auc(healthy_scores, anomalous_scores, format_wise= "2Dslice", agg_function=np.mean, data = "validation",deformation = deformation)
                    compute_average_precision(healthy_scores, anomalous_scores, format_wise= "2Dslice", agg_function=np.mean, data = "validation")
                else:
                    logging.info('2DSlice Wise: Same class for all - cannot compute')

def plot_slices(images_dict, most_separable_patients_z_slices, least_separable_patients_z_slices, save_dir_test_images, time_steps = 1):
    # Define which images to use for each case
    most_separable_images = ['image_highest_anomaly_score_anomalous_subject', 'image_lowest_anomaly_score_healthy_subject']
    least_separable_images = ['image_lowest_anomaly_score_anomalous_subject', 'image_highest_anomaly_score_healthy_subject']
    
    for case, image_names in zip(['Most separable', 'Least separable'], [most_separable_images, least_separable_images]):
        for image_name in image_names:
            image = images_dict[image_name]
            subject_index = images_dict[image_name.replace("image_", "")]
            output = images_dict[image_name.replace("image_", "output_")]
            
            # get the corresponding z-slice index
            # get the corresponding z-slice index
            if case == 'Most separable':
                if 'anomalous' in image_name:
                    max_index = most_separable_patients_z_slices['sick_max_index']
                    min_index = most_separable_patients_z_slices['sick_min_index']
                else:  # healthy
                    max_index = most_separable_patients_z_slices['healthy_max_index']
                    min_index = most_separable_patients_z_slices['healthy_min_index']
            else:  # case == 'Least separable'
                if 'anomalous' in image_name:
                    max_index = least_separable_patients_z_slices['sick_max_index']
                    min_index = least_separable_patients_z_slices['sick_min_index']
                else:  # healthy
                    max_index = least_separable_patients_z_slices['healthy_max_index']
                    min_index = least_separable_patients_z_slices['healthy_min_index']

            if time_steps == 1:
                # Then we plot only the max and min image with their respective output for min_time_steo
                max_time_step = np.argmax(np.nanmax(image[max_index-1][:, :, :, 1], axis=(0, 1)))
                min_time_step = np.argmin(np.nanmin(image[min_index-1][:, :, :, 1], axis=(0, 1)))
                
                time_steps_to_plot = [max_time_step, min_time_step]
                fig, axs = plt.subplots(2, 2, figsize=(8,4))
                # extract the max and min slices (both at max time step)
                max_image = image[max_index-1][:, :, max_time_step, 1]
                min_image = image[min_index-1][:, :, max_time_step, 1]
                # Only one channel in the output for SSL methods
                max_output = output[max_index-1][0, :, :, max_time_step]
                min_output = output[min_index-1][0, :, :, max_time_step]
                im1 = axs[0, 0].imshow(max_image, cmap='gray')
                axs[0, 0].set_title(f'max image - slice {max_index}, time_step {max_time_step}')
                fig.colorbar(im1, ax=axs[0, 0])
                im2 = axs[1, 0].imshow(max_output, cmap='gray')
                axs[1, 0].set_title(f'max output - slice {max_index}, time_step {max_time_step}')
                fig.colorbar(im2, ax=axs[1, 0])
                im3 = axs[0, 1].imshow(min_image, cmap='gray')
                axs[0, 1].set_title(f'min image - slice {min_index}, time_step {max_time_step}')
                fig.colorbar(im3, ax=axs[0, 1])
                im4 = axs[1, 1].imshow(min_output, cmap='gray')
                axs[1, 1].set_title(f'min output - slice {min_index}, time_step {max_time_step}')
                fig.colorbar(im4, ax=axs[1, 1])

                for ax in axs.flatten():
                    ax.axis('off')
                fig.suptitle(f'{case} - {image_name} - subject index: {subject_index}')
                fig.tight_layout()
                save_path = os.path.join(save_dir_test_images, f'{case}_{image_name}_subject_index_{subject_index}_max_time_step_{max_time_step}.png')
                plt.savefig(save_path)
                plt.close()
                # Let us save the 4 images as well in a new directory plus their output
                save_dir_test_images_4_images = os.path.join(save_dir_test_images, 'most_least_separable_4_images')
                make_dir_safely(save_dir_test_images_4_images)
                np.save(os.path.join(save_dir_test_images_4_images, f'{case}_{image_name}_subject_index_{subject_index}_max_time_step_{max_time_step}_max_image.npy'), max_image)
                np.save(os.path.join(save_dir_test_images_4_images, f'{case}_{image_name}_subject_index_{subject_index}_max_time_step_{max_time_step}_max_output.npy'), max_output)
                np.save(os.path.join(save_dir_test_images_4_images, f'{case}_{image_name}_subject_index_{subject_index}_max_time_step_{max_time_step}_min_image.npy'), min_image)
                np.save(os.path.join(save_dir_test_images_4_images, f'{case}_{image_name}_subject_index_{subject_index}_max_time_step_{max_time_step}_min_output.npy'), min_output)
                # Plot them one by one in different figures with colorbar + save
                # Define the function to create and save the image
                def create_and_save_image(image, cmap, save_dir, filename):
                    fig, ax = plt.subplots()
                    im = ax.imshow(image, cmap=cmap)
                    fig.colorbar(im)
                    ax.axis('off')  # to remove the axis
                    save_path = os.path.join(save_dir, filename)
                    plt.savefig(save_path, bbox_inches='tight', pad_inches = 0)
                    plt.close(fig)

                # Create and save the images
                create_and_save_image(max_image, 'gray', save_dir_test_images_4_images, f'{case}_{image_name}_subject_index_{subject_index}_max_time_step_{max_time_step}_max_image.png')
                create_and_save_image(max_output, 'gray', save_dir_test_images_4_images, f'{case}_{image_name}_subject_index_{subject_index}_max_time_step_{max_time_step}_max_output.png')
                create_and_save_image(min_image, 'gray', save_dir_test_images_4_images, f'{case}_{image_name}_subject_index_{subject_index}_max_time_step_{max_time_step}_min_image.png')
                create_and_save_image(min_output, 'gray', save_dir_test_images_4_images, f'{case}_{image_name}_subject_index_{subject_index}_max_time_step_{max_time_step}_min_output.png')

                

            else:
                time_steps_to_plot = range(time_steps)
                fig, axs = plt.subplots(4, len(time_steps_to_plot), figsize=(20, 4))
        
                for i, time_step in enumerate(time_steps_to_plot):
                    # extract the max and min slices
                    max_image = image[max_index-1][:, :, time_step, 1]
                    min_image = image[min_index-1][:, :, time_step, 1]
                    # Only one channel in the output for SSL methods
                    max_output = output[max_index-1][0, :, :, time_step]
                    min_output = output[min_index-1][0, :, :, time_step]

                    # plot the max and min slices
                    im1 = axs[0, i].imshow(max_image, cmap='gray')
                    fig.colorbar(im1, ax=axs[0, i])
                    im2 = axs[1, i].imshow(max_output, cmap='gray')
                    fig.colorbar(im2, ax=axs[1, i])
                    im3 = axs[2, i].imshow(min_image, cmap='gray')
                    fig.colorbar(im3, ax=axs[2, i])
                    im4 = axs[3, i].imshow(min_output, cmap='gray')
                    fig.colorbar(im4, ax=axs[3, i])
                    for ax in axs[:, i]:
                        ax.axis('off')  # to hide x, y axis

                    if i == 0:
                        axs[0, i].set_title(f'max image - slice {max_index}, time_step {time_step}')
                        axs[1, i].set_title(f'max output - slice {max_index}, time_step {time_step}')
                        axs[2, i].set_title(f'min image - slice {min_index}, time_step {time_step}')
                        axs[3, i].set_title(f'min output - slice {min_index}, time_step {time_step}')


                # Set the overall title for the figure
                fig.suptitle(f'{case} - {image_name}: {subject_index}')
                plt.tight_layout()
                # Save figure
                save_path = os.path.join(save_dir_test_images, f'{case}_{image_name}_subject_index_{subject_index}_time_steps_{time_steps}.png')
                plt.savefig(save_path)
                plt.close()

def return_indexes_to_remove(model_name, deformation_list, images, with_noise = False):
            
            n_images = images.shape[0]
            indexes = set(np.arange(n_images))
            if with_noise:
                if 'with_interpolation_training' in model_name:
                    deformation_list.remove('patch_interpolation')
                    indices_to_remove = [896, 1344]
                elif 'poisson_mix_training' in model_name:
                    deformation_list.remove('poisson_with_mixing')
                    indices_to_remove = [1344, 1792]
                else:
                    deformation_list = ['None','deformation', 'patch_interpolation']
                    indices_to_remove = [1344, 2240]
            else:
                if 'with_interpolation_training' in model_name:
                    deformation_list.remove('patch_interpolation')
                    indices_to_remove = [1792, 2240]
                elif 'poisson_mix_training' in model_name:
                    deformation_list.remove('poisson_with_mixing')
                    indices_to_remove = [2240, 2688]
                else:
                    deformation_list = ['None', 'noisy', 'deformation', 'hollow_circle', 'patch_interpolation']
                    indices_to_remove = [2240, 3136]
            
            
            indices_to_remove = set(np.arange(indices_to_remove[0], indices_to_remove[1]))
            diff = indexes - indices_to_remove
            indexes = np.array(list(diff))
            return indexes, deformation_list

def statistical_tests(healthy_scores, anomalous_scores):
    
    # Conduct Shapiro-Wilk Test for Normality
    _, p_sick = stats.shapiro(anomalous_scores)
    _, p_healthy = stats.shapiro(healthy_scores)

    logging.info(f"Case scores normality test p-value: {p_sick}")
    logging.info(f"Control scores normality test p-value: {p_healthy}")
    logging.info('If P-value > 0.05, the data is normally distributed\n')

    # If p-value > 0.05 for both, the data is normally distributed

    # Conduct Levene's Test for Equality of Variances
    _, p_levene = stats.levene(anomalous_scores, healthy_scores)
    logging.info(f"Equality of variances test p-value: {p_levene}")
    logging.info('If P-value > 0.05, variances are equal\n')

    # If p-value > 0.05, variances are equal

    # Depending on the above results, conduct appropriate t-test
    if p_sick > 0.05 and p_healthy > 0.05 and p_levene > 0.05:
        logging.info('Normally distributed and equal variances - Conduct Student\'s t-test')
        # Conduct Student's t-test
        _, p_ttest = stats.ttest_ind(anomalous_scores, healthy_scores)
        logging.info(f"Student's t-test p-value: {p_ttest}")
    elif p_sick <= 0.05 or p_healthy <= 0.05 or p_levene <= 0.05:
        logging.info('Not normally distributed or unequal variances - Conduct Mann-Whitney U Test')
        # Conduct Mann-Whitney U Test
        _, p_mannwhitney = stats.mannwhitneyu(anomalous_scores, healthy_scores)
        logging.info(f"Mann-Whitney U test p-value: {p_mannwhitney}\n")
        # Or Conduct Welch's t-test
        logging.info('Conduct Welch\'s t-test')
        _, p_welch = stats.ttest_ind(anomalous_scores, healthy_scores, equal_var=False)
        logging.info(f"Welch's t-test p-value: {p_welch}\n")
        # Not valid but still conduct T-test
        logging.info('Conduct Student\'s t-test - Careful Not Valid')
        _, p_ttest = stats.ttest_ind(anomalous_scores, healthy_scores)
        logging.info(f"Student's t-test p-value: {p_ttest}")
    logging.info('If p < 0.05, the difference is statistically significant at the 5% level')

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
    


def plot_imagewise_level(sick_means, sick_stds, healthy_means, healthy_stds, save_dir, data, note):
    # Prepare the data for seaborn
    sick_df = pd.DataFrame({
        'Subject Status': np.repeat('Case', sick_means.size),
        'Subject': np.repeat(np.arange(sick_means.shape[0]), sick_means.shape[1]),
        'imagewise': np.tile(np.arange(1, sick_means.shape[1] + 1), sick_means.shape[0]),
        'Mean Score': sick_means.flatten(),
        'Std Deviation': sick_stds.flatten()
    })

    healthy_df = pd.DataFrame({
        'Subject Status': np.repeat('Control', healthy_means.size),
        'Subject': np.repeat(np.arange(healthy_means.shape[0]), healthy_means.shape[1]),
        'imagewise': np.tile(np.arange(1, healthy_means.shape[1] + 1), healthy_means.shape[0]),
        'Mean Score': healthy_means.flatten(),
        'Std Deviation': healthy_stds.flatten()
    })

    # Concatenate the two dataframes
    df = pd.concat([sick_df, healthy_df])
    
        

    # Create the line plot
    plt.figure(figsize=(12, 6))
    line_plot = sns.lineplot(data=df, x='imagewise', y='Mean Score', hue='Subject Status', style="Subject")
    line_plot.legend_.remove()  # Remove the original legend
    plt.xlabel('Z Slice')
    # Create a new legend only for Subject Status (Color)
    handles, labels = line_plot.get_legend_handles_labels()
    line_plot.legend(handles=handles[:3], labels=labels[:3], loc='upper left')

    

    #plt.title('Mean Anomaly Score for each subject group')
    plt.ylabel('Mean Anomaly Score')
    if note:        
        plt.savefig(os.path.join(save_dir+ '/' + f'{data}_mean_imagewise_scores_{note}.png'))
        # Set y axis to log scale
        plt.yscale('log')
        plt.savefig(os.path.join(save_dir+ '/' + f'{data}_mean_imagewise_scores_{note}_log.png'))
    else:
        plt.savefig(os.path.join(save_dir+ '/' + f'{data}_mean_imagewise_scores_by_patient.png'))
        # Set y axis to log scale
        plt.yscale('log')
        plt.savefig(os.path.join(save_dir+ '/' + f'{data}_mean_imagewise_scores_by_patient_log.png'))


    plt.close()
    # Create the line plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='imagewise', y='Mean Score', hue='Subject Status', err_style="bars", errorbar='sd')
    plt.xlabel('Z Slice')
    plt.legend(title='Subject Status')
    

    #plt.title('Mean Anomaly Score for each subject group')
    plt.ylabel('Mean Anomaly Score')
    if note:
        plt.savefig(os.path.join(save_dir+ '/' + f'{data}_mean_imagewise_scores_{note}.png'))
        # Set y axis to log scale
        plt.yscale('log')
        plt.savefig(os.path.join(save_dir+ '/' + f'{data}_mean_imagewise_scores_{note}_log.png'))

    else:
        plt.savefig(os.path.join(save_dir+ '/' + f'{data}_mean_imagewise_scores.png'))
        # Set y axis to log scale
        plt.yscale('log')
        plt.savefig(os.path.join(save_dir+ '/' + f'{data}_mean_imagewise_scores_log.png'))
    plt.close()
            
    # If note exists we want to return the indexes of the slices with the min nad max score for the healthy and sick patient (there is only one patient in the healthy group and one in the sick group)
    # Save them in dictionary to return
    if note:
        # Get the indexes of the slices with the min and max score for the healthy and sick patient
        sick_min_index = df.loc[df['Subject Status'] == 'Case']['Mean Score'].idxmin()
        sick_max_index = df.loc[df['Subject Status'] == 'Case']['Mean Score'].idxmax()
        healthy_min_index = df.loc[df['Subject Status'] == 'Control']['Mean Score'].idxmin()
        healthy_max_index = df.loc[df['Subject Status'] == 'Control']['Mean Score'].idxmax()
        # Get the imagewise indexes
        sick_min_index = df.loc[df['Subject Status'] == 'Case']['imagewise'][sick_min_index]
        sick_max_index = df.loc[df['Subject Status'] == 'Case']['imagewise'][sick_max_index]
        healthy_min_index = df.loc[df['Subject Status'] == 'Control']['imagewise'][healthy_min_index]
        healthy_max_index = df.loc[df['Subject Status'] == 'Control']['imagewise'][healthy_max_index]
        # Save them in dictionary
        indexes = {'sick_min_index': sick_min_index, 'sick_max_index': sick_max_index, 'healthy_min_index': healthy_min_index, 'healthy_max_index': healthy_max_index}
        return indexes
    plt.close()




def plot_patient_level(sick_means, sick_stds, healthy_means, healthy_stds, save_dir, data, note):
    df_sick = pd.DataFrame({
            'Subject': ['Case '+str(i) for i in range(len(sick_means))],
            'Mean Score': sick_means,
            'Std Deviation': sick_stds,
            'Status': ['Case']*len(sick_means)
        })

    # Create a dataframe for healthy patients
    df_healthy = pd.DataFrame({
        'Subject': ['Control '+str(i) for i in range(len(healthy_means))],
        'Mean Score': healthy_means,
        'Std Deviation': healthy_stds,
        'Status': ['Control']*len(healthy_means)
    })

    # Concatenate the dataframes
    df = pd.concat([df_sick, df_healthy])
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Subject', y='Mean Score', hue='Status', data=df, yerr=df['Std Deviation'], capsize=.2)
    plt.xlabel('')
    plt.xticks(rotation=90)  # This will rotate the x-axis labels to avoid overlap
    

    #plt.title('Mean Anomaly Score with Standard Deviation for each subject')
    plt.ylabel('Mean Anomaly Score')
    plt.tight_layout()
    if note:
        plt.savefig(os.path.join(save_dir+ '/' + f'{data}_patient_mean_wise_scores_{note}.png'))
        # Limit the y axis
        plt.ylim(-0.04, 0.055)
        plt.savefig(os.path.join(save_dir+ '/' + f'{data}_patient_mean_wise_scores_{note}_y_lim.png'))
        plt.close()
    else:
        plt.savefig(os.path.join(save_dir+ '/' + f'{data}_patient_mean_wise_scores.png'))
        # Limit the y axis
        plt.ylim(-0.04, 0.055)
        plt.savefig(os.path.join(save_dir+ '/' + f'{data}_patient_mean_wise_scores_y_lim.png'))    
        plt.close()

