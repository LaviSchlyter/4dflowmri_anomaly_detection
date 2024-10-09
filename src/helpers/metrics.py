"""
Evaluation Metrics for Anomaly Detection

This script contains functions to compute evaluation metrics for anomaly detection in medical images.
The implemented metrics include Root Mean Squared Error (RMSE), Area Under the Receiver Operating
Characteristic Curve (AUC-ROC), and Average Precision (AP) scores.

Functions:
- RMSE: Computes the root mean squared error between original and predicted tensors.
- compute_auc_roc_score: Computes the AUC-ROC score for pixel-wise, image-wise, or 2D slice-wise comparisons.
- compute_average_precision_score: Computes the average precision score for pixel-wise, image-wise, or 2D slice-wise comparisons.

pixel-wise: compares each pixel in the image
image-wise: compares the mean of the image (x,y,t,z)

Following experiments, for the paper we use the following validation_metric_format:
2D slice-wise: compares the mean of the 2D slice (x,y,t) 
"""
import torch 
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def RMSE(x, y, dim=(0, 1, 2, 3, 4)):
    """
    Computes the root mean squared error (RMSE) between an original and predicted tensor.

    Parameters:
    - x (torch.Tensor): The original tensor.
    - y (torch.Tensor): The predicted tensor.
    - dim (tuple): The dimensions over which to compute the RMSE.

    Returns:
    - torch.Tensor: The RMSE value.
    """
    result = torch.sqrt(torch.mean((x - y) ** 2, dim=dim))
    return result




# Compute AUC-ROC score
def compute_auc_roc_score(predicted_errors, val_masks, config):
    """
    Computes the AUC-ROC score for anomaly detection.

    Parameters:
    - predicted_errors (list of numpy.ndarray): List with potential anomaly scores.
    - val_masks (list of torch.Tensor or numpy.ndarray): List of validation masks.
    - config (dict): Configuration dictionary containing 'self_supervised' and 'validation_metric_format'.

    Returns:
    - float: The AUC-ROC score.
    """
    
    predicted_errors = np.concatenate(predicted_errors, axis=0)

    if torch.is_tensor(val_masks[0]):
        val_masks = torch.concatenate(val_masks, axis = 0).cpu().numpy()
    else:
        val_masks = np.concatenate(val_masks, axis=0)

    # If self-supervised, the masks channel dimesion is 1 and not 4 (it was duplicated)
    if config['self_supervised']:
        val_masks = val_masks[:, 0:1, :, :, :]

    if predicted_errors.shape != val_masks.shape:
        raise ValueError(f"predicted_errors and val_masks have different shapes: {predicted_errors.shape} and {val_masks.shape}")
    
    if config['validation_metric_format'] == 'pixelwise':
        auc_roc = roc_auc_score(val_masks.flatten(), predicted_errors.flatten())
    elif config['validation_metric_format'] == 'imagewise':
        predicted_errors_slice_mean = np.mean(predicted_errors, axis=(1,2,3,4))
        val_masks_slice_max = np.max(val_masks, axis=(1,2,3,4))
        auc_roc = roc_auc_score(val_masks_slice_max.flatten(), predicted_errors_slice_mean.flatten())
    elif config['validation_metric_format'] == '2Dslice':
        predicted_errors_2D_slice_mean = np.mean(predicted_errors, axis=(1,2,3))
        val_masks_2D_slice_max = np.max(val_masks, axis=(1,2,3))
        auc_roc = roc_auc_score(val_masks_2D_slice_max.flatten(), predicted_errors_2D_slice_mean.flatten())
    else:
        raise ValueError('validation_metric_format not recognized')
        
    return auc_roc


def compute_average_precision_score(predicted_errors, val_masks, config):
    """
    Computes the average precision (AP) score for anomaly detection.

    Parameters:
    - predicted_errors (list of numpy.ndarray): List with potential anomaly scores.
    - val_masks (list of torch.Tensor or numpy.ndarray): List of validation masks.
    - config (dict): Configuration dictionary containing 'self_supervised' and 'validation_metric_format'.

    Returns:
    - float: The average precision score.
    """

    predicted_errors = np.concatenate(predicted_errors, axis=0)

    if torch.is_tensor(val_masks[0]):
        val_masks = torch.concatenate(val_masks, axis = 0).cpu().numpy()
    else:
        val_masks = np.concatenate(val_masks, axis=0)

    if config['self_supervised']:
        val_masks = val_masks[:, 0:1, :, :, :]

    if config['validation_metric_format'] == 'pixelwise':
        ap = average_precision_score(val_masks.flatten(), predicted_errors.flatten())
    elif config['validation_metric_format'] == 'imagewise':
        predicted_errors_slice_mean = np.mean(predicted_errors, axis=(1,2,3,4))
        val_masks_slice_max = np.max(val_masks, axis=(1,2,3,4))
        ap = average_precision_score(val_masks_slice_max.flatten(), predicted_errors_slice_mean.flatten())
    elif config['validation_metric_format'] == '2Dslice':
        predicted_errors_2D_slice_mean = np.mean(predicted_errors, axis=(1,2,3))
        val_masks_2D_slice_max = np.max(val_masks, axis=(1,2,3))
        ap = average_precision_score(val_masks_2D_slice_max.flatten(), predicted_errors_2D_slice_mean.flatten())
    else:
        raise ValueError('validation_metric_format not recognized')

    return ap