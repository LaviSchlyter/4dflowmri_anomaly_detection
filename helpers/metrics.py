import torch 
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def RMSE(x,y, dim=(0,1,2,3,4)):
    """
    Returns the root mean squared error between an original and predicted tensor
    :parameter: x - original
    :parameter: y - prediction
    """    
    result = torch.sqrt(torch.mean((x-y)**2, dim=dim))
    return result




# Compute AUC-ROC score
def compute_auc_roc_score(images_error, val_masks, config):
    
    images_error = np.concatenate(images_error, axis=0)
    if torch.is_tensor(val_masks[0]):
        val_masks = torch.concatenate(val_masks, axis = 0).cpu().numpy()
    else:
        val_masks = np.concatenate(val_masks, axis=0)
    # If self-supervised, the masks channel dimesion is 1 and not 4 (it's the same )
    if config['self_supervised']:
        val_masks = val_masks[:, 0:1, :, :, :]
    # Check that images_error and val_masks have the same shape
    if images_error.shape != val_masks.shape:
        raise ValueError(f"images_error and val_masks have different shapes: {images_error.shape} and {val_masks.shape}")
    
    if config['validation_metric_format'] == 'pixelwise':
        # Pixel level
        auc_roc = roc_auc_score(val_masks.flatten(), images_error.flatten())
    elif config['validation_metric_format'] == 'imagewise':
        # Image level
        images_error_slice_mean = np.mean(images_error, axis=(1,2,3,4))
        val_masks_slice_max = np.max(val_masks, axis=(1,2,3,4)) # Tells you if one is in there
        auc_roc = roc_auc_score(val_masks_slice_max.flatten(), images_error_slice_mean.flatten())
    elif config['validation_metric_format'] == '2Dslice':
        # 2D slice level
        images_error_2D_slice_mean = np.mean(images_error, axis=(1,2,3))
        val_masks_2D_slice_max = np.max(val_masks, axis=(1,2,3))
        auc_roc = roc_auc_score(val_masks_2D_slice_max.flatten(), images_error_2D_slice_mean.flatten())
    else:
        raise ValueError('validation_metric_format not recognized')
    return auc_roc

# Compute average precision score
def compute_average_precision_score(images_error, val_masks, config):

    images_error = np.concatenate(images_error, axis=0)
    if torch.is_tensor(val_masks[0]):
        val_masks = torch.concatenate(val_masks, axis = 0).cpu().numpy()
    else:
        val_masks = np.concatenate(val_masks, axis=0)
    # If self-supervised, the masks channel dimesion is 1 and not 4 (it's the same )
    if config['self_supervised']:
        val_masks = val_masks[:, 0:1, :, :, :]

    if config['validation_metric_format'] == 'pixelwise':
        # Pixel level
        ap = average_precision_score(val_masks.flatten(), images_error.flatten())
    elif config['validation_metric_format'] == 'imagewise':
        # Image level
        images_error_slice_mean = np.mean(images_error, axis=(1,2,3,4))
        val_masks_slice_max = np.max(val_masks, axis=(1,2,3,4))
        ap = average_precision_score(val_masks_slice_max.flatten(), images_error_slice_mean.flatten())
    elif config['validation_metric_format'] == '2Dslice':
        # 2D slice level
        images_error_2D_slice_mean = np.mean(images_error, axis=(1,2,3))
        val_masks_2D_slice_max = np.max(val_masks, axis=(1,2,3))
        ap = average_precision_score(val_masks_2D_slice_max.flatten(), images_error_2D_slice_mean.flatten())
    else:
        raise ValueError('validation_metric_format not recognized')
    return ap