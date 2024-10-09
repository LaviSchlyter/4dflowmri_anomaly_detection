#%%
import torch
import numpy as np 
from scipy.special import eval_genlaguerre as L

def l2loss(x, y):
    """
    Computes l2 loss of input x and output y.
    :param x: input
    :param y: output of the network
    :result: L2 Loss
    """
    l2_loss = torch.mean((x - y)**2, dim=(1, 2, 3, 4))
    return l2_loss


def kl_loss_1d(z_mean, z_stddev):
    """
    Computes the Kullback Leibler divergence for flattened 1D latent space
    """
    latent_loss = torch.mean(torch.square(z_mean) + torch.square(z_stddev) - 2. * torch.log(torch.abs(z_stddev + 1e-7)) - 1, dim=(1, 2, 3, 4))
    return 0.5 * latent_loss

