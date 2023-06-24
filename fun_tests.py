# %%
import torch
from torch.autograd import Variable


import random
import numpy as np
from matplotlib import pyplot as plt
import math, os

from models.vae import VAE, VAE_convT, VAE_linear

import h5py
from helpers.synthetic_anomalies import create_cube_mask
import sys
# For the patch blending we import from another directory
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/git_repos/many-tasks-make-light-work')
from multitask_method.tasks.patch_blending_task import TestPatchInterpolationBlender, \
    TestPoissonImageEditingMixedGradBlender, TestPoissonImageEditingSourceGradBlender, TestCutPastePatchBlender

from multitask_method.tasks.labelling import FlippedGaussianLabeller
from multitask_method.tasks.cutout_task import Cutout
labeller = FlippedGaussianLabeller(0.2)



data = h5py.File('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/data/train_masked_sliced_images_from_0_to_35.hdf5', 'r')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
images = data['sliced_images_train']
test_image = images[140]
# transpose to channels first
test_image = np.transpose(test_image, (3, 0, 1, 2))
print(test_image.shape)
#plt.imshow(test_image[:,:,3,1])
# Apply deformation on image

random_id = 1418#random.randrange(images.shape[0])
print(random_id)
image_to_blend = images[random_id]
image_to_blend = np.transpose(image_to_blend, (3, 0, 1, 2))

# Create anomaly mask
mask_blending = create_cube_mask((32,32,24), WH= 20, depth= 12,  inside=True).astype(np.bool_)
cutout_task = Cutout(labeller)
cutpaste_task = TestCutPastePatchBlender(labeller, image_to_blend, mask_blending)
patch_interp_task = TestPatchInterpolationBlender(labeller, image_to_blend, mask_blending)
poisson_image_editing_mixed_task = TestPoissonImageEditingMixedGradBlender(labeller, image_to_blend, mask_blending)
poisson_image_editing_source_task = TestPoissonImageEditingSourceGradBlender(labeller, image_to_blend, mask_blending)
anom_image, anom_mask = poisson_image_editing_mixed_task(test_image, mask_blending)
print(np.unique(np.where(anom_image != test_image)[-1]))
# %% 
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
t = 8

ax[0].imshow(test_image[1,:,:,t])
ax[1].imshow(anom_image[1,:,:,t])
ax[2].imshow(anom_mask[:,:,t])
plt.show()
#%%
# Repeat the image on the batch dimension
test_image = np.repeat(test_image[np.newaxis,...], 64, axis=0)
anom_mask = np.repeat(anom_mask[np.newaxis,...], 64, axis=0)
input_image = torch.from_numpy(test_image).float().to(device)
label = torch.from_numpy(anom_mask).float().unsqueeze(1).to(device)
input_image.shape, label.shape  


criterion = torch.nn.BCEWithLogitsLoss()

#%%
# Create training loop
# Load the model

#model = VAE(in_channels=4, gf_dim=8, out_channels=1).to(device)
model = VAE_convT(in_channels=4, gf_dim=16, out_channels=1, apply_initialization= False).to(device)
#model = VAE_linear(in_channels=4, gf_dim=8, out_channels=1, z_dim = 2048, interpolate=False).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



model.train()
losses = []
epochs = []
for epoch in range(300):
    #print(f'Epoch {epoch}')

    optimizer.zero_grad()
    output_dict = model(input_image)
    loss = criterion(output_dict['decoder_output'], label)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        losses.append(loss.item())
        epochs.append(epoch + 1)
        print(f'Epoch {(epoch + 1)}, loss: {loss.item()}')
        fig, ax = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(5):
            t_ = 8
            mask_1 = ax[0,i].imshow(torch.sigmoid(output_dict['decoder_output'])[0,0,:,:,i+t_].cpu().detach().numpy())
            fig.colorbar(mask_1, ax=ax[0,i])
            im = ax[1,i].imshow(anom_mask[0,:,:,i+t_])
            fig.colorbar(im, ax=ax[1,i])
        # Add colorbar
        plt.show()

plt.plot(np.array(epochs), np.array(losses))
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()

    

# %%
3072