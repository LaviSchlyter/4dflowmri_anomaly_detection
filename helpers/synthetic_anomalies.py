# We want to create a dataset with synthetic anomalies
# %%
# Import packages
import os
import sys
import h5py
import numpy as np

sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection')


from config import system as sys_config
from helpers.utils import make_dir_safely
from data_loader import load_data


from scipy.interpolate import RegularGridInterpolator
import copy

from scipy.ndimage import binary_fill_holes
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import cdist

#%%
def generate_noise(batch_size, mean_range = (0, 0.1), std_range = (0, 0.2), n_samples = 30):
    """
    Generates noise to be added to the images
    :param batch_size: number of images in the batch
    :param mean_range: range of means for the normal distribution
    :param std_range: range of standard deviations for the normal distribution
    :return: mean, std, noise
    """
    means = np.random.uniform(mean_range[0], mean_range[1], n_samples)
    stds = np.random.uniform(std_range[0], std_range[1], n_samples)
    #noise_parameters = [(0.0, 0.02), (0.05, 0.05),(0.1, 0.1),(0.1, 0.2),(0.1, 0.3), (0.2, 0.1), (0.2, 0.2) ,(0.2, 0.4), (0.3, 0.1), (0.3, 0.2), (0.3, 0.3)] # additive noise to be tested, each tuple has mean and stdev for the normal distribution

    for (mean, std) in zip(means, stds):

        # Add some random noise to the image
        part_noise = np.random.normal(mean, std, (batch_size, 5, 5, 24, 4))
        full_noise = np.zeros((batch_size, 32, 32, 24, 4))
        full_noise[:, 14:19, 14:19,:, :] = part_noise

        yield mean, std, full_noise

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

def create_deformation_chan(im,center,width,polarity=1):
    # Create the deformation and apply it to all channels
    dims = np.array(np.shape(im))
    mask = np.zeros_like(im)
    
    center = np.array(center)
    wv,xv,yv,zv = np.arange(dims[0]),np.arange(dims[1]),np.arange(dims[2]),np.arange(dims[3])
    interp_samp = RegularGridInterpolator((wv, xv, yv, zv), im)
    
    for i in range(dims[1]):
        for j in range(dims[2]):
            for k in range(dims[3]):
                dist_i = calc_distance([i,j,k],center)
                displacement_i = (dist_i/width)**2
                
                if displacement_i < 1.:
                    #within width
                    if polarity > 0:
                        #push outward
                        diff_i = np.array([i,j,k])-center
                        new_coor =  center + diff_i*displacement_i
                        
                    else:
                        #pull inward
                        cur_coor = np.array([i,j,k])
                        diff_i = cur_coor-center
                        new_coor = cur_coor + diff_i*(1-displacement_i)
                        
                    new_coor = np.clip(new_coor,np.zeros(len(dims[1:])),dims[1:]-1)
                    new_coor_chan = np.tile(np.arange(dims[0]),(len(dims),1)).T
                    new_coor_chan[:,1:] = new_coor
                    mask[:,i,j,k]= interp_samp(new_coor_chan)
                else:
                    mask[:,i,j,k] = im[:,i,j,k]
    return mask

def generate_deformation_chan(im):
    # Copy to avoid changing the original image
    im2 = copy.deepcopy(im)
    masks = np.zeros_like(im)

    for batch_i in range(im.shape[0]):
        im_in = im2[batch_i,:,:,:,:]
        dims = np.array(np.shape(im_in))

        # For the third dimesnion (time) we want the anomalies to be in the start
        core = dims[:3]/[4,4,4] #width of core region
        offset = (dims[:3]-core)/[2,2,18]#offset to center core
        
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

        # Put channels first for the deformation function
        im_in = np.transpose(im_in, (3, 0, 1, 2))
        out_image = create_deformation_chan(im_in,sphere_center,sphere_width,sphere_polarity)
        #[c, w, h, t]
        # Put channels last for output
        out_image = np.transpose(out_image, (1, 2, 3, 0))
        im2[batch_i,:,:,:,:] = out_image
        masks[batch_i,:,:,:,:] = mask_i

    return im2, masks

# ==========================================
# ==========================================
# Hollow mask
# ==========================================
# ==========================================


# %%


def find_closest_distance_edge(mask, center_of_mass):
    # Find the closest distance from the center of mass to the edge of the mask (first zero)
    # Find the indices of zero values in the mask
    zero_indices = np.argwhere(mask == 0)

    # Calculate the Euclidean distances between the zero indices and the center of mass
    distances = cdist(zero_indices, [center_of_mass])

    # Find the minimum distance
    closest_distance = np.min(distances)

    return closest_distance

# Fill in the circle with random noise
def fill_circle(image, center, radius, mean, std):
    # The mean and std are the same as randomly generated for noisy image

    # Generate coordinate grids
    x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))

    # Calculate distance from center for each pixel
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # Create a mask based on which pixels are within the radius
    mask = distances <= radius
    
    # Fill in the circle with random noise
    noise = np.random.normal(mean, std, size=image.shape)
    return mask, noise

def create_hollow_noise(im, mean, std, ratio = 0.8):
    # We only create the mask in the x,y plane, we then propagate on time (later)

    # Copy to avoid changing the original image
    image_copy = copy.deepcopy(im)
    # Take x,y dimensions
    #image = image[:,:, 0,0]
    image = image_copy[:,:,0,0]

    # Binarize the image
    image[image != 0] = 1

    # Fill holes in the image
    image = binary_fill_holes(image).astype(int)

    # Find center of mass
    center = center_of_mass(image)

    # They return in (y,x) format
    center = [center[1], center[0]]

    # Round to integer
    center = np.round(center).astype(int)

    # Find the closest distance from the center of mass to the edge of the mask (first zero)
    radius = find_closest_distance_edge(image, center)

    # Generate random noise
    mask, noise = fill_circle(image, center, radius, mean, std)

    # Create a smaller circle with same noise
    radius_2 = radius * ratio

    mask_2, _ = fill_circle(image, center, radius_2, mean, std)

    # Subtract the two masks to create the hollow mask with noise
    final_noise = noise*mask - noise*mask_2

    # We need to extend the noise to the time dimension and channel dimensions
    # Note that we only apply the hollow noise on timsteps [2:7]
    # That's when the blood pumps and we can see potential anomalies 

    # Reshape the image to (x,y, 1, 1)
    final_noise_resaped = final_noise.reshape(final_noise.shape[0], final_noise.shape[1], 1, 1)

    # Create an array of zeros of size (x,y, 24, 4)
    noisy_mask = np.zeros((final_noise.shape[0], final_noise.shape[1], 24, 4), dtype=final_noise.dtype)    

    # Assign the noisy_mask image to the first 5 dimensions of 24
    noisy_mask[:, :, 2:7, :] = final_noise_resaped

    # Divide the magnitude noise by 10
    noisy_mask = (noisy_mask[:,:,:,:]/[10,1,1,1])

    return noisy_mask



def prepare_and_write_synthetic_data(data,
                                    deformation_list,
                                    filepath_output):
    
    data_shape = data.shape # [Number of patients * z_slices, x, y, t, num_channels]

    # ==========================================
    # we will stack all images along their z-axis
    # --> the network will analyze (x,y,t) volumes
    # ==========================================

    # For each patient, for each slice we create a new image with the anomaly
    # The dataset also contains the healthy images (None as deformation type)
    images_dataset_shape = [data_shape[0] * len(deformation_list),
                            data_shape[1],
                            data_shape[2],
                            data_shape[3],
                            data_shape[4]]
    
    dataset = {}
    hdf5_file = h5py.File(filepath_output, 'w')

    # Create the dataset
    dataset['images'] = hdf5_file.create_dataset("images", images_dataset_shape, dtype='float32')

    # Create the mask 
    dataset['masks'] = hdf5_file.create_dataset("masks", images_dataset_shape, dtype='float32')

    # Noise generator
    noise_generator = generate_noise(batch_size = 1, n_samples= data_shape[0] * len(deformation_list))
    
    # For each type of deformation, we create a new image for each patient and each slice
    for i, deformation_type in enumerate(deformation_list):
        if deformation_type == 'None':
            # We do not create a new image, we just copy the original image
            dataset['images'][i*data_shape[0]:(i+1)*data_shape[0],:,:,:,:] = data
            dataset['masks'][i*data_shape[0]:(i+1)*data_shape[0],:,:,:,:] = np.zeros_like(data)
        else:
            for j, image in enumerate(data):
                # Add batch dimension to 1
                image = np.expand_dims(image, axis=0)
                if deformation_type == 'noisy':
                    # We add noise to the image
                    # We scale the noise of the channel down by a factor of 10 (not as high intensity as the other velocity channels)
                    mean, std, noise = next(noise_generator)
                    noise = noise/[10,1,1,1]
                    dataset['images'][i*data_shape[0] + j,:,:,:,:] = image + noise
                    dataset['masks'][i*data_shape[0] + j,:,:,:,:] = noise
                    
                elif deformation_type == 'deformation':
                    # We create a deformation in the image
                    deformation, mask = generate_deformation_chan(image)
                    dataset['images'][i*data_shape[0] + j,:,:,:,:] = deformation
                    dataset['masks'][i*data_shape[0] + j,:,:,:,:] = mask
                    
                elif deformation_type == 'hollow circle':
                    noisy_mask = create_hollow_noise(image, mean=mean, std=std)
                    dataset['images'][i*data_shape[0] + j,:,:,:,:] = image + noisy_mask
                    dataset['masks'][i*data_shape[0] + j,:,:,:,:] = noisy_mask
                elif deformation_type == 'patch':
                    pass
                else:
                    raise ValueError('deformation_type must be either None, noisy, deformation, hollow circle or patch and not {}'.format(deformation_type))

    hdf5_file.close()

    return 0            



def load_synthetic_data(data,
                        deformation_list,
                        idx_start,
                        idx_end,
                        anomaly_type = 'all',
                        force_overwrite=False,
                        ):
    savepath= sys_config.project_code_root + "data"
    make_dir_safely(savepath)
    dataset_filepath = savepath + f'/{anomaly_type}_anomalies_images_from_' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'
    
    if not os.path.exists(dataset_filepath) or force_overwrite:
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_synthetic_data(
                                data = data,
                                deformation_list = deformation_list,
                                filepath_output = dataset_filepath
                                )
    else:
        print('Already preprocessed this configuration. Loading now...')
    
    return h5py.File(dataset_filepath, 'r')

#%%

if __name__ == '__main__':
    #%%
    # Type of deformation
    # 'None', 'noisy', 'deformation', 'hollow circle', 'patch', 'all'
    #deformation_list = ['None', 'noisy', 'deformation', 'hollow circle', 'patch']
    deformation_list = ['None', 'noisy', 'deformation', 'hollow circle']
    deformation_type = 'all'

    # Set config
    config = dict()
    # 'None', 'mask', 'slice', 'masked_slice', 'sliced_full_aorta', 'masked_sliced_full_aorta', 'mock_square'
    config['preprocess_method'] = 'masked_slice' 

    # Load the validation data on which we apply the synthetic anomalies
    _, images_vl, _ = load_data(config=config, sys_config=sys_config, idx_start_tr = 0, idx_end_tr = 5, idx_start_vl = 35, idx_end_vl = 42, idx_start_ts = 0, idx_end_ts = 2)

    # Create synthetic anomalies
    
    data = load_synthetic_data(data = images_vl,
                        deformation_list = deformation_list,
                        idx_start = 35,
                        idx_end = 42,
                        force_overwrite=True)
    
    
    
                               
    # Test the synthetic anomalies
    import matplotlib.pyplot as plt
    images = data['images']
    masks = data['masks']
    #%%
    data.close()
    image = images[32]
    image.shape    
    




# %%
plt.imshow(images[30,:,:,3,1])
plt.colorbar()
# %%
plt.imshow(images[(30) + 7*64,:,:,3,1])
#plt.imshow(masks[(30) + 7*64,:,:,3,1])
plt.colorbar()
# %%
plt.imshow(images[(30) + 14*64, :,:,3,1])
plt.imshow(masks[(30) + 14*64, :,:,3,1], alpha= 0.1)
# %%
masks[0]
# %%
a = np.ones_like(masks[0])
# %%
b = (a/[10,1,1,1])

# %%
b[...,0]
# %%

patient = images_vl[40]
image = patient[:,:,3,1]
image[image != 0] = 1
plt.imshow(image)
plt.show()


# %%
# We want to binarize the image, all non-zero values are set to 1


# We want to fill in the holes in the image that may have zeros
from scipy.ndimage import binary_fill_holes
image = binary_fill_holes(image, structure=np.ones((3,3))).astype(int)
plt.imshow(image)


# %%
plt.imshow(image)


print(image.shape)

# Generate coordinate grids
x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))

# Calculate center of mass
center_of_mass = np.array([np.sum(x * image) / np.sum(image),
                           np.sum(y * image) / np.sum(image)])

print("Center of Mass:", center_of_mass)

print("Center of Mass:", center_of_mass)

# %%
plt.imshow(image)
plt.scatter(center_of_mass[0], center_of_mass[1], c='r')
# %%
center_of_mass
# %%
from scipy.spatial.distance import cdist

def find_closest_zero_distance(mask, center_of_mass):
    # Find the indices of zero values in the mask
    zero_indices = np.argwhere(mask == 0)
    
    # Calculate the Euclidean distances between the zero indices and the center of mass
    distances = cdist(zero_indices, [center_of_mass])
    
    # Find the minimum distance
    closest_distance = np.min(distances)
    
    return closest_distance


closest_distance = find_closest_zero_distance(image, center_of_mass)
print("Closest distance:", closest_distance)

# %%


# %%
def plot_circle(center, radius):
    fig, ax = plt.subplots()
    
    # Create a circle patch with the given radius and center
    circle = plt.Circle(center, radius, edgecolor='r', facecolor='none')
    
    # Add the circle to the plot
    ax.add_patch(circle)
    
    # Set the aspect ratio to equal to ensure the circle appears circular
    ax.set_aspect('equal')
    
    # Set the x and y limits to include the entire circle
    #ax.set_xlim(center[0] - radius, center[0] + radius)
    #ax.set_ylim(center[1] - radius, center[1] + radius)
    #plt.imshow(image)
    patient[:,:,3,1]
    
    # Show the plot
    plt.show()

# Example usage
center = center_of_mass
radius = closest_distance

plot_circle(center, radius)



# %%
import cv2

def remove_isolated_zeros(binary_mask, min_area=5):
    # Convert the binary mask to uint8 image format
    image = binary_mask.astype(np.uint8)

    # Perform a morphological closing operation to connect nearby ones
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # Perform a morphological opening operation to remove isolated zeros
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Find contours in the opened image
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours based on area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    # Create a new binary mask with the filtered contours
    filtered_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filtered_mask, filtered_contours, -1, 1, thickness=cv2.FILLED)

    return filtered_mask


filtered_mask = remove_isolated_zeros(image, min_area=20)

closest_distance = find_closest_zero_distance(filtered_mask, center_of_mass)
print("Closest distance:", closest_distance)
# %%
# Example usage
center = center_of_mass
radius = closest_distance

plot_circle(center, radius)


# %%
