import os

print(os.listdir("../input"))

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from glob import glob

import PIL

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
data = pd.read_csv('../input/covid19-ct-scans/metadata.csv')

data.sample(5)
def read_nii(filepath):

    ct_scan = nib.load(filepath)

    array   = ct_scan.get_fdata()

    array   = np.rot90(np.array(array))

    return(array)

import nibabel as nib

sample_ct   = read_nii(data.loc[0,'ct_scan'])

sample_lung = read_nii(data.loc[0,'lung_mask'])

sample_infe = read_nii(data.loc[0,'infection_mask'])

sample_all  = read_nii(data.loc[0,'lung_and_infection_mask'])
def plot_sample(array_list, color_map = 'nipy_spectral'):

    fig = plt.figure(figsize=(18,15))



    plt.subplot(1,4,1)

    plt.imshow(array_list[0], cmap='bone')

    plt.title('Original Image')



    plt.subplot(1,4,2)

    plt.imshow(array_list[0], cmap='bone')

    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)

    plt.title('Lung Mask')



    plt.subplot(1,4,3)

    plt.imshow(array_list[0], cmap='bone')

    plt.imshow(array_list[2], alpha=0.5, cmap=color_map)

    plt.title('Infection Mask')



    plt.subplot(1,4,4)

    plt.imshow(array_list[0], cmap='bone')

    plt.imshow(array_list[3], alpha=0.5, cmap=color_map)

    plt.title('Lung and Infection Mask')



    plt.show()
plot_sample([sample_ct[...,120], sample_lung[...,120], sample_infe[...,120], sample_all[...,120]])
from nibabel.testing import data_path

example_filename = os.path.join(data_path, '/kaggle/input/covid19-ct-scans/ct_scans/coronacases_org_008.nii')
img = nib.load(example_filename)
import matplotlib.pyplot as plt

im_fdata=img.get_fdata()



plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(20):

    plt.subplot(5, 5, i + 1)



    plt.imshow(im_fdata[:,:,i])

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()  
import matplotlib.pyplot as plt

im_fdata=img.get_fdata()



plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(20):

    plt.subplot(5, 5, i + 1)



    plt.imshow(im_fdata[i,:,:])

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()  