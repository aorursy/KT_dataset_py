import numpy as np

import h5py

from glob import glob

from scipy.io import loadmat

import os





def load_images(shuffle=True):

    """Load the images from all sinograms



    :return: numpy array containing the images (images, width, height, channel)

    """

    files = glob('/kaggle/input/artificial-edx-elemental-maps/sinograms/*.mat')

    sinograms = [np.array(loadmat(file)['sinogram']) for file in files]



    pre_images = np.concatenate(sinograms, axis=1)

    images = np.zeros((pre_images.shape[1], pre_images.shape[0], pre_images.shape[0]))

    for i in np.arange(pre_images.shape[1]):

        images[i, :, :] = pre_images[:, i, :] / pre_images[:, i, :].max()



    images = np.expand_dims(images, axis=3)

    if shuffle:

        np.random.seed(1)

        np.random.shuffle(images)

    return images



images = load_images()

with h5py.File('images.h5', 'w') as hf:

    hf.create_dataset(name='edx-maps', data=images)