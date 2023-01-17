import dicom

import cv2

import os

import pandas as pd

import numpy as np

import glob

import pickle

import scipy.ndimage

from operator import itemgetter

from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection





INPUT_FOLDER = '../input/sample_images/'

patients = os.listdir(INPUT_FOLDER)

patients.sort()





# Load the scans in given folder path

def load_scan(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:

        s.SliceThickness = slice_thickness

    return slices





def get_pixels_hu(scans):

    image = np.stack([s.pixel_array for s in scans])

    # Convert to int16 (from sometimes int16),

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)

    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)





def resample(image, scan, new_spacing=[1, 1, 1]):

    # Determine current pixel spacing

    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))

    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing





def normalize(image):

    MIN_BOUND = -1000.0

    MAX_BOUND = 400.0

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)

    image[image > 1] = 1.

    image[image < 0] = 0.

    return image
# Function which creates set of images for each axis

def create_set_of_png_for_patient(patient):

    needed_shape = (640, 640)

    first_patient = load_scan(INPUT_FOLDER + patient)

    first_patient_pixels = get_pixels_hu(first_patient)

    print('Number of scans: {}'.format(len(first_patient_pixels)))

    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])

    print("Shape before resampling:", first_patient_pixels.shape)

    print("Shape after resampling:", pix_resampled.shape)

    print("X Slices")

    total = 0

    image_list = []

    for i in range(pix_resampled.shape[0]):

        im = pix_resampled[i, :, :]

        image_list.append(cv2.resize(255*normalize(im), needed_shape))

    print("Y Slices")

    for i in range(pix_resampled.shape[1]):

        im = pix_resampled[:, i, :]

        image_list.append(cv2.resize(255*normalize(im), needed_shape))

    print("Z Slices")

    for i in range(pix_resampled.shape[2]):

        im = pix_resampled[:, :, i]

        image_list.append(cv2.resize(255*normalize(im), needed_shape))

    return image_list
def create_video(image_list, out_file):

    height, width = image_list[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'X264')

    # In case X264 doesn't work

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')

    fps = 30.0

    video = cv2.VideoWriter(out_file, fourcc, fps, (width, height), False)

    for im in image_list:

        video.write(im.astype(np.uint8))

    cv2.destroyAllWindows()

    video.release()
if __name__ == '__main__':

    p = patients[1]

    print('Create video for {}'.format(p))

    image_list = create_set_of_png_for_patient(p)

    create_video(image_list, "output.avi")