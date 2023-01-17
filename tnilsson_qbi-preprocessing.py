import pydicom as dicom

import re

import os

import numpy as np
class Patient(object):

    def __init__(self, directory, subdir):

        # deal with any intervening directories

        while True:

            subdirs = next(os.walk(directory))[1]

            if len(subdirs) == 1:

                directory = os.path.join(directory, subdirs[0])

            else:

                break



        slices = []

        for s in subdirs:

            m = re.match("sax_(\d+)", s)

            if m is not None:

                slices.append(int(m.group(1)))



        slices_map = {}

        first = True

        times = []

        for s in slices:

            files = next(os.walk(os.path.join(directory, "sax_%d" % s)))[2]

            offset = None



            for f in files:

                m = re.match("IM-(\d{4,})-(\d{4})\.dcm", f)

                if m is not None:

                    if first:

                        times.append(int(m.group(2)))

                    if offset is None:

                        offset = int(m.group(1))



            first = False

            slices_map[s] = offset



        self.directory = directory

        self.time = sorted(times)

        self.slices = sorted(slices)

        self.slices_map = slices_map

        self.name = subdir



    def _filename(self, s, t):

        fname = os.path.join(self.directory,

                                 "sax_%d" % s, 

                                 "IM-%04d-%04d.dcm" % (self.slices_map[s], t))

        return fname



    def _read_dicom_image(self, filename):

        d = dicom.read_file(filename)

        img = d.pixel_array

        return np.array(img)



    def _read_all_dicom_images(self):

        f1 = self._filename(self.slices[0], self.time[0])

        f2 = self._filename(self.slices[1], self.time[0])

        

        d1 = dicom.read_file(f1)

        d2 = dicom.read_file(f2)

        

        (x, y) = d1.PixelSpacing

        (x, y) = (float(x), float(y))

        self.col_scaling = x

        self.row_scaling = y

        

        # try a couple of things to measure distance between slices

        try:

            dist = np.abs(d2.SliceLocation - d1.SliceLocation)

        except AttributeError:

            try:

                dist = d1.SliceThickness

            except AttributeError:

                dist = 8  # better than nothing...



        # 4D image array

        self.images = np.array([[self._read_dicom_image(self._filename(d, i))

                                for i in self.time]

                                for d in self.slices])

        

        # Distance between slices in mm

        self.dist = dist

        

        # Calculate depth as distance between slices times no. of slices

        self.deph_mm = self.dist * (self.images.shape[0] - 1)

        

        # Area scaling, mm per pixel

        self.area_multiplier = x * y

        

        # Orientation

        self.orientation = d1.ImageOrientationPatient

        

    def load(self):

        self._read_all_dicom_images()
def load_patient(patient_id, root_dir):

    patient_id = str(patient_id)

    base_path = os.path.join(root_dir, patient_id)

    try:

        tdata = Patient(base_path, patient_id)

        tdata.load()

        # If data does not contain 4 dimensions, throw it away

        if len(tdata.images.shape) == 4:

            return tdata

    except (ValueError, TypeError, IndexError, AttributeError, FileNotFoundError):

        return None

    

def load_multiple_patients(root_dir, verbose=False):

    """

    :param patient_ids: ids of patients to load [list of integers]

    :param root_dir: name of root dir, defaults to Kaggle root directory [string]

    :param verbose: Whether to print every patient id when loading [boolean]

    :return: list of [Patient] objects

    """

    # Inspect all ids in the directory

    patient_ids = sorted(os.listdir(root_dir))

    patient_list = []

    for pid in patient_ids:

        if verbose:

            print('Loading patient %i...' % pid)

        p_data = load_patient(pid, root_dir=root_dir)

        if p_data:

            patient_list.append(p_data)

    return patient_list
import numpy as np



# Based on https://gist.github.com/ajsander/fb2350535c737443c4e0#file-tutorial-md

def fourier_time_transform_slice(image_3d):

    '''

    3D array -> 2D array

    [slice, height, width] -> [height, width]

    Returns (width, height) matrix

    Fourier transform for 3d data (time,height,weight)

    '''

    # Apply FFT to the selected slice

    fft_img_2d = np.fft.fftn(image_3d)[1, :, :]

    return np.abs(np.fft.ifftn(fft_img_2d))





def fourier_time_transform(patient_images):

    '''

    4D array -> 3D array (compresses time dimension)

    Concretely, [slice, time, height, width] -> [slice, height, width]

    Description: Fourier transform for analyzing movement over time.

    '''



    ftt_image = np.array([

        fourier_time_transform_slice(patient_slice)

        for patient_slice in patient_images

    ])

    return ftt_image
import numpy as np

from skimage.morphology import binary_dilation, binary_erosion, binary_opening, binary_closing, disk

from skimage.filters import threshold_otsu





def thresh_segmentation(patient_img):

    """Returns matrix

    Segmententation of patient_img with k-means

    """

    #Z = np.float32(np.ravel(patient_img))

    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    #flags = cv2.KMEANS_RANDOM_CENTERS

    #compactness, labels, centers = cv2.kmeans(Z, 2, None, criteria, 10, flags)

    #center = np.uint8(centers)

    thresh = threshold_otsu(patient_img)

    binary = patient_img > thresh

    return binary



def segment_multiple(patient_img):

    """Returns list

    List of segmented slices with function thresh_segmentation()

    """

    num_slices, height, width = patient_img.shape

    segmented_slices = np.zeros((num_slices, height, width))



    for i in range(num_slices):

        seg_slice = thresh_segmentation(patient_img[i])

        if seg_slice.sum() > seg_slice.size * 0.5:

            seg_slice = 1 - seg_slice

        segmented_slices[i] = seg_slice



    return segmented_slices



def roi_mean_yx(patient_img):

    """Returns mean(y) and mean(x) [double]

    Mean coordinates in segmented patients slices.

    This function performs erosion to get a better result.

    Original: See https://nbviewer.jupyter.org/github/kmader/Quantitative-Big-Imaging-2019/blob/master/Lectures/06-ShapeAnalysis.ipynb

    """

    seg_slices = segment_multiple(patient_img)

    num_slices = seg_slices.shape[0]

    y_all, x_all = np.zeros(num_slices), np.zeros(num_slices)

    neighborhood = disk(2)

    

    for i,seg_slice in enumerate(seg_slices):

        # Perform erosion to get rid of wrongly segmented small parts

        seg_slices_eroded = binary_erosion(seg_slice, neighborhood) 

        

        # Filter out background of slice, after erosion [background=0, foreground=1]

        y_coord, x_coord = seg_slices_eroded.nonzero()

        

        # Save mean coordinates of foreground 

        y_all[i], x_all[i] = np.mean(y_coord), np.mean(x_coord)

    

    # Return mean of mean foregrounds - this gives an estimate of ROI coords.

    mean_y = int(np.mean(y_all))

    mean_x = int(np.mean(x_all))

    return mean_y, mean_x
from skimage import exposure



def histogram_normalize_4d(images, clip_limit=0.03):

    slices, time, _, _ = images.shape

    norm_imgs_4d = np.empty(images.shape)

    for i in range(slices):

        for j in range(time):

            norm_imgs_4d[i,j] = exposure.equalize_adapthist(images[i,j].astype(np.uint16), 

                                                            clip_limit=clip_limit)

    return norm_imgs_4d
import cv2



def rescale_patient_4d_imgs(patient, img_4d):

    if len(img_4d.shape) < 4: raise Exception("Patient images are not 4D!")

    num_slices, time, _, _ = img_4d.shape

    

    # Extract scaled DICOM width/height multipliers

    # http://dicom.nema.org/dicom/2013/output/chtml/part03/sect_10.7.html

    fx, fy = patient.col_scaling, patient.row_scaling

    

    # Rescale the first 2d image, in order to find out the resulting dimensions

    example_img = cv2.resize(src=img_4d[0,0], dsize=None, fx=fx, fy=fy)

    scaled_height, scaled_width = example_img.shape

    scaled_imgs = np.zeros((num_slices, time, scaled_height, scaled_width))

    

    for i in range(num_slices):

        for j in range(time):

            scaled_imgs[i,j] = cv2.resize(src=img_4d[i,j], dsize=None, fx=fx, fy=fy)

    

    return scaled_imgs



def crop_roi(img, dim_y, dim_x, cy, cx):

    """

    Crops an image from the given coords (cy, cx), such that the resulting img is of

    dimensions [dim_y, dim_x], i.e. height and width.

    Resulting image is filled out from top-left corner, and remaining pixels are left black.

    """

    cy, cx = int(round(cy)), int(round(cx))

    h, w = img.shape

    if dim_x > w or dim_y > h: raise ValueError('Crop dimensions larger than image dimension!')

    new_img = np.zeros((dim_y, dim_x))

    dx, dy = int(dim_x / 2), int(dim_y / 2)

    dx_odd, dy_odd = int(dim_x % 2 == 1), int(dim_y % 2 == 1)



    # Find boundaries for cropping [original img]

    dx_left = max(0, cx - dx)

    dx_right = min(w, cx + dx + dx_odd)

    dy_up = max(0, cy - dy)

    dy_down = min(h, cy + dy + dy_odd)



    # Find how many pixels to fill out in new image

    range_x = dx_right - dx_left

    range_y = dy_down - dy_up

    



    # Fill out new image from top left corner

    # Leave pixels outside range as 0's (black)

    new_img[0:range_y, 0:range_x] = img[dy_up:dy_down, dx_left:dx_right]

    return new_img



def crop_heart(images_4d, heart_pixel_size=200):

    # Find center for cropping

    ft_imges = fourier_time_transform(images_4d)

    y, x = roi_mean_yx(ft_imges)

    

    # Create new 4d image array

    num_slices, time, h, w = images_4d.shape

    heart_cropped_img_4d = np.zeros((num_slices, time, heart_pixel_size, heart_pixel_size))

    

    for i in range(num_slices):

        for j in range(time):

            heart_cropped_img_4d[i,j] = crop_roi(images_4d[i,j], heart_pixel_size, heart_pixel_size, y, x)

    

    return heart_cropped_img_4d



def rotate_images_210_deg(images_4d, orientation):

    """

    Return 4d image

    Params 4d numpy, int

    Idea from: kaggle.com/c/second-annual-data-science-bowl/discussion/19378

    Description: 

                Rotates image if orientation angle is -30 degreees, which ensures

                that the left ventricle is in the top right corner of the image.

    """

    angle = np.arctan2(orientation[:3], orientation[:3]) / np.pi * 180 - 75

    rotation_needed = angle[2] > (-210)

    

    # Check if rotation needed

    if rotation_needed:

        # Calculate resulting dimensions for numpy array

        slices, time, _, _ = images_4d.shape

        rot_width, rot_height = np.rot90(images_4d[0,0], k=1).shape

        rot_images = np.zeros((slices, time, rot_width, rot_height))

        

        # Rotate images

        for i in range(slices):

            for j in range(time):

                rot_images[i,j] = np.rot90(images_4d[i,j], k=1)

        return rot_images

    

    # Otherwise if no rotation needed, return original images

    return images_4d
from skimage.morphology import opening, disk

from scipy.ndimage import distance_transform_edt

from skimage.morphology import watershed

from skimage.feature import peak_local_max



# Code from: https://nbviewer.jupyter.org/github/kmader/Quantitative-Big-Imaging-2019/blob/master/Lectures/07-ComplexObjects.ipynb

def watershed_img(image):

    # Distance map

    image_dmap = distance_transform_edt(image)

    # Distance peaks

    image_peaks = label(peak_local_max(image_dmap, indices=False, footprint=np.ones((40, 40)),labels=image, exclude_border=True))

    # Watershed first once

    ws_labels = watershed(-image_dmap, image_peaks, mask=image)

    

    # Reomve small segments

    label_area_dict = {i: np.sum(ws_labels == i)for i in np.unique(ws_labels[ws_labels > 0])}

    clean_label_maxi = image_peaks.copy()

    lab_areas = list(label_area_dict.values())

    # Remove 20 percentile

    area_cutoff = np.percentile(lab_areas, 15)

    for i, k in label_area_dict.items():

        if k <= area_cutoff:

            clean_label_maxi[clean_label_maxi == i] = 0

    # Watershed again

    ws_labels = watershed(-image_dmap, clean_label_maxi, mask=image)



    return ws_labels
from skimage.measure import label



def labeled_segmented_images(images):

    """

    Returns numpy array (4d)

    Segments image and used watershed for labeling.

    """

    

    num_slices, time, height, width = images.shape

    segmented_slices = np.zeros((num_slices, time, height, width))

    

    # Iterate over all slices and whole timeseries for images

    for i in range(num_slices):

        for j in range(time):

            # Segmentation

            seg_slice = thresh_segmentation(images[i,j])

            

            # Watershed

            labels = watershed_img(seg_slice)

            

            # Writes labeled segmented object to return images                     

            segmented_slices[i,j] = labels



    return segmented_slices.astype(np.uint8)
from skimage.measure import regionprops



def find_left_ventricle(images):

    """

    Returns numpy array (4d)

    Finds left ventricle from labeled segmented images

    """

    

    num_slices, time, height, width = images.shape

    segmented_slices = np.zeros((num_slices, time, height, width))

    

    all_labels = labeled_segmented_images(images)

    

    # Iterate over all slices and whole timeseries for images

    for i in range(num_slices):

        for j in range(time):

            

            labels = all_labels[i,j]

            min_dist = 50

            min_dist_label = 0

            segment_found =  False

            

            # Iterate over every label in watershed labels to predict which is the left ventricle.

            for label in np.unique(labels):

        

                # yx coordinates for labaled segmentation

                yx_coord_labels = np.where(labels == label)

                

                # Do not count small or big segmatations (removes dots and background)

                if len(yx_coord_labels[0]) > 8000 or len(yx_coord_labels[0]) < 100:

                    continue

                

                # Upper right middle coordinates

                cx = 3*(height/4)

                cy = width/4

                

                # Calculates euclidiean distance between mean coordinates for segmentated labels and middle of image

                euclidiean_dist = np.sqrt((int(cy)-np.mean(yx_coord_labels[0]))**2+(int(cx)-np.mean(yx_coord_labels[1]))**2)

                

                # Gets min distance

                if euclidiean_dist < min_dist:

                    

                    # Check if segment shape is round.

                    regions = regionprops((labels == label).astype(int))

                    props = regions[0]

                    y0, x0 = props.centroid

                    orientation = props.orientation

                    x1 = x0 + np.cos(orientation) * 0.5 * props.major_axis_length

                    y1 = y0 - np.sin(orientation) * 0.5 * props.major_axis_length

                    x2 = x0 - np.sin(orientation) * 0.5 * props.minor_axis_length

                    y2 = y0 - np.cos(orientation) * 0.5 * props.minor_axis_length

                

                    d1_dist = np.sqrt(abs(x0-x1)**2+abs(y0-y1)**2)

                    d2_dist = np.sqrt(abs(x0-x2)**2+abs(y0-y2)**2)

                    

                    # Checks if segment is round.

                    if abs(d1_dist-d2_dist) > 20:

                        continue

                    

                    min_dist_label = label

                    min_dist = euclidiean_dist

                    segment_found = True

            

            # Checks if we found a image or not

            if segment_found:

                # Writes segmented object to return images                     

                segmented_slices[i,j] = (labels == min_dist_label).astype(int)

            else:

                segmented_slices[i,j] = np.zeros(labels.shape)

                

    return segmented_slices.astype(np.uint8), all_labels.astype(np.uint8)
from scipy.misc import imresize



def take_middle_slice(images_4d, expand_dims=True):

    """Converts 4d image array to 3d image array"""

    middle = int(images_4d.shape[0] / 2)

    images_3d = images_4d[middle]

    

    # Expand dims, effectively makes the first dimension 1, only necessary if the pipeline expects 4d images

    if expand_dims:

        images_3d = np.expand_dims(images_3d, axis=0)

    return images_3d



def resize_4d_img(image_4d, img_size):

    n_slices, n_time = image_4d.shape[0], image_4d.shape[1]

    resized_imgs = []

    for img_3d in image_4d:

        resized_3d = []

        for img_2d in img_3d:

            resized_2d = imresize(img_2d, (img_size, img_size))

            resized_3d.append(resized_2d)

        resized_imgs.append(resized_3d)



    return np.array(resized_imgs)



def preprocess_pipeline(patient, heart_pixel_size=180):

    """

    [Patient Object] -> [4D np.array] (segmented left ventricle)

    

    Preprosessing pipeline for patient:

        1. Rescale images (1 pixel = 1 mm)

        2. Histogram Normalize (some images are brighter than others)

        3. Crop images aroind ROI (identified using Fourier Transform over time)

        4. Rotate images (such that left ventricle is in top right part of img)

        5. Segment out left ventricle (for each 2d slice)

    """

    

    # Take middle SAX slice, best approach uses this

    middle_slice_3d_images = take_middle_slice(patient.images)

    

    # Rescale images such that 1 pixel = 1 mm

    rescaled_imgs = rescale_patient_4d_imgs(patient, middle_slice_3d_images)

    

    # Histogram normalize

    normalized_imgs = histogram_normalize_4d(rescaled_imgs)

    

    # Crop around ROI

    cropped_imgs = crop_heart(normalized_imgs, heart_pixel_size=heart_pixel_size)

   

    # Rotate images

    rotated_images = rotate_images_210_deg(cropped_imgs, patient.orientation)

    

    # Downcast to float 16 bit

    return rotated_images.astype(np.float16)
import numpy as np

import matplotlib.pyplot as plt

from skimage.util import montage as montage2d



montage3d = lambda x, **k: montage2d(np.stack([montage2d(y, **k) for y in x], 0))



def plot_patient_slices_3d(patient_slices, title=False, figsize=(20, 20)):

    patient_slices = patient_slices.astype(np.float64)

    '''Plots a 2D image per slice in series (3D in total)'''

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    image = montage2d(patient_slices)

    if title: ax.set_title(title)

    ax.imshow(image, cmap='bone')





def plot_patient_data_4d(patient_data, all_slices=False, num_slices=[0], figsize=(20, 20)):

    patient_data = patient_data.astype(np.float64)

    '''Plots a 3D image per time step in patient data (4D in total)'''

    if all_slices:

        # Number of slices is equal to the first dimension of the patient image array

        num_slices = range(patient_data.shape[0])

    for i in num_slices:

        plot_patient_slices_3d(patient_data[i],

                               title=('Showing slice %i' % i))
train_dir = os.path.join('..', 'input', 'train', 'train')

test_dir = os.path.join('..', 'input', 'test', 'test')
test_patient = load_patient(30, train_dir)

processed_imgs = preprocess_pipeline(test_patient)

plot_patient_data_4d(processed_imgs)
def gb_used(numpy_array):

    return numpy_array.nbytes / (1024**3)



from keras.utils.generic_utils import Progbar

import pandas as pd

import numpy as np
def make_dataset(dataset_type, limit=None):

    """

    dataset_type: train/validate/test

    """

    TEST = dataset_type == "test"

    if not TEST:

        df = pd.read_csv("../input/%s.csv" % dataset_type)

    

    directory = os.path.join('..', 'input', dataset_type, dataset_type)

    patient_ids = [int(i) for i in os.listdir(directory)[:limit]]



    # Use lists to avoid index errors which causes everything to crash...

    X = []

    y = []



    n_patients = len(patient_ids)

    progbar = Progbar(n_patients)

    X_mask = []

    error_patients = []

    print("Processing %i patients..." % n_patients)



    progbar.add(0)

    for i, pid in enumerate(patient_ids):

        try:

            p = load_patient(pid, directory)

            if p != None:

                # Add the processed middle slice as the X_train[i]

                X.append(preprocess_pipeline(p)[0])

                if not TEST:

                    # Extract the ground truth data for the patient id

                    y_truth = np.array(df[df.Id == pid].iloc[:, 1:])[0]

                    y.append(y_truth)            

                # Mark patient as used

                X_mask.append(pid)

            else:

                error_patients.append(pid)

        except ValueError:

            error_patients.append(pid)

        # Update progress bar

        progbar.add(1)



    # Convert to numpy arrays

    X = np.array(X)

    y = np.array(y)

    X_mask = np.array(X_mask)



    print("Finished processing %i patients!" % n_patients)

    np.savez("X_%s_mask" % dataset_type, X_mask)

    np.savez("X_%s" % dataset_type, X)

    if not TEST:

        np.savez("y_%s" % dataset_type, y)

    print("Patient data saved!")



    print("The following patients could not be loaded:")

    print(error_patients)

    

    print("Total space usage:", 

      gb_used(X) + gb_used(y), 

      "GB")
make_dataset("train", limit=None)
make_dataset("validate", limit=None)
make_dataset("test", limit=None)
os.listdir("../working")