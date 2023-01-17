import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dicom

import os

import scipy.ndimage

import matplotlib.pyplot as plt



from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# Some constants 

INPUT_FOLDER = '../input/sample_images/'

patients = os.listdir(INPUT_FOLDER)

patients.sort()
# Load the scans in given folder path

def load_scan(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices
def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        

        if slope != 1:

            image[slice_number] = slope * image[slice_number].astype(np.float64)

            image[slice_number] = image[slice_number].astype(np.int16)

            

        image[slice_number] += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)
def get_unique_counts(hu_pixels):

    unique, counts = np.unique(hu_pixels, return_counts=True)

    unique_counts = np.asarray((unique, counts)).T



    return unique_counts





def get_unique_counts_vector(unique_counts):

    neg_vec = np.zeros(1500, dtype=np.int32)

    pos_vec = np.zeros(2001, dtype=np.int32)



    for count in unique_counts:

        if -1500 <= count[0] <= 2000:

            if count[0] < 0:

                neg_vec[(count[0]*-1)-1] = count[1]

            else:

                pos_vec[count[0]-1] = count[1]



    neg_vec = neg_vec[::-1]

    vec = np.append(neg_vec, pos_vec)



    return vec
def create_header(filename, test=False):

    header = ['ID'] + range(-1500, 2001)

    if not test:

        header += ['CANCER']

    with open(filename, 'wb') as f:

        writer = csv.writer(f)

        writer.writerows([header])



def process_training_data(filename, patients, labels):

    with open(filename, 'a') as output:

        for patient in patients:

            try:

                cancer = int(labels.loc[labels['id'] == patient]['cancer'])

                slices = load_slices(INPUT_FOLDER + patient)

                pixels = get_pixels_hu(slices)

                unique_counts = get_unique_counts(pixels)

                counts_vec = get_unique_counts_vector(unique_counts)



                row = [patient] + list(counts_vec) + [cancer]

                writer = csv.writer(output)

                writer.writerows([row])

            except:

                print(patient)

                

def process_test_data(filename, test_patients):

    with open(filename, 'a') as output:

        for ix, s in test_patients.iterrows():

            try:

                patient = s.get(0)

                slices = load_slices(INPUT_FOLDER + patient)

                pixels = get_pixels_hu(slices)

                unique_counts = get_unique_counts(pixels)

                counts_vec = get_unique_counts_vector(unique_counts)



                row = [patient] + list(counts_vec)

                writer = csv.writer(output)

                writer.writerows([row])

            except:

                print(patient)

                

# test_patients = pd.read_csv('../input/sample_images/stage1_sample_submission.csv', index_col=None)



# patients = os.listdir(INPUT_FOLDER)

# labels = pd.read_csv('../input/stage1_labels.csv', index_col=0)

# create_header('hu_counts.csv')

# process_training_data('hu_counts.csv', patients, labels) 