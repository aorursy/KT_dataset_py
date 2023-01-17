%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dicom

import os

import scipy.ndimage

import matplotlib.pyplot as plt

from PIL import Image



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
if (len(patients) == 21):

    del patients[0]

print(len(patients))



for patient in patients:

    path = INPUT_FOLDER + patient

    patient_slices = load_scan(path)

    stacked_slices = get_pixels_hu(patient_slices)

    

    N=len(stacked_slices)    

    print ("number of slices: ", N)

    arr = np.zeros((512, 512), np.int16)

    count = 3

    for im in stacked_slices:

        smallest = np.amin(im)

        biggest = np.amax(im)

        

        #imarr = np.array(im, dtype=np.int16)        

        arr = arr + (1 - im) * np.log(count)/(biggest - smallest)



        #print ((N * 14)/ np.log10(count))

        count = count + 1

        #arr = np.array(np.round(arr), dtype=np.uint8)

        arr = np.array(np.round(arr),dtype=np.uint8)

    #out=Image.fromarray(arr, mode='L')



    imName = patient + ".jpeg"

    print(imName)

    plt.imshow(arr, cmap=plt.cm.gray)

    plt.show()

#out.save(imName)

#plt.show()