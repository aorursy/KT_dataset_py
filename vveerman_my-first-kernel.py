# First we load som useful packages 

%matplotlib inline 

# to make sure all pictures and images are diplayed in the notebook



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import dicom # stands for : Digital imaging and communications in medicine

# is a standard for handling, storing, printing, and transmitting information in medical imaging.

import os

import scipy.ndimage # Multi-dimensional image processing

import matplotlib.pyplot as plt # ploting module 



from skimage import measure, morphology # image processing. 

from mpl_toolkits.mplot3d.art3d import Poly3DCollection # provides some basic 3D plotting (scatter, surf, line, mesh) tools



# Some constants 

INPUT_FOLDER = '../input/sample_images/'

patients = os.listdir(INPUT_FOLDER) 

# The method listdir(path) returns a list containing the names of the entries in the directory given by path

patients.sort()





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# So basicly you first have the folder with sample images. 

# There are a number of patients set in the patients variable

# each have a certain number of pictures 

# for example lets look at the list of dicom files for patient number 0 



path_0 = os.listdir(INPUT_FOLDER + patients[0]) 

path_0
# Now we can have a look at the first element in that folder 



slice_0 = [dicom.read_file(INPUT_FOLDER + patients[0] + '/' + path_0[0])]

slice_0

# you can see that there are any different infos in a single dcm file 
# There are a number of attibutes in the dcm files that you can use 

dir(slice_0[0])
#slice_0[2].ImagePositionPatient[2]

# there are several pices of infomation that are quiet important in the dcm files 

# the first is slide location 

# this is the attibute tha gives you the position were the radio image was taken

slice_0[0].SliceLocation
# the second is the Pixel data that contains the data of the CT scan and enable to plot the data 

slice_0[0].pixel_array
# However, the standard in medical imagiary is the Hounsefield unit. This is obtained by modifing the pixel data as follows

# hu = pixel*slope + intercept

slope = slice_0[0].RescaleSlope

intercept = slice_0[0].RescaleIntercept

# The instance Number is also important. it tells you the CT number in the set of CT scans for a given patient
image = np.stack([slice_0[0].pixel_array])



image = np.array(image, dtype=np.int16)

image[image == -2000] = 0

image = image.astype(np.int16)

image = slice_0[0].RescaleSlope * image.astype(np.float64)

image = image.astype(np.int16)

image += np.int16(slice_0[0].RescaleIntercept)



plt.hist(image.flatten(), bins=80, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")







# from the char below we see the Housefield Unit 

# this is a measure of the radio density 

# -2000 corresponds to air 

# 0 is water 

# 30 to 45 is blood

# +10 to +40 and so on
# Load the scans in given folder path

# now this function will get all the dcm files for a given patient and assign it a slice thickness between each picture



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
first_patient = load_scan(INPUT_FOLDER + patients[0])

# first patient now contains all of the info in the dcm files for patient[0]. basicly all the dcm files opened up

first_patient_pixels = get_pixels_hu(first_patient)

# this will contain a set of arrays of all the dcm files of patient[0]. H

plt.hist(first_patient_pixels.flatten(), bins=80, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()



# Show some slice in the middle

plt.imshow(first_patient_pixels[0], cmap=plt.cm.gray)

plt.show()