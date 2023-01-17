from IPython.display import HTML



HTML('<center><iframe width="560" height="315" src="https://www.youtube.com/embed/1IBvrOBQ268" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')
!conda install -c conda-forge gdcm -y

!pip install pandas-profiling -y
# Setup

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib import animation, rc

import seaborn as sns



sns.set_style('darkgrid')

import pydicom as dcm

import scipy.ndimage

import gdcm

import glob

import imageio

from IPython import display



from skimage import measure 

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage.morphology import disk, opening, closing

from tqdm import tqdm



from IPython.display import HTML

from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



from os import listdir, mkdir



path = "../input/rsna-str-pulmonary-embolism-detection/"

files = glob.glob(path+'/train/*/*/*.dcm')
# Reading Data

train = pd.read_csv(path + "train.csv")

test = pd.read_csv(path + "test.csv")

print("Train Data Shape:",train.shape)

print("Test Data Shape:",test.shape)
train.head(5).T
train.info()
test.info()
train.describe()
test.describe()
print('Missing values in train data:',train.isnull().sum().sum())

print('Missing values in test data:',test.isnull().sum().sum())
from pandas_profiling import ProfileReport

profile = ProfileReport(train, title='Training Data Report')
profile.to_notebook_iframe()
cols = train.copy()

cols.drop(['StudyInstanceUID','SeriesInstanceUID','SOPInstanceUID'],axis=1,inplace=True)

columns = cols.columns



corr = cols.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(12, 12))

    ax = sns.heatmap(corr,mask=mask,square=True,linewidths=.8,cmap="gnuplot",annot=True)
fig, ax = plt.subplots(7,2,figsize=(16,28))

for i,col in enumerate(columns): 

    plt.subplot(7,2,i+1)

    sns.countplot(cols[col],palette="gnuplot")   
from random import randint

# This function reads dicom images from the given path

def load_dicom(path):

    files = listdir(path)

    f = [dcm.dcmread(path + "/" + str(file)) for file in files]

    return f



random_integer = randint(0,len(train))

example = path + "train/" + train.StudyInstanceUID.values[random_integer] +'/'+ train.SeriesInstanceUID.values[random_integer]

scans = load_dicom(example)

scans[randint(0,len(scans))]
f, plots = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(15, 15))

for i in range(9):

    plots[i // 3, i % 3].axis('off')

    plots[i // 3, i % 3].imshow(dcm.dcmread(np.random.choice(files[:3000])).pixel_array,cmap='gist_earth_r')
fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(dcm.dcmread(np.random.choice(files[:3000])).pixel_array, cmap="gist_earth_r")
def load_slice(path):

    slices = [dcm.read_file(path + '/' + s) for s in listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



def transform_to_hu(slices):

    images = np.stack([file.pixel_array for file in slices])

    images = images.astype(np.int16)



    # convert ouside pixel-values to air:

    # I'm using <= -1000 to be sure that other defaults are captured as well

    images[images <= -1000] = 0

    

    # convert to HU

    for n in range(len(slices)):

        

        intercept = slices[n].RescaleIntercept

        slope = slices[n].RescaleSlope

        

        if slope != 1:

            images[n] = slope * images[n].astype(np.float64)

            images[n] = images[n].astype(np.int16)

            

        images[n] += np.int16(intercept)

    

    return np.array(images, dtype=np.int16)
first_patient = load_slice(path+'/train/0003b3d648eb/d2b2960c2bbf')

first_patient_pixels = transform_to_hu(first_patient)



fig, plots = plt.subplots(8, 10, sharex='col', sharey='row', figsize=(20, 16))

for i in range(80):

    plots[i // 10, i % 10].axis('off')

    plots[i // 10, i % 10].imshow(first_patient_pixels[i], cmap="gray") 
imageio.mimsave("/tmp/gif.gif", first_patient_pixels, duration=0.08)

display.Image(filename="/tmp/gif.gif", format='png')
HTML('<center><iframe width="560" height="315" src="https://www.youtube.com/embed/SdYUniRMtz4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')