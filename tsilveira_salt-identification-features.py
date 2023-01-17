## This kernel must be run on Python=3.6
import numpy as np 
import pandas as pd  #Python Data Analysis Library
import random

import scipy.ndimage as scipyImg
import scipy.misc as misc

import matplotlib.pyplot as plt 
import seaborn as sns

import os

%matplotlib inline
## Disabling filter warnings
import warnings
warnings.filterwarnings("ignore")
## Defining basic functions
def basic_readImg(directory, filename):
    '''Reading an RGB image through the scipy library. Provides an array.
    Sintaxe: basic_readImg(directory, filename).'''
    sample = scipyImg.imread(directory + filename, mode='RGB')
    if sample.shape[2] != 3:
        return 'The input must be an RGB image.'
    return sample

def basic_showImg(img, size=4):
    ''' Displays the image at the chosen size. The image (img) should be read through basic_readImg().
    Sintaxe: basic_showImg(img, size=4).'''
    plt.figure(figsize=(size,size))
    plt.imshow(img)
    plt.show()
    
def basic_writeImg(directory, filename, img):
    misc.imsave(directory+filename, img)
## Loading the dataset and showing the first rows:
depths = pd.read_csv('../input/depths.csv')
depths.head(2)
train_masks = pd.read_csv('../input/train.csv')
train_masks.head(2)
def rleToMask(rleString,height,width):
    rows,cols = height,width
    try:
        rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
        rlePairs = np.array(rleNumbers).reshape(-1,2)
        img = np.zeros(rows*cols,dtype=np.uint8)
        for index,length in rlePairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
    except:
        img = np.zeros((cols,rows))
    return img
file_imgs = os.listdir(path='../input/train/images/')
file_masks = os.listdir(path='../input/train/masks/')
print('Images found: {0}\nCorresponding masks: {1}'.format(len(file_imgs), len(file_masks)))
## Defining a function since there's sample without valid RLE.
def choose_sample(data=train_masks):
    ## Choosing a random image from train dataset:
    sample = random.choice(range(len(data)))

    ## Parsing the sample information:
    sample_id = data['id'][sample]
    sample_depth = depths[depths['id'] == sample_id]['z'].values[0]
    sample_RLEstring = data['rle_mask'][sample]
    try: 
        sample_RLE = rleToMask(sample_RLEstring, 101,101)
    except: 
        sample_RLE = np.zeros((101,101))
    file_name = sample_id + '.png'
    sample_img = basic_readImg('../input/train/images/',file_name)
    sample_mask = basic_readImg('../input/train/masks/',file_name)
    
    fig1, axes = plt.subplots(1,3, figsize=(10,4))
    axes[0].imshow(sample_img)
    axes[0].set_xlabel('Subsurface image')
    axes[1].imshow(sample_mask)
    axes[1].set_xlabel('Provided mask')
    axes[2].imshow(sample_RLE)
    axes[2].set_xlabel('Decoded RLE mask')
    fig1.suptitle('Image ID = {0}\nDepth = {1} ft.'.format(sample_id, sample_depth));
    return
choose_sample()
df1 = depths.set_index('id')
df2= train_masks.set_index('id')
dataset = pd.concat([df1, df2], axis=1, join='inner')
dataset = dataset.reset_index()
dataset['mask'] = dataset['rle_mask'].apply(lambda x: rleToMask(x, 101,101))
def salt_proportion(imgArray):
    try: 
        unique, counts = np.unique(imgArray, return_counts=True)
        ## The total number of pixels is 101*101 = 10,201
        return counts[1]/10201.
    except: 
        return 0.0
dataset['salt_proportion'] = dataset['mask'].apply(lambda x: salt_proportion(x))
dataset.head()
sns.set();
sns.distplot(dataset['z'], bins=20)
sns.pairplot(dataset, vars=['z', 'salt_proportion'])
dataset['target'] = pd.cut(dataset['salt_proportion'], bins=[0, 0.001, 0.1, 0.4, 0.6, 0.9, 1.0], 
       include_lowest=True, labels=['No salt', 'Very low', 'Low', 'Medium', 'High', 'Very high'])
dataset.tail()
