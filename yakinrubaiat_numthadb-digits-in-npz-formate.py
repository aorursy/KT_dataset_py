import numpy as np

import pandas as pd

import os

import sklearn

import matplotlib.pyplot as plt

from scipy.misc.pilutil import Image

import cv2

import glob

%matplotlib inline
data_dir = os.path.join('..','input')

paths_train = glob.glob(os.path.join(data_dir,'training-*','*.png'))
path_label_train = glob.glob(os.path.join(data_dir,'training-*.csv'))
paths_test = glob.glob(os.path.join(data_dir,'testing-*','*.png'))+glob.glob(os.path.join(data_dir,'testing-*','*.JPG'))
def get_key(path):

    # seperates the key of an image from the filepath

    key=path.split(sep=os.sep)[-1]

    return key
pal=0

for path_label in path_label_train:

    df = pd.read_csv(path_label)

    if pal==0: 

        df1 = df

    else:

        df1 = pd.concat([df1,df])

    pal=1    
def get_data(paths_img,path_labels=None,resize_dim=None):

    '''reads images from the filepaths, resizes them (if given), and returns them in a numpy array

    Args:

        paths_img: image filepaths

        path_label: pass image label filepaths while processing training data, defaults to None while processing testing data

        resize_dim: if given, the image is resized to resize_dim x resize_dim (optional)

    Returns:

        X: group of images

        y: categorical true labels

    '''

    X=[] # initialize empty list for resized images

    for i,path in enumerate(paths_img):

        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE) # images loaded in color (BGR)

        ret,img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        img = cv2.bilateralFilter(img,9,75,75)

        img = cv2.medianBlur(img,5)

        #img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

        #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # cnahging colorspace to GRAY

        if resize_dim is not None:

            img=cv2.resize(img,(resize_dim,resize_dim),interpolation=cv2.INTER_AREA) # resize image to 28x28

        #X.append(np.expand_dims(img,axis=2)) # expand image to 28x28x1 and append to the list.

        gaussian_3 = cv2.GaussianBlur(img, (9,9), 10.0) #unblur

        img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)

        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #filter

        img = cv2.filter2D(img, -1, kernel)

        thresh = 200

        maxValue = 255

        th, img = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY);

        ret,img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        x = img

        k = [[0 if x[i,j]==255 else 1 for j in range(x.shape[1])] for i in range(x.shape[0])]

        X.append(k) # expand image to 28x28x1 and append to the list

        # display progress

        if i==len(paths_img)-1:

            end='\n'

        else: end='\r'

        print('processed {}/{}'.format(i+1,len(paths_img)),end=end)

        

    X=np.array(X).astype('float32') # tranform list to numpy array

    if  path_labels is None:

        return X

    else:

        df = path_labels

        df=df.set_index('filename') 

        y_label=[df.loc[get_key(path)]['digit'] for path in  paths_img] # get the labels corresponding to the images

        return X, y_label
PIC_SIZE = 28

X_train,y_train=get_data(paths_train,df1,resize_dim=PIC_SIZE)
X_test = get_data(paths_test,resize_dim=PIC_SIZE)
plt.figure(figsize = (10, 8))

a, b = 9, 3

for i in range(27):

    plt.subplot(b, a, i+1)

    plt.imshow(X_train[i])

plt.show()
np.savez('NumthDB_training.npz', data=X_train, label=y_train)
np.savez('NumthDB_test.npz', data=X_test)
X_train = np.load('NumthDB_training.npz')
X_train.files
X_train['data'].shape
X_test = np.load('NumthDB_test.npz')
X_test.files