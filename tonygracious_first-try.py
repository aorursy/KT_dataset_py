# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import os

import cv2
import glob
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import color, exposure
import time
drivers = pd.read_csv('../input/driver_imgs_list.csv')
train_files = [f for f in glob.glob("../input/train/*/*.jpg")]
test_files = ["../input/test/" + f for f in os.listdir("../input/test/")]
print(train_files[:10])
print(test_files[:10])
print('Shape of the image',cv2.imread(train_files[0]).shape)

import random
fi = random.choice(train_files)
print(fi)
im = cv2.imread(fi)
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')





lbl = {'c0' : 'safe driving', 
'c1' : 'texting - right', 
'c2' : 'talking on the phone - right', 
'c3' : 'texting - left', 
'c4' : 'talking on the phone - left', 
'c5' : 'operating the radio', 
'c6' : 'drinking', 
'c7' : 'reaching behind', 
'c8' : 'hair and makeup', 
'c9' : 'talking to passenger'}

plt.rcParams['figure.figsize'] = (8.0, 20.0)
plt.subplots_adjust(wspace=0, hspace=0)
i_ = 0
for l in lbl:
    tf = ["../input/train/" + l + "/" + f for f in os.listdir("../input/train/" + l + "/")]
    fi = random.choice(tf)
    print(fi)
    im = cv2.imread(fi)
    plt.subplot(5, 2, i_+1).set_title(lbl[l])
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 1

from sklearn.feature_extraction.image import extract_patches_2d
i = random.choice(train_files)
print(fi)
im = cv2.imread(fi)
im.shape
im= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
p=extract_patches_2d(im,(200, 200),max_patches=20,
                                  random_state=0)
for i in range(0,20):
    plt.subplot(10, 2, i+1).set_title('patches_'+str(i))
    plt.imshow(p[i,:,:],cmap='Greys_r'); plt.axis('off')
    
def feature_extraction(filename) :
    im = cv2.imread(filename)
    im= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    p=extract_patches_2d(im,(200, 200),max_patches=20,
                                  random_state=0)
    feature_vec=np.zeros((0))
    for i in range(0,20):
        f=np.histogram(hog(p[i,:,:]), bins=50, density=True)[0]
        feature_vec=np.hstack((feature_vec,  f))
        
   
    return feature_vec


    
    
    
start=time.time()
features=np.zeros((len(train_files), 1000))
Labels=[]
for i in range(len(train_files)):
    features[i,:] =feature_extraction(train_files[i])
    Labels.append(train_files[0][15:17])
    if ((i+1)%100)==0 :
        end=time.time()
        break
print(end-start)
    
len(train_files)* 45/(100*60*60)
