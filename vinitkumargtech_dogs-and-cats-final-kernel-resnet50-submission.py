# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFile 
import numpy as np
import os 
import cv2
from tqdm import tqdm_notebook
from random import shuffle
import pandas as pd
import random
from tqdm import tqdm
import seaborn as sns
import math
import keras
from keras import applications
from keras import optimizers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import Callback,ModelCheckpoint
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.models import load_model
from keras.preprocessing.image import load_img
batch_size = 32
epochs = 20
num_classes = 2
num_t_samples = 20000
num_v_samples = 5000
path='../input/dogs-vs-cats-redux-kernels-edition/'
dir= '../input/dogscats/data/data/'
train_data_path = path+'train/'
test_data_path =  path+'test/'
train_data_dir = dir+'train/'
validation_data_dir=dir+'validation/'
test_dir='../input/dogscatstest/test1/test1/'
img_size=224
#Process training data to make it ready for fitting.
train_datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_size, img_size),
                                                    batch_size=batch_size,class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir,target_size=(img_size, img_size),batch_size=batch_size,
                                                  class_mode='categorical',shuffle=False)
filename=test_generator.filenames
model=load_model('../input/resnet50-all/resnet50_model.h5')
test_generator.reset()
pred=model.predict_generator(test_generator,steps=math.ceil(test_generator.samples/test_generator.batch_size),verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
new_preds=[]
for i in range(len(predictions)):
    if predictions[i]=='dogs':
        new_preds.append('dog')
    else:
        new_preds.append('cat')
predicted_class_indices
def display_testdata(testdata,filenames):
    f, ax = plt.subplots(5,5, figsize=(15,15))
    i=0
    for a,b in zip(testdata,filenames):
        pred_label=a
        fname=b
        title = 'Prediction :{}'.format(pred_label)   
        original = load_img('{}/{}'.format(test_dir,fname))
        ax[i//5,i%5].axis('off')
        ax[i//5,i%5].set_title(title)
        ax[i//5,i%5].imshow(original)
        i=i+1
    plt.show()
display_testdata(new_preds[11100:11125],filename[11100:11125])
def create_submission():
    labels=[]
    file_index=[]
    for a,b in zip(predicted_class_indices,filename):
        pred_label=a
        fname=b[5:]
        fname=fname.split('.')[0]
        labels.append(pred_label)
        file_index.append(fname)
    results=pd.DataFrame({"id":file_index,
                      "label":labels})
    results.to_csv("submission.csv",index=False)
create_submission()