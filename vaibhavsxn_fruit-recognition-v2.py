# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.image import imread

%matplotlib inline

import os
my_data_dir = '/kaggle/input/fruit-recognition/'
apple = my_data_dir+'Apple/'+'Total Number of Apples'

peach = my_data_dir+'Peach'

carambola = my_data_dir+'Carambola'

kiwi = my_data_dir+'Kiwi/'+'Total Number of Kiwi fruit'

tomatoes = my_data_dir+'Tomatoes'

persimmon = my_data_dir+'Persimmon'

plum = my_data_dir+'Plum'

guava = my_data_dir+'Guava/'+'guava total final'

pear = my_data_dir+'Pear'

mango = my_data_dir+'Mango'

muskmelon = my_data_dir+'muskmelon'

banana = my_data_dir+'Banana'

pomegranate = my_data_dir+'Pomegranate'

pitaya = my_data_dir+'Pitaya'

orange = my_data_dir+'Orange'
dim1 = []

dim2 = []

for image_filename in os.listdir(apple):

    

    img = imread(apple+'/'+image_filename)

    d1,d2,colors = img.shape

    dim1.append(d1)

    dim2.append(d2)
image_shape = (283,383,3)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees

                               width_shift_range=0.10, # Shift the pic width by a max of 5%

                               height_shift_range=0.10, # Shift the pic height by a max of 5%

                               rescale=1/255, # Rescale the image by normalzing it.

                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)

                               zoom_range=0.1, # Zoom in by 10% max

                               horizontal_flip=True, # Allo horizontal flipping

                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value

                              )
image_gen.flow_from_directory(my_data_dir)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
pwd
model = load_model('/kaggle/input/trained/fruit.h5')
from tensorflow.keras.preprocessing import image
carambola1 = carambola+'/'+os.listdir(carambola)[15]
my_image = image.load_img(carambola1,target_size=image_shape)
my_image
my_image = image.img_to_array(my_image)
my_image = np.expand_dims(my_image, axis=0)
model.predict(my_image)
pomegranate1 = pomegranate+'/'+os.listdir(pomegranate)[15]

my_image = image.load_img(pomegranate1,target_size=image_shape)

my_image
my_image = image.img_to_array(my_image)

my_image = np.expand_dims(my_image, axis=0)

model.predict(my_image)