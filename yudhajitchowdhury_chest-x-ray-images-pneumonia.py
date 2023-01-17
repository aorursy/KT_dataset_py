# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # forlinear algebra

import matplotlib.pyplot as plt #for plotting things

import os

from PIL import Image

print(os.listdir("../input"))



# Keras Libraries

import keras

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.metrics import classification_report, confusion_matrix
mainDIR = os.listdir('../input/chest-xray-pneumonia/chest_xray')

print(mainDIR)
train_folder= '../input/chest-xray-pneumonia/chest_xray/train/'

val_folder = '../input/chest-xray-pneumonia/chest_xray/val/'

test_folder = '../input/chest-xray-pneumonia/chest_xray/test/'
# train 

os.listdir(train_folder)

train_n = train_folder+'NORMAL/'

train_p = train_folder+'PNEUMONIA/'
#Normal pic 

print(len(os.listdir(train_n)))

rand_norm= np.random.randint(0,len(os.listdir(train_n)))

norm_pic = os.listdir(train_n)[rand_norm]

print('normal picture title: ',norm_pic)



norm_pic_address = train_n+norm_pic



#Pneumonia

rand_p = np.random.randint(0,len(os.listdir(train_p)))



sic_pic =  os.listdir(train_p)[rand_norm]

sic_address = train_p+sic_pic

print('pneumonia picture title:', sic_pic)



# Load the images

norm_load = Image.open(norm_pic_address)

sic_load = Image.open(sic_address)



#Let's plt these images

f = plt.figure(figsize= (10,6))

a1 = f.add_subplot(1,2,1)

img_plot = plt.imshow(norm_load)

a1.set_title('Normal')



a2 = f.add_subplot(1, 2, 2)

img_plot = plt.imshow(sic_load)

a2.set_title('Pneumonia')
# let's build the CNN model



cnn = Sequential()



#Convolution

cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))



#Pooling

cnn.add(MaxPooling2D(pool_size = (2, 2)))



# 2nd Convolution

cnn.add(Conv2D(32, (3, 3), activation="relu"))



# 2nd Pooling layer

cnn.add(MaxPooling2D(pool_size = (2, 2)))



# Flatten the layer

cnn.add(Flatten())



# Fully Connected Layers

cnn.add(Dense(activation = 'relu', units = 128))

cnn.add(Dense(activation = 'sigmoid', units = 1))



# Compile the Neural network

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
num_of_test_samples = 600

batch_size = 32
# Fitting the CNN to the images

# The function ImageDataGenerator augments your image by iterating through image as your CNN is getting ready to process that image



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.



training_set = train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/train',

                                                 target_size = (64, 64),

                                                 batch_size = 32,

                                                 class_mode = 'binary')



validation_generator = test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/val/',

    target_size=(64, 64),

    batch_size=32,

    class_mode='binary')



test_set = test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/test',

                                            target_size = (64, 64),

                                            batch_size = 32,

                                            class_mode = 'binary')
cnn.summary()
cnn_model = cnn.fit_generator(training_set,

                         steps_per_epoch = 163,

                         epochs = 1,

                         validation_data = validation_generator,

                         validation_steps = 624)
test_accu = cnn.evaluate_generator(test_set,steps=624)
print('The testing accuracy is :',test_accu[1]*100, '%')