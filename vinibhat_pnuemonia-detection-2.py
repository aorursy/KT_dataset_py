# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import keras

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/"))








my_cnn=Sequential()

my_cnn.add(Conv2D(32,(5,5), input_shape=(160,160,3),activation='relu'))



# maxpooling

my_cnn.add(MaxPooling2D(pool_size=(2,2)))





#adding one more convolution layer.. 

my_cnn.add(Conv2D(16,(3,3), activation='relu'))





my_cnn.add(MaxPooling2D(pool_size=(2,2))) # second layer we dont need to give input size since it uses 1st layer output as input





my_cnn.add(Flatten())



#Fully connected layer1

my_cnn.add(Dense(units=1024,activation='relu'))





#Fully connected layer2



my_cnn.add(Dense(units=512,activation='relu'))





#Fully connected layer3



my_cnn.add(Dense(units=128,activation='relu'))





my_cnn.add(Dense(units=1,activation='sigmoid'))







# compiling

my_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])





checkpoint = ModelCheckpoint("pnemonia1.h5",

                             monitor="val_loss",

                             mode="min",

                             save_best_only = True,

                             verbose=1)



earlystop = EarlyStopping(monitor = 'val_loss', 

                          min_delta = 0, 

                          patience = 5,

                          verbose = 1,

                          restore_best_weights = True)

callbacks = [earlystop, checkpoint]





from keras.preprocessing.image import ImageDataGenerator



# creating train and test model

train_model=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)



# for testing the above experience is not needed hence we dont perform flip zoom...

test_model=ImageDataGenerator(rescale=1./255)





# create train set and test set

# mode is binary since only 2 clasess in that folder

train_set=train_model.flow_from_directory('../input/chest_xray/chest_xray/train', target_size=(160,160), batch_size=32,class_mode='binary')

test_set=test_model.flow_from_directory('../input/chest_xray/chest_xray/test', target_size=(160,160), batch_size=8,class_mode='binary')

my_cnn.fit_generator(train_set,steps_per_epoch=5216//32, epochs=15, callbacks = callbacks, validation_data=test_set,validation_steps=624//32)



#Predict for a new image. it is in the folder val from our model.



train_model=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)



train_set=train_model.flow_from_directory('../input/chest_xray/chest_xray/train', target_size=(160,160), batch_size=32,class_mode='binary')



from keras.models import load_model

mymodel=load_model('./pnemonia1.h5')



from keras.preprocessing import image

test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1442-0001.jpeg',target_size=(160,160))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = mymodel.predict(test_image)

#looking at the indices assign values for prediction(result[][])



train_set.class_indices



if result[0][0] == 1:

    prediction = 'Pneumonia'

    print(" The test image is")

    print(prediction)

else:

    prediction = 'Normal'

    print(" The test image is")

    print(prediction)