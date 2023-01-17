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
# Importing packages



import keras

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.optimizers import Adam
# Initializing the CNN

classifier = Sequential()



# Convolution

classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))



# Pooling

classifier.add(MaxPooling2D(pool_size=(2,2)))



# Adding a second Convolution layer

classifier.add(Conv2D(32,(3,3),activation='relu'))



# Adding a second Pooling layer

classifier.add(MaxPooling2D(pool_size=(2,2)))



# Flattening

classifier.add(Flatten())



# Full connection

classifier.add(Dense(units=128,activation='relu'))



# Output layer

classifier.add(Dense(units=1,activation='sigmoid'))



# Compiling the CNN

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator





train_model=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)



test_model=ImageDataGenerator(rescale=1./255)





train_set = train_model.flow_from_directory('../input/chest_xray/chest_xray/train',target_size=(64,64), batch_size=32, class_mode='binary')



val_gen = test_model.flow_from_directory('../input/chest_xray/chest_xray/val', target_size=(64, 64), batch_size=32, class_mode='binary')



test_set = test_model.flow_from_directory('../input/chest_xray/chest_xray/test',target_size=(64,64), batch_size=32, class_mode='binary')
classifier.summary()



classifier.fit_generator(train_set, steps_per_epoch=5216/32, epochs=10, validation_data = val_gen, validation_steps=624/32)
# Plotting the Accuracy and Loss



import matplotlib.pyplot as plt # for plotting graphs



#Accuracy

plt.plot(classifier.history.history['acc'])

plt.plot(classifier.history.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training_set', 'Validation_set'], loc='upper left')

plt.show()



# Loss 

plt.plot(classifier.history.history['val_loss'])

plt.plot(classifier.history.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Test set'], loc='upper left')

plt.show()
# Test accuracy



test_accu = classifier.evaluate_generator(test_set,steps=624)



print('The testing accuracy is :', test_accu[1]*100, '%')
# Prediction



from keras.preprocessing import image

test_image = image.load_img('../input/chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg', target_size = (64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)

print(result)

train_set.class_indices

print(train_set.class_indices)

if result[0][0] == 0:

    prediction = 'Normal'

    print(" The test image is")

    print(prediction)

else:

    prediction = 'Pneumonia'

    print(" The test image is")

    print(prediction)