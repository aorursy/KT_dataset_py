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
# IMPORTING LIBRARIES



import keras

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers import Dense

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator
# 3x3 CONVOLUTION LAYERS AND 2 DENSE LAYERS WITHOUT DROPOUT



# INITIALISING THE CNN



flower_cnn = Sequential()



# CONVOLUTION LAYER 1



flower_cnn.add(Conv2D(16,(3,3),input_shape = (32,32,3), activation = 'relu'))



# MAX POOLING LAYER 1



flower_cnn.add(MaxPooling2D(pool_size = (2,2)))



# CONVOLUTION LAYER 2



flower_cnn.add(Conv2D(8,(3,3), activation = 'relu'))



# MAX POOLING LAYER 1



flower_cnn.add(MaxPooling2D(pool_size=(2,2)))



# FLATTEN LAYER



flower_cnn.add(Flatten())



# INITIALISING ANN

# FULLY CONNECTED LAYERS



flower_cnn.add(Dense(units=512, kernel_initializer = 'uniform', activation = 'relu'))



flower_cnn.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))



# OUTPUT LAYER



flower_cnn.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'softmax'))



# COMPILING THE CNN



flower_cnn.compile(optimizer = RMSprop(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])



# FITTING THE MODEL



train_model = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)



test_model = ImageDataGenerator(rescale=1./255)



train_set = train_model.flow_from_directory('../input/flowers/Flowers/Training', target_size=(32,32), batch_size=32, class_mode='categorical')



test_set = test_model.flow_from_directory('../input/flowers/Flowers/Test', target_size=(32,32), batch_size=32, class_mode='categorical')



flower_cnn.fit_generator(train_set, steps_per_epoch=1071, epochs=10, validation_data=test_set, validation_steps=272)

# PREDICTING THE FLOWERS



from keras.preprocessing import image



img_set = ['image_0080', 'image_0160', 'image_0240', 'image_0320', 'image_0400', 'image_0480', 

           'image_0560', 'image_0640', 'image_0720', 'image_0800', 'image_0880', 'image_0960', 

           'image_1040', 'image_1120', 'image_1200', 'image_1280', 'image_1360']



for img in img_set:

    test_image = image.load_img('../input/flowers/Flowers/Validation/'+img+'.jpg', 

                                target_size = (32, 32))

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis = 0)

    result = flower_cnn.predict(test_image)

    #print(training_set.class_indices)

    

    if result[0][0] == 1:

        prediction = 'Flower 1'

        print(" The test image is")

        print(prediction)

    elif result[0][1] == 1:

        prediction = 'Flower 10'

        print(" The test image is")

        print(prediction)

    elif result[0][2] == 1:

        prediction = 'Flower 11'

        print(" The test image is")

        print(prediction)

    elif result[0][3] == 1:

        prediction = 'Flower 12'

        print(" The test image is")

        print(prediction)

    elif result[0][4] == 1:

        prediction = 'Flower 13'

        print(" The test image is")

        print(prediction)

    elif result[0][5] == 1:

        prediction = 'Flower 14'

        print(" The test image is")

        print(prediction)

    elif result[0][6] == 1:

        prediction = 'Flower 15'

        print(" The test image is")

        print(prediction)

    elif result[0][7] == 1:

        prediction = 'Flower 16'

        print(" The test image is")

        print(prediction)

    elif result[0][8] == 1:

        prediction = 'Flower 17'

        print(" The test image is")                            

        print(prediction)

    elif result[0][9] == 1:

        prediction = 'Flower 2'

        print(" The test image is")

        print(prediction)

    elif result[0][10] == 1:

        prediction = 'Flower 3'

        print(" The test image is")

        print(prediction)

    elif result[0][11] == 1:

        prediction = 'Flower 4'

        print(" The test image is")

        print(prediction)

    elif result[0][12] == 1:

        prediction = 'Flower 5'

        print(" The test image is")

        print(prediction)

    elif result[0][13] == 1:

        prediction = 'Flower 6'

        print(" The test image is")

        print(prediction)

    elif result[0][14] == 1:

        prediction = 'Flower 7'

        print(" The test image is")

        print(prediction)

    elif result[0][15] == 1:

        prediction = 'Flower 8'

        print(" The test image is")

        print(prediction)

    elif result[0][16] == 1:

        prediction = 'Flower 9'

        print(" The test image is")

        print(prediction)
# 3x3 CONVOLUTION LAYERS AND 2 DENSE LAYERS WITH DROPOUT



# INITIALISING THE CNN



flower_cnn1d = Sequential()



# CONVOLUTION LAYER 1



flower_cnn1d.add(Conv2D(16,(3,3),input_shape = (32,32,3), activation = 'relu'))



# MAX POOLING LAYER 1



flower_cnn1d.add(MaxPooling2D(pool_size = (2,2)))



# CONVOLUTION LAYER 2



flower_cnn1d.add(Conv2D(8,(3,3), activation = 'relu'))



# MAX POOLING LAYER 1



flower_cnn1d.add(MaxPooling2D(pool_size=(2,2)))



# DROPOUT LAYER



flower_cnn1d.add(Dropout(0.5))



# FLATTEN LAYER



flower_cnn1d.add(Flatten())



# INITIALISING ANN

# FULLY CONNECTED LAYERS



flower_cnn1d.add(Dense(units=512, kernel_initializer = 'uniform', activation = 'relu'))



flower_cnn1d.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))



# OUTPUT LAYER



flower_cnn1d.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'softmax'))



# COMPILING THE CNN



flower_cnn1d.compile(optimizer = RMSprop(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])



# FITTING THE MODEL



train_model = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)



test_model = ImageDataGenerator(rescale=1./255)



train_set = train_model.flow_from_directory('../input/flowers/Flowers/Training', target_size=(32,32), batch_size=32, class_mode='categorical')



test_set = test_model.flow_from_directory('../input/flowers/Flowers/Test', target_size=(32,32), batch_size=32, class_mode='categorical')



flower_cnn1d.fit_generator(train_set, steps_per_epoch=1071, epochs=10, validation_data=test_set, validation_steps=272)

# PREDICTING THE FLOWERS



from keras.preprocessing import image



img_set = ['image_0080', 'image_0160', 'image_0240', 'image_0320', 'image_0400', 'image_0480', 

           'image_0560', 'image_0640', 'image_0720', 'image_0800', 'image_0880', 'image_0960', 

           'image_1040', 'image_1120', 'image_1200', 'image_1280', 'image_1360']



for img in img_set:

    test_image = image.load_img('../input/flowers/Flowers/Validation/'+img+'.jpg', 

                                target_size = (32, 32))

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis = 0)

    result = flower_cnn1d.predict(test_image)

    #print(training_set.class_indices)

    

    if result[0][0] == 1:

        prediction = 'Flower 1'

        print(" The test image is")

        print(prediction)

    elif result[0][1] == 1:

        prediction = 'Flower 10'

        print(" The test image is")

        print(prediction)

    elif result[0][2] == 1:

        prediction = 'Flower 11'

        print(" The test image is")

        print(prediction)

    elif result[0][3] == 1:

        prediction = 'Flower 12'

        print(" The test image is")

        print(prediction)

    elif result[0][4] == 1:

        prediction = 'Flower 13'

        print(" The test image is")

        print(prediction)

    elif result[0][5] == 1:

        prediction = 'Flower 14'

        print(" The test image is")

        print(prediction)

    elif result[0][6] == 1:

        prediction = 'Flower 15'

        print(" The test image is")

        print(prediction)

    elif result[0][7] == 1:

        prediction = 'Flower 16'

        print(" The test image is")

        print(prediction)

    elif result[0][8] == 1:

        prediction = 'Flower 17'

        print(" The test image is")                            

        print(prediction)

    elif result[0][9] == 1:

        prediction = 'Flower 2'

        print(" The test image is")

        print(prediction)

    elif result[0][10] == 1:

        prediction = 'Flower 3'

        print(" The test image is")

        print(prediction)

    elif result[0][11] == 1:

        prediction = 'Flower 4'

        print(" The test image is")

        print(prediction)

    elif result[0][12] == 1:

        prediction = 'Flower 5'

        print(" The test image is")

        print(prediction)

    elif result[0][13] == 1:

        prediction = 'Flower 6'

        print(" The test image is")

        print(prediction)

    elif result[0][14] == 1:

        prediction = 'Flower 7'

        print(" The test image is")

        print(prediction)

    elif result[0][15] == 1:

        prediction = 'Flower 8'

        print(" The test image is")

        print(prediction)

    elif result[0][16] == 1:

        prediction = 'Flower 9'

        print(" The test image is")

        print(prediction)
# 5x5 CONVOLUTION LAYER AND 3 DENSE LAYERS WITHOUT DROPOUT



# INITIALISING THE CNN



flower_cnn2 = Sequential()



# CONVOLUTION LAYER 1



flower_cnn2.add(Conv2D(32,(5,5),input_shape = (32,32,3), activation = 'relu'))



# MAX POOLING LAYER 1



flower_cnn2.add(MaxPooling2D(pool_size = (2,2)))



# CONVOLUTION LAYER 2



flower_cnn2.add(Conv2D(16,(3,3), activation = 'relu'))



# MAX POOLING LAYER 1



flower_cnn2.add(MaxPooling2D(pool_size=(2,2)))



# FLATTEN LAYER



flower_cnn2.add(Flatten())



# INITIALISING ANN

# FULLY CONNECTED LAYERS



flower_cnn2.add(Dense(units=1024, kernel_initializer = 'uniform', activation = 'relu'))



flower_cnn2.add(Dense(units=512, kernel_initializer = 'uniform', activation = 'relu'))



flower_cnn2.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))



# OUTPUT LAYER



flower_cnn2.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'softmax'))



# COMPILING THE CNN



flower_cnn2.compile(optimizer = RMSprop(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])



# FITTING THE MODEL



train_model = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)



test_model = ImageDataGenerator(rescale=1./255)



train_set = train_model.flow_from_directory('../input/flowers/Flowers/Training', target_size=(32,32), batch_size=32, class_mode='categorical')



test_set = test_model.flow_from_directory('../input/flowers/Flowers/Test', target_size=(32,32), batch_size=32, class_mode='categorical')



flower_cnn2.fit_generator(train_set, steps_per_epoch=1071, epochs=10, validation_data=test_set, validation_steps=272)

# PREDICTING THE FLOWERS



from keras.preprocessing import image



img_set = ['image_0080', 'image_0160', 'image_0240', 'image_0320', 'image_0400', 'image_0480', 

           'image_0560', 'image_0640', 'image_0720', 'image_0800', 'image_0880', 'image_0960', 

           'image_1040', 'image_1120', 'image_1200', 'image_1280', 'image_1360']



for img in img_set:

    test_image = image.load_img('../input/flowers/Flowers/Validation/'+img+'.jpg', 

                                target_size = (32, 32))

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis = 0)

    result = flower_cnn2.predict(test_image)

    #print(training_set.class_indices)

    

    if result[0][0] == 1:

        prediction = 'Flower 1'

        print(" The test image is")

        print(prediction)

    elif result[0][1] == 1:

        prediction = 'Flower 10'

        print(" The test image is")

        print(prediction)

    elif result[0][2] == 1:

        prediction = 'Flower 11'

        print(" The test image is")

        print(prediction)

    elif result[0][3] == 1:

        prediction = 'Flower 12'

        print(" The test image is")

        print(prediction)

    elif result[0][4] == 1:

        prediction = 'Flower 13'

        print(" The test image is")

        print(prediction)

    elif result[0][5] == 1:

        prediction = 'Flower 14'

        print(" The test image is")

        print(prediction)

    elif result[0][6] == 1:

        prediction = 'Flower 15'

        print(" The test image is")

        print(prediction)

    elif result[0][7] == 1:

        prediction = 'Flower 16'

        print(" The test image is")

        print(prediction)

    elif result[0][8] == 1:

        prediction = 'Flower 17'

        print(" The test image is")                            

        print(prediction)

    elif result[0][9] == 1:

        prediction = 'Flower 2'

        print(" The test image is")

        print(prediction)

    elif result[0][10] == 1:

        prediction = 'Flower 3'

        print(" The test image is")

        print(prediction)

    elif result[0][11] == 1:

        prediction = 'Flower 4'

        print(" The test image is")

        print(prediction)

    elif result[0][12] == 1:

        prediction = 'Flower 5'

        print(" The test image is")

        print(prediction)

    elif result[0][13] == 1:

        prediction = 'Flower 6'

        print(" The test image is")

        print(prediction)

    elif result[0][14] == 1:

        prediction = 'Flower 7'

        print(" The test image is")

        print(prediction)

    elif result[0][15] == 1:

        prediction = 'Flower 8'

        print(" The test image is")

        print(prediction)

    elif result[0][16] == 1:

        prediction = 'Flower 9'

        print(" The test image is")

        print(prediction)
# 5x5 CONVOLUTION LAYER AND 3 DENSE LAYERS WITH DROPOUT



# INITIALISING THE CNN



flower_cnn2d = Sequential()



# CONVOLUTION LAYER 1



flower_cnn2d.add(Conv2D(32,(5,5),input_shape = (32,32,3), activation = 'relu'))



# MAX POOLING LAYER 1



flower_cnn2d.add(MaxPooling2D(pool_size = (2,2)))



# CONVOLUTION LAYER 2



flower_cnn2d.add(Conv2D(16,(3,3), activation = 'relu'))



# MAX POOLING LAYER 1



flower_cnn2d.add(MaxPooling2D(pool_size=(2,2)))



# DROPOUT LAYER



flower_cnn2d.add(Dropout(0.5))



# FLATTEN LAYER



flower_cnn2d.add(Flatten())



# INITIALISING ANN

# FULLY CONNECTED LAYERS



flower_cnn2d.add(Dense(units=1024, kernel_initializer = 'uniform', activation = 'relu'))



flower_cnn2d.add(Dense(units=512, kernel_initializer = 'uniform', activation = 'relu'))



flower_cnn2d.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))



# OUTPUT LAYER



flower_cnn2d.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'softmax'))



# COMPILING THE CNN



flower_cnn2d.compile(optimizer = RMSprop(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])



# FITTING THE MODEL



train_model = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)



test_model = ImageDataGenerator(rescale=1./255)



train_set = train_model.flow_from_directory('../input/flowers/Flowers/Training', target_size=(32,32), batch_size=32, class_mode='categorical')



test_set = test_model.flow_from_directory('../input/flowers/Flowers/Test', target_size=(32,32), batch_size=32, class_mode='categorical')



flower_cnn2d.fit_generator(train_set, steps_per_epoch=1071, epochs=10, validation_data=test_set, validation_steps=272)

# PREDICTING THE FLOWERS



from keras.preprocessing import image



img_set = ['image_0080', 'image_0160', 'image_0240', 'image_0320', 'image_0400', 'image_0480', 

           'image_0560', 'image_0640', 'image_0720', 'image_0800', 'image_0880', 'image_0960', 

           'image_1040', 'image_1120', 'image_1200', 'image_1280', 'image_1360']



for img in img_set:

    test_image = image.load_img('../input/flowers/Flowers/Validation/'+img+'.jpg', 

                                target_size = (32, 32))

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis = 0)

    result = flower_cnn2d.predict(test_image)

    #print(training_set.class_indices)

    

    if result[0][0] == 1:

        prediction = 'Flower 1'

        print(" The test image is")

        print(prediction)

    elif result[0][1] == 1:

        prediction = 'Flower 10'

        print(" The test image is")

        print(prediction)

    elif result[0][2] == 1:

        prediction = 'Flower 11'

        print(" The test image is")

        print(prediction)

    elif result[0][3] == 1:

        prediction = 'Flower 12'

        print(" The test image is")

        print(prediction)

    elif result[0][4] == 1:

        prediction = 'Flower 13'

        print(" The test image is")

        print(prediction)

    elif result[0][5] == 1:

        prediction = 'Flower 14'

        print(" The test image is")

        print(prediction)

    elif result[0][6] == 1:

        prediction = 'Flower 15'

        print(" The test image is")

        print(prediction)

    elif result[0][7] == 1:

        prediction = 'Flower 16'

        print(" The test image is")

        print(prediction)

    elif result[0][8] == 1:

        prediction = 'Flower 17'

        print(" The test image is")                            

        print(prediction)

    elif result[0][9] == 1:

        prediction = 'Flower 2'

        print(" The test image is")

        print(prediction)

    elif result[0][10] == 1:

        prediction = 'Flower 3'

        print(" The test image is")

        print(prediction)

    elif result[0][11] == 1:

        prediction = 'Flower 4'

        print(" The test image is")

        print(prediction)

    elif result[0][12] == 1:

        prediction = 'Flower 5'

        print(" The test image is")

        print(prediction)

    elif result[0][13] == 1:

        prediction = 'Flower 6'

        print(" The test image is")

        print(prediction)

    elif result[0][14] == 1:

        prediction = 'Flower 7'

        print(" The test image is")

        print(prediction)

    elif result[0][15] == 1:

        prediction = 'Flower 8'

        print(" The test image is")

        print(prediction)

    elif result[0][16] == 1:

        prediction = 'Flower 9'

        print(" The test image is")

        print(prediction)