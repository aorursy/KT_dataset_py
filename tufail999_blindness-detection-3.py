# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading CSV file

train_valid_data = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv', dtype=str)

test_data = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/test.csv', dtype=str)



def append_ext(fn):

    return fn+".png"

train_valid_data["id_code"]=train_valid_data["id_code"].apply(append_ext)

test_data["id_code"]=test_data["id_code"].apply(append_ext)

#splitting data set into traning and test keeping images in folder



train_data = pd.DataFrame(train_valid_data.iloc[ 0:2930 , :].values)

train_data.columns = ['filename' , 'class']

valid_data = pd.DataFrame(train_valid_data.iloc[ 2930:3662, :].values)

valid_data.columns = ['filename' , 'class']

#Building CNN



from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense



classifier = Sequential()





classifier.add(Conv2D(64, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))#Note- few values will be changed for black and white images



# Step 2 - Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Adding a second convolutional layer

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(units = 256, activation = 'relu'))

classifier.add(Dense(units = 5, activation = 'softmax'))#Note- sigmoid function will be changed if output class is more than 2 # here i used softmax because the outcome class is more than 2



# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])#binary_crossentropy will be changed if o/p class is more than 2 #used categorical crossentropy as more than 2 class is there

"""

# Part 2 - Fitting the CNN to the images

import keras

keras.utils.np_utils.to_categorical

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow_from_dataframe(

        dataframe=train_data,

        directory='/kaggle/input/aptos2019-blindness-detection/train_images',

        x_col="filename",

        y_col="class",

        target_size=(150, 150),

        batch_size=32,

        class_mode='categorical')



validation_generator = test_datagen.flow_from_dataframe(

        dataframe=valid_data,

        directory='/kaggle/input/aptos2019-blindness-detection/train_images',

        x_col="filename",

        y_col="class",

        target_size=(150, 150),

        batch_size=32,

        class_mode='categorical')



"""



#training

"""

classifier.fit_generator(

        train_generator,

        steps_per_epoch=1000,

        epochs=10,

        validation_data=validation_generator,

        validation_steps=500)

"""
#loading trnined modoel

from keras.models import load_model

classifier = load_model('/kaggle/input/trained-model-02/code_without_splitting_folder_and_train_test.h5')
#making predictions



from keras.preprocessing import image as image_utils

images = []

for root, dirs, files in os.walk('/kaggle/input/aptos2019-blindness-detection/test_images'):

    for filename in files:

        img = os.path.join(root, filename)

        img = image_utils.load_img(img, target_size=(150, 150))

        img = image_utils.img_to_array(img)

        img = np.expand_dims(img, axis=0)

        images.append(img)

 

# stack up images list to pass for prediction

images = np.vstack(images)

classes = classifier.predict_classes(images, batch_size=10)

print(classes)

################################making dataframe with the help of lists



final_output = pd.DataFrame(np.column_stack([files, classes]), 

                               columns=['id_code', 'diagnosis'])



final_output.to_csv('submission.csv',index=False)
'''#saving the model

classifier.save("code_without_splitting_folder_and_train_test.h5")



classifier.save("code_without_splitting_folder_and_train_test")

'''