'''Importing the Moduls'''

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow 

print('Tensorflow version',tensorflow.__version__)



import matplotlib.pyplot as plt

import seaborn as sns

import cv2



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers, models



import os

print(os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images"))
!pip install split-folders
'''Train and Test split on a dataset with directories'''

import split_folders

orig_path = '../input/cell-images-for-detecting-malaria/cell_images/cell_images'

output_path = '../output'

split_folders.ratio(orig_path, output=output_path, seed=1, ratio=(.8, .2))
'''Preview the split'''

data_dir = '../output'

print(os.listdir(data_dir))
'''Creating train and test paths'''

train = data_dir+'/train/'

test = data_dir+'/val/'
'''Preview the train and test directories'''

print(os.listdir(train))

print('\n')

print(os.listdir(test))
'''Reading the Single Uninfected image'''

print('Uninfected image:',os.listdir(train+'Uninfected')[0])

print('Parasitizes image:',os.listdir(train+'Parasitized')[0])
uninf_cell = train+'Uninfected/'+'C230ThinF_IMG_20151112_150329_cell_162.png'

para_cell = train+'Parasitized/'+'C182P143NThinF_IMG_20151201_171950_cell_202.png'
"""Let's read the image file"""

cv2.imread(uninf_cell)
"""Let's see the shape of the image"""

cv2.imread(uninf_cell).shape
'''Preview the image'''

plt.figure(1, figsize = (15 , 7))

plt.subplot(1 , 2 , 1)

plt.imshow(cv2.imread(uninf_cell))

plt.title('Uninfected Cell')

plt.xticks([]) , plt.yticks([])



plt.subplot(1 , 2 , 2)

plt.imshow(cv2.imread(para_cell))

plt.title('Infected Cell')

plt.xticks([]) , plt.yticks([])



plt.show()
'''Number of image in the dataset'''

print('lenght of train parasitized', len(os.listdir(train+'Parasitized')))

print('lenght of train uninfected', len(os.listdir(train+'Uninfected')))

print('lenght of test parasitized', len(os.listdir(test+'Parasitized')))

print('lenght of test uninfected', len(os.listdir(test+'Uninfected')))
'''Creating the loop to get the dimension of the image'''

dim1 = []

dim2 = []



for image_filename in os.listdir(train+'Uninfected'):

    try:

        img = cv2.imread(train+'Uninfected/'+image_filename)

        d1,d2,colors = img.shape

        dim1.append(d1)

        dim2.append(d2)

        

    except AttributeError:

        print('')
'''Plot the distribution of images dimensions'''

plt.figure(figsize=(10,10))

sns.jointplot(dim1,dim2, color='teal',alpha=0.5)

plt.show()
'''Average of the dimensions'''

print(np.mean(dim1))

print(np.mean(dim2))
'''Final image shape that I will be feeding in my convolution network'''

image_shape = (130,130, 3)



# Then later on or actually preparing the data for the model we'll resize everything to these dimensions.
'''Create image generator'''

# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator



train_image_gen = ImageDataGenerator(rotation_range=30,

                               width_shift_range=0.1,

                               height_shift_range=0.1,

                               rescale=1./255,

                               shear_range=0.1,

                               zoom_range=0.1,

                               horizontal_flip=True,

                               fill_mode='nearest')



test_image_gen = ImageDataGenerator(rescale=1./255)
'''Preview the image'''

para_img = cv2.imread(para_cell)



plt.figure(1, figsize = (15 , 7))

plt.subplot(1 , 2 , 1)

plt.imshow(para_img)

plt.title('Before Processed Image')

plt.xticks([]) , plt.yticks([])



plt.subplot(1 , 2 , 2)

plt.imshow(train_image_gen.random_transform(para_img))

plt.title('After Processed Image')

plt.xticks([]) , plt.yticks([])



plt.show()  
'''Set the CNN model'''

model = models.Sequential()



model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.25))

          

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.25))



model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.25))



          

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping



early_stop = EarlyStopping(monitor='val_loss',patience=2)
'''Set up the generator to flow batches from directory'''

batch_size = 16



train_generator = train_image_gen.flow_from_directory(train,

                                                      target_size=image_shape[:2],

                                                      color_mode='rgb',

                                                      batch_size=batch_size,

                                                      class_mode='binary')



test_generator = test_image_gen.flow_from_directory(test,

                                                    target_size=image_shape[:2],

                                                    color_mode='rgb',

                                                    batch_size=batch_size,

                                                    class_mode='binary',

                                                    shuffle=False)
"""Let's see the target"""

train_generator.class_indices
'''Training the model'''

history = model.fit_generator(train_generator, epochs=20,

                              validation_data = test_generator,

                              callbacks=[early_stop])
'''Training and validation curves'''

fig, ax = plt.subplots(2,1, figsize = (8,8))

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)