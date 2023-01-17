import pandas as pd

import cv2

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import os

import gc

import random
# import data

train_dir = '../input/train/'

test_dir = '../input/test/'



train_dogs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i] # get dog images

train_cats = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]  # get cat images



test_imgs = ['../input/test/{}'.format(i) for i in os.listdir(test_dir)] # get test images

train_imgs = train_dogs[:2000] + train_cats[:2000] # slice the dataset and use 2000 in each class



random.shuffle(train_imgs) # suffle it randomly



# clear list that are useless

del train_dogs

del train_cats

gc.collect()
import matplotlib.image as mpimg

for ima in train_imgs[0:3]:

    img = mpimg.imread(ima)

    plt.imshow(img)

    plt.show()
# image dimensions and colour channels

nrows = 150

ncols = 150

channel = 3

# A function to read and process the images to an acceptable format for our model

def read_and_process_image(list_of_images):

    """

    Returns two arrays:

        X is an array of resized images

        y is an array of labels

    """

    X = []

    y = []



    for img in list_of_images:

        try:

            X.append(cv2.resize(cv2.cvtColor(cv2.imread(img, 1), cv2.COLOR_BGR2RGB), (nrows,ncols), interpolation = cv2.INTER_CUBIC)) # read the images

            # get the labels  classify dog as 1 and cat as 0

            if 'dog' in img:

                y.append(1)

            elif 'cat' in img:

                y.append(0)

        except Exception as e:

            print('Exception raised as',e)

            

    return X,y 
X, y = read_and_process_image(train_imgs)
plt.figure(figsize=(20,10))

columns = 5

for i in range(columns):

    plt.subplot(1,columns,i+1)

    plt.imshow(X[i])

    plt.title('dog' if y[i] else 'cat')
del train_imgs

gc.collect()
import seaborn as sns

# convert list to numpy array

X = np.array(X)

y = np.array(y)



sns.countplot(y)

plt.title('count of dog and cat classes')

plt.show()
print(X.shape, y.shape)
# split the data into train and test set

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)



print('Shape of train images is', X_train.shape)

print('Shape of validation images is', X_val.shape)

print('Shape of train labels', y_train.shape)

print('Shape of validation labels', y_val.shape)

del X

del y

gc.collect()
# get the length of the train and validation set

ntrain = len(X_train)

nval = len(X_val)



batch_size = 32 # factor of 2,8,16,32,64,128 so that values can be stored in cache memory.
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255)

val_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow(X_train, y_train, batch_size=30)

val_generator = val_datagen.flow(X_val, y_val, batch_size=30)
for data_batch, labels_batch in val_generator:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break
plt.imshow(data_batch[3])

plt.show()
labels_batch[3]
from keras import layers 

from keras import models



model = models.Sequential() 

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2))) 

model.add(layers.Conv2D(64, (3, 3), activation='relu')) 

model.add(layers.MaxPooling2D((2, 2))) 

model.add(layers.Conv2D(128, (3, 3), activation='relu')) 

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten()) 

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
from keras import optimizers



model.compile(loss='binary_crossentropy',



optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, 

                              validation_data=val_generator, validation_steps=50)
import matplotlib.pyplot as plt



acc = history.history['acc'] 

val_acc = history.history['val_acc'] 

loss = history.history['loss'] 

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label='Training acc') 

plt.plot(epochs, val_acc, 'b', label='Validation acc') 

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss') 

plt.legend()



plt.show()
datagen = ImageDataGenerator(rotation_range=40, 

                             width_shift_range=0.2,

                             height_shift_range=0.2, 

                             shear_range=0.2,

                             zoom_range=0.2,

                             horizontal_flip=True, 

                             fill_mode='nearest'

                            )
# train_generator = train_datagen.flow(X_train, y_train, batch_size=30)

plt.imshow(X_train[0])

plt.show()
from keras.preprocessing import image

x = X_train[0]

x = x.reshape((1,) + x.shape)

i = 0

for batch in datagen.flow(x, batch_size=1):

    plt.figure(i) 

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0:

        break
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu')) 

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu')) 

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten()) 

model.add(layers.Dropout(0.5)) 

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',



optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
# Use data augmentation

train_datagen = ImageDataGenerator( rescale=1./255,

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2, 

                                   zoom_range=0.2, 

                                   horizontal_flip=True,

                                  )



val_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow(X_train, y_train, batch_size=30)

val_generator = val_datagen.flow(X_val, y_val, batch_size=30)
history = model.fit_generator(train_generator, steps_per_epoch=100,

                              epochs=20, validation_data=val_generator,

                              validation_steps=50)
model.save('./cat_dog_model.h5')
import matplotlib.pyplot as plt



acc = history.history['acc'] 

val_acc = history.history['val_acc'] 

loss = history.history['loss'] 

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label='Training acc') 

plt.plot(epochs, val_acc, 'b', label='Validation acc') 

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss') 

plt.legend()



plt.show()
from keras.applications import VGG16



conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.summary()
from keras import models

from keras import layers



model = models.Sequential() 

model.add(conv_base) 

model.add(layers.Flatten()) 

model.add(layers.Dense(256, activation='relu')) 

model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers



train_datagen = ImageDataGenerator( rescale=1./255,

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True,

                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow(X_train, y_train, batch_size=20)

val_generator = val_datagen.flow(X_val, y_val, batch_size=20)
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])



history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, 

                              validation_data=val_generator, validation_steps=50)
del X_train

del X_val
import matplotlib.pyplot as plt



acc = history.history['acc'] 

val_acc = history.history['val_acc'] 

loss = history.history['loss'] 

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label='Training acc') 

plt.plot(epochs, val_acc, 'b', label='Validation acc') 

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss') 

plt.legend()



plt.show()
# # import the modules we'll need

# from IPython.display import HTML

# import pandas as pd

# import numpy as np

# import base64



# # function that takes in a dataframe and creates a text link to  

# # download it (will only work for files < 2MB or so)

# def create_download_link(df, title = "solution", filename = "data.csv"):  

#     csv = df.to_csv()

#     b64 = base64.b64encode(csv.encode())

#     payload = b64.decode()

#     html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

#     html = html.format(payload=payload,title=title,filename=filename)

#     return HTML(html)



# # create a random sample dataframe

# # create a link to download the dataframe

# create_download_link(solution)
from keras.models import load_model



model = load_model('./cat_dog_model.h5')



model.summary()
from keras import models



layer_outputs = [layer.output for layer in model.layers[:8]]

activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
from keras.preprocessing import image

x = X_train[40]

x = x.reshape((1,) + x.shape)

i = 0
import matplotlib.pyplot as plt



plt.imshow(x[0])

plt.show()
activations = activation_model.predict(x)
first_layer_activation = activations[0]



print(first_layer_activation.shape)
import matplotlib.pyplot as plt



plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
layer_names = [] 

for layer in model.layers[:4]:

    layer_names.append(layer.name)

images_per_row = 16



for layer_name, layer_activation in zip(layer_names, activations):

    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row

    display_grid = np.zeros((size * n_cols, images_per_row * size))



    for col in range(n_cols):

        for row in range(images_per_row):

            channel_image = layer_activation[0, :, :, col * images_per_row + row] 

            channel_image -= channel_image.mean() 

            channel_image /= channel_image.std()

            channel_image *= 64 

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image



    scale = 1. / size 

    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))

    plt.title(layer_name) 

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')