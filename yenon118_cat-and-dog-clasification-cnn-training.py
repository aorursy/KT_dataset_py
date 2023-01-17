# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import tqdm



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras import preprocessing, layers, models, optimizers

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.optimizers import RMSprop, SGD, Adam

from keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!ls ../input/test_set/test_set

!ls ../input/training_set/training_set
path_cats = []

train_path_cats = '../input/training_set/training_set/cats'

for path in os.listdir(train_path_cats):

    if '.jpg' in path:

        path_cats.append(os.path.join(train_path_cats, path))

path_dogs = []

train_path_dogs = '../input/training_set/training_set/dogs'

for path in os.listdir(train_path_dogs):

    if '.jpg' in path:

        path_dogs.append(os.path.join(train_path_dogs, path))

        

print(len(path_cats))

print(len(path_dogs))



# n = 6000

n = 4000



training_set = np.zeros((n, 150, 150, 3), dtype='float32')

for i in range(n):

    if i < int(n/2):

        path = path_dogs[i]

        img = preprocessing.image.load_img(path, target_size=(150, 150))

        training_set[i] = preprocessing.image.img_to_array(img)

    else:

        path = path_cats[i - int(n/2)]

        img = preprocessing.image.load_img(path, target_size=(150, 150))

        training_set[i] = preprocessing.image.img_to_array(img)
x_training_set = training_set

x_training_set = x_training_set / 255





# y_cat = [1,0]

# y_dog = [0,1]



# y_training_set = np.zeros((n, 2), dtype='float32')

# for i in range(n):

#   if i < int(n/2):

#     y_training_set[i] = y_cat

#   else:

#     y_training_set[i] = y_dog



y_training_set = np.zeros((n), dtype='float32')

for i in range(n):

  if i < int(n/2):

    # dog is 0

    y_training_set[i] = 0

  else:

    # Cat is 1

    y_training_set[i] = 1





print(x_training_set.shape)

print(y_training_set.shape)



# Set the random seed

random_seed = 2
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(x_training_set, y_training_set, test_size = 0.1, random_state=random_seed, shuffle = True)



print(X_train.shape)

print(X_val.shape)

print(Y_train.shape)

print(Y_val.shape)

print("\n\n")





# Some training examples

x = 2

y = 5

fig, axs = plt.subplots(x, y, sharex=True, sharey=True)

for i in range(x*y):

  if i < y:

    axs[0, i%y].imshow(X_train[i])

    axs[0, i%y].set_title(Y_train[i])

  elif i >= y and i < (x*y):

    axs[1, i%y].imshow(X_train[i])

    axs[1, i%y].set_title(Y_train[i])



plt.show()
plt.clf()



# Set the CNN model 

# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (3,3), activation ='relu', input_shape = (150, 150, 3)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(0.2))





model.add(Conv2D(filters = 128, kernel_size = (3,3), activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 128, kernel_size = (3,3), activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(0.2))





model.add(Flatten())

model.add(Dense(512, activation = "relu"))

# model.add(Dropout(0.2))

model.add(Dense(1, activation = "sigmoid"))

model.summary()

# Define the optimizer



# optimizer = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)

# optimizer = SGD(lr=0.0001)



optimizer = optimizers.Adam(lr=1e-3)
# Compile the model



# model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# model.compile(optimizer = 'Adam', loss = "binary_crossentropy", metrics=['accuracy'])



model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics=['accuracy'])
epochs = 30

batch_size = 32

steps_per_epoch = int((n / batch_size) * 2.5)



# Use data augmentation

train_datagen = preprocessing.image.ImageDataGenerator(

#     rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

)



train_generator = train_datagen.flow(

    X_train,

    Y_train,

    batch_size = batch_size)



# do not augment validation data

test_datagen = preprocessing.image.ImageDataGenerator(

#     rescale=1./255

)



validation_generator = test_datagen.flow(

    X_val,

    Y_val,

    batch_size = batch_size

)



# train

history = model.fit_generator(

    train_generator,

    steps_per_epoch = steps_per_epoch,

    epochs= epochs,

    validation_data = validation_generator,

    validation_steps = steps_per_epoch

)

# epochs = 30

# batch_size = 32



# history = model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=epochs, validation_data = (X_val,Y_val), verbose = 1)

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
path_cats = []

test_path_cats = '../input/test_set/test_set/cats'

for path in os.listdir(test_path_cats):

    if '.jpg' in path:

        path_cats.append(os.path.join(test_path_cats, path))

path_dogs = []

test_path_dogs = '../input/test_set/test_set/dogs'

for path in os.listdir(test_path_dogs):

    if '.jpg' in path:

        path_dogs.append(os.path.join(test_path_dogs, path))

        



print(len(path_cats))

print(len(path_dogs))



n = 200



x_test = np.zeros((n, 150, 150, 3), dtype='float32')

y_test = np.zeros((n), dtype='float32')

for i in range(n):

    if i < int(n/2):

        path = path_dogs[i]

        img = preprocessing.image.load_img(path, target_size=(150, 150))

        x_test[i] = preprocessing.image.img_to_array(img)

        y_test[i] = 0

    else:

        path = path_cats[i - int(n/2)]

        img = preprocessing.image.load_img(path, target_size=(150, 150))

        x_test[i] = preprocessing.image.img_to_array(img)

        y_test[i] = 1



print(x_test.shape)

print(y_test.shape)

x_test = x_test / 255.0



# Some test examples

x = 2

y = 5

fig, axs = plt.subplots(x, y, sharex=True, sharey=True)

for i in range(x*y):

  if i < y:

    axs[0, i%y].imshow(x_test[i])

    axs[0, i%y].set_title(y_test[i])

  elif i >= y and i < (x*y):

    axs[1, i%y].imshow(x_test[i])

    axs[1, i%y].set_title(y_test[i])



plt.show()
plt.clf()





results = model.predict(x_test)



# print(results)



y_results = np.zeros((n), dtype='float32')

for i in range(len(results)):

  if results[i][0] < 0.5:

    y_results[i] = 0

  else:

    y_results[i] = 1



print(y_test)

print(y_results)

n_sum = 0

for i in range(n):

  if y_results[i] != y_test[i]:

      n_sum = n_sum + 1



print(str(n_sum) + " out of " + str(n) + " predicted wrong \n")

print(str(100*n_sum/n) + " percent not predicted accurately \n")



# Some false prediction examples

x = 0

y = 6

if(n_sum%y == 0):

    x = int(n_sum/y)

else:

    x = int(n_sum/y)+1

fig, axs = plt.subplots(x, y, sharex=True, sharey=True)

i = 0

for j in range(n):

  if y_test[j] != y_results[j]:

    axs[int(i/y), i%y].imshow(x_test[j])

#     axs[int(i/y), i%y].set_title(y_test[j])

    i += 1



plt.show()