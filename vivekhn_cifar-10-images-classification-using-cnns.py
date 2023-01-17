# loading the required libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn 
# install keras neural-network library

!pip install keras
# loading the dataset 

from keras.datasets import cifar10

(X_train, y_train) , (X_test, y_test) = cifar10.load_data()
# checking the shape

print('X_train shape - ',X_train.shape)

print('X_test shape - ',X_test.shape)

print('y_train shape - ',y_train.shape)

print('y_test shape - ',y_test.shape)

# visualizing few samples



labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']



W_grid = 10

L_grid = 10



fig, axes = plt.subplots(L_grid, W_grid, figsize = (25, 25))

axes = axes.ravel()



n_training = len(X_train)



for i in range(0,L_grid*W_grid):

    index = np.random.randint(0,n_training) # pick a random number

    axes[i].imshow(X_train[index])

    index = y_train[index]

    axes[i].set_title(labels[int(index)])

    axes[i].axis('off')

    

plt.subplots_adjust(hspace = 0.4)
# converting the values into float and normalizing

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train = X_train/255

X_test = X_test/255

# encoding the Predictor variable

import keras

y_train = keras.utils.to_categorical(y_train, 10)

y_test = keras.utils.to_categorical(y_test, 10)
# Storing the input shape

Input_shape = X_train.shape[1:]

Input_shape
import keras

from keras.models import Sequential

from keras.layers import Conv2D,Dense,AveragePooling2D,MaxPooling2D, Flatten, Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard
cnn_model = Sequential()

cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = Input_shape))

cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))

cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Dropout(0.4))





cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))

cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))

cnn_model.add(AveragePooling2D(2,2))

cnn_model.add(Dropout(0.4))



cnn_model.add(Flatten())



cnn_model.add(Dense(units = 1024, activation = 'relu'))



cnn_model.add(Dense(units = 1024, activation = 'relu'))



cnn_model.add(Dense(units = 10, activation = 'softmax'))
cnn_model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(lr = 0.001), metrics = ['accuracy'])
cnn_model.summary()
# fitting the train data into the model

history = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 10, shuffle = True,validation_data=(X_test,y_test))
# Evaluating the model performance

evaluation = cnn_model.evaluate(X_test, y_test)

print('Test Accuracy: {}'.format(evaluation[1]))
predicted_classes = cnn_model.predict_classes(X_test) 

predicted_classes
y_test = y_test.argmax(1)

y_test
y_test 
# plotting Confusion Matrix

from sklearn.metrics import confusion_matrix

import seaborn as sns



cm = confusion_matrix(y_test, predicted_classes)

cm

plt.figure(figsize = (10, 10))

sns.heatmap(cm, annot = True)
# generating more data using the existing data

from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

                            #width_shift_range = 0.1,

                            horizontal_flip = True,

                             )

datagen.fit(X_train)
# retraining the model



cnn_model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32), epochs = 5)
# encoding the Predictor variable

y_test = keras.utils.to_categorical(y_test, 10)
# printing the accuracy

score = cnn_model.evaluate(X_test, y_test)

print('Test accuracy', score[1])