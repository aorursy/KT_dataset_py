import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau





sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

submit = pd.read_csv("../input/sample_submission.csv")
test.head()
train.head()
x_train = train.iloc[:,1:]

y_train = train.iloc[:,0]
plt.subplot(221)

plt.imshow(np.reshape(np.array(x_train.iloc[0]),(28,28)), cmap=plt.get_cmap('gray'))

plt.subplot(222)

plt.imshow(np.reshape(np.array(x_train.iloc[1]),(28,28)), cmap=plt.get_cmap('gray'))

plt.subplot(223)

plt.imshow(np.reshape(np.array(x_train.iloc[2]),(28,28)), cmap=plt.get_cmap('gray'))

plt.subplot(224)

plt.imshow(np.reshape(np.array(x_train.iloc[3]),(28,28)), cmap=plt.get_cmap('gray'))

plt.show()
sns.catplot(x="label",kind="count", data=train,height=7)
x_train=x_train/255.0

test=test/255.0
x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
y_train=to_categorical(y_train,num_classes=10)

print(x_train.shape,y_train.shape,test.shape)
num_classes = y_train.shape[1]

num_pixels = x_train.shape[1]
seed=7

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.10, random_state=seed)
def cnn_model():

    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu', input_shape = (28,28,1)))

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.20))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Dropout(0.20))

    model.add(Flatten())

    model.add(Dense(128, activation = "relu"))

    model.add(Dropout(0.5))

    model.add(Dense(128, activation = "relu"))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation = "softmax"))

    model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])

    return model



model=cnn_model()
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau

filepath1="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

filepath2 = "best_weights.hdf5"

checkpoint1 = ModelCheckpoint(filepath1, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

checkpoint2 = ModelCheckpoint(filepath2, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')



reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 

                              patience=3, 

                              verbose=1, 

                              factor=0.5, 

                              min_lr=0.00001)



callbacks_list = [checkpoint1,checkpoint2,reduce_lr]





epochs = 1 #increase for better results

batch_size = 640
datagen = ImageDataGenerator(

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1)  # randomly shift images vertically (fraction of total height)





datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test,y_test),

                              verbose = 1, steps_per_epoch=x_train.shape[0], callbacks=callbacks_list)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
model.load_weights("best_weights.hdf5")

submit.shape
test.shape
submit.Label =model.predict_classes(test)
submit.head()

submit.to_csv('submit.csv',index=False)