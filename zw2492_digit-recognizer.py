import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for plotting

from collections import Counter



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
#load the data 

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

print(train.shape)

print(test.shape)
#examples of training data

train.head()

# 28 * 28 = 784
#explore the distribution of labels in training data

train_label_ct = Counter(train['label'])

train_label_ct
#examples of test data

test.head()
x_train = (train.ix[:,1:].values).astype('float32') # all pixel values

y_train = train.ix[:,0].values.astype('int32') # only labels i.e targets digits

x_test = test.values.astype('float32')
#example of image

plt.imshow(x_train[100].reshape((28,28)))
#normalize the data

x_train = x_train/255.0

x_test = x_test/255.0
#reshape to the input size 28 * 28 * 1

X_train = x_train.reshape(x_train.shape[0], 28, 28,1)

X_test = x_test.reshape(x_test.shape[0], 28, 28,1)
#set the parameters

batch_size = 64

num_classes = 10

epochs = 20

input_shape = (28, 28, 1)
#convert label to one-hot coding

y_train = keras.utils.to_categorical(y_train, num_classes)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)
#build the model

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))

model.add(MaxPool2D((2, 2)))

model.add(Dropout(0.20))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.RMSprop(),

              metrics=['accuracy'])



#decay the learning rate 

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.0001)



#data augmentation

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images
model.summary()
datagen.fit(X_train)

hist = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction],)
#final loss and final accuracy

final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=0)

print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
hist.history.keys()
#visualize the acc vs number of epochs

plt.plot(hist.history['accuracy'], label='acc')

plt.plot(hist.history['val_accuracy'], label = 'val_acc')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(loc='lower right')
#make the prediction and submission

y_predict = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(y_predict)+1)),

                         "Label": y_predict})

submissions.to_csv("sub.csv", index=False, header=True)