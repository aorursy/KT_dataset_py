import keras

import matplotlib.pyplot as plt

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPool2D,Dropout

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

trdata = ImageDataGenerator()

traindata = trdata.flow_from_directory(directory="../input/covid19-vs-healthy-xray/COVID19/Train",target_size=(384,384))

tsdata = ImageDataGenerator()

testdata = tsdata.flow_from_directory(directory="../input/covid19-vs-healthy-xray/COVID19/Test", target_size=(384,384))
model = Sequential()



# 1st Convolutional Layer

model.add(Conv2D(filters=16, input_shape=(384,384,3), kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu"))

model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu"))

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu"))

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu"))

model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu"))

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu"))

# Max Pooling

model.add(MaxPool2D(pool_size=(64,64), strides=(2,2), padding="valid"))

model.add(MaxPool2D(pool_size=(32,32), strides=(2,2), padding="valid"))

model.add(MaxPool2D(pool_size=(16,16), strides=(2,2), padding="valid"))

model.add(MaxPool2D(pool_size=(8,8), strides=(2,2), padding="valid"))

model.add(MaxPool2D(pool_size=(4,4), strides=(2,2), padding="valid"))



model.add(Flatten())

model.add(Dense(units = 288, activation = "relu"))

model.add(Dense(units = 64, activation = "relu"))

model.add(Dropout(0.2, input_shape=(64,)))

model.add(Dense(units = 32, activation = "relu"))

model.add(Dense(units = 2, activation = "softmax"))

model.summary()
from keras.optimizers import Adam

opt = Adam(lr=0.0001)

model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("Covid19Net.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')

hist = model.fit_generator(steps_per_epoch=10,generator=traindata, validation_data= testdata, validation_steps=10,epochs=16,callbacks=[checkpoint,early])
model.save('Covid19Net.h5')
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# Plot training & validation accuracy values

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

from keras.preprocessing import image

img = image.load_img("../input/covid19-vs-healthy-xray/COVID19/Negative.jpeg",target_size=(384,384))

img = np.asarray(img)

plt.imshow(img)

img = np.expand_dims(img, axis=0)

from keras.models import load_model

saved_model = load_model("Covid19Net.h5")

output = saved_model.predict(img)

if output[0][0] > output[0][1]:

    print("Positive")

else:

    print('Negative')
from keras.applications.imagenet_utils import preprocess_input

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.models import Model

from matplotlib import pyplot

from numpy import expand_dims

# load the model

# redefine model to output right after the first hidden layer

ixs = [1, 2, 3, 4, 5,6]

outputs = [model.layers[i].output for i in ixs]

model = Model(inputs=model.inputs, outputs=outputs)

# load the image with the required shape

img = load_img('../input/covid19-vs-healthy-xray/COVID19/Positive.jpg', target_size=(384, 384))

# convert the image to an array

img = img_to_array(img)

# expand dimensions so that it represents a single 'sample'

img = expand_dims(img, axis=0)

# prepare the image (e.g. scale pixel values for the vgg)

img = preprocess_input(img)

# get feature map for first hidden layer

feature_maps = model.predict(img)

# plot the output from each block

square = 1

for fmap in feature_maps:

    ix = 1

    for _ in range(square):

        ax = pyplot.subplot(square, square, ix)

        ax.set_xticks([])

        ax.set_yticks([])

        pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')

        ix += 1

    pyplot.show()
from keras.preprocessing import image

import os

positive = 0

negative = 0

saved_model = load_model("Covid19Net.h5")

for dirname, _, filenames in os.walk('../input/covid19-vs-healthy-xray/COVID19/Test/'):

    for filename in filenames:

        img = image.load_img(os.path.join(dirname, filename),target_size=(384,384))

        img = np.asarray(img)

        img = np.expand_dims(img, axis=0)

        output = saved_model.predict(img)

        if output[0][0] > output[0][1]:

            print("Positive")

            positive+=1

        else:

            print('Negative')

            negative+=1

print("Total Positive Predicted ->",positive)

print("Total Negative Predicted ->",negative)
from keras.models import load_model

pos = 0

neg = 0

saved_model = load_model("Covid19Net.h5")

for dirname, _, filenames in os.walk('/kaggle/input/covid19-detection-xray-dataset/NonAugmentedTrain/Normal/'):

    for filename in filenames:

        img = image.load_img(os.path.join(dirname, filename),target_size=(384,384))

        img = np.asarray(img)

        img = np.expand_dims(img, axis=0)

        output = saved_model.predict(img)

        if output[0][0] > output[0][1]:

            print("Positive")

            pos+=1

        else:

            print('Negative')

            neg+=1

print("Total Positive Predicted ->",pos)

print("Total Negative Predicted ->",neg)
pos1 = 0

neg1 = 0

saved_model = load_model("Covid19Net.h5")

for dirname, _, filenames in os.walk('/kaggle/input/covid19-detection-xray-dataset/NonAugmentedTrain/COVID-19/'):

    for filename in filenames:

        img = image.load_img(os.path.join(dirname, filename),target_size=(384,384))

        img = np.asarray(img)

        img = np.expand_dims(img, axis=0)

        output = saved_model.predict(img)

        if output[0][0] > output[0][1]:

            print("Positive")

            pos1+=1

        else:

            print('Negative')

            neg1+=1

print("Total Positive Predicted ->",pos1)

print("Total Negative Predicted ->",neg1)