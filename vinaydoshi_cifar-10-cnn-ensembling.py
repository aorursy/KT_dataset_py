# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import classification_report



from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD

from keras.datasets import cifar10

import seaborn as sns

import numpy as np

import argparse

import os

os.chdir(r'/kaggle/working')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.
ap = argparse.ArgumentParser()

ap.add_argument('-o','--output',required=True, help = 'Path to HDF5 database')

ap.add_argument('-m','--model', required=True, help = 'Path to output model')

ap.add_argument('-n','--num-models', default=5, type=int, help = '# of models to train')

#args = vars(ap.parse_args())

args = vars(ap.parse_args([r'--output=/kaggle/working/',

                           r'--model=/kaggle/working/',

                           r'--num-models=5'

                          ]))
from keras import backend as K

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dense

from keras.layers.core import Dropout







class MiniVGGNet:

    def build(width, height, depth, classes):

        model = Sequential()

        inputShape = (height, width, depth)

        chanDim = -1

        dataFormat = 'channels_last'

        

        if K.image_data_format()=='channels_first':

            inputShape = (depth, height, width)

            chanDim = 1

            dataFormat = 'channels_first'

        

        model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', input_shape = inputShape))

        model.add(Activation('relu'))

        model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', input_shape = inputShape))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis = chanDim))

        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), data_format = dataFormat))

        model.add(Dropout(0.25))

        

        model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', input_shape = inputShape))

        model.add(Activation('relu'))

        model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', input_shape = inputShape))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis = chanDim))

        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), data_format = dataFormat))

        model.add(Dropout(0.25))

        

        model.add(Flatten())

        model.add(Dense(512))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis = -1))

        model.add(Dropout(0.5))

        

        model.add(Dense(classes))

        model.add(Activation('softmax'))

        

        return model

        
(X_train,  y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float')/255.0

X_test = X_test.astype('float')/255.0

lb = LabelBinarizer()

y_train = lb.fit_transform(y_train)

y_test = lb.transform(y_test)
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
aug = ImageDataGenerator(rotation_range=30, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1,

                         shear_range=0.2, fill_mode='nearest', vertical_flip=True

                        )
predictions=[]

for i in np.arange(0, args['num_models']):

    

    print(f"Training model: {i+1}/{args['num_models']}")

    opt = SGD(lr=0.01, decay = 0.01/40, momentum=0.9, nesterov=True)

    model = MiniVGGNet.build(width=32, height=32, classes=10, depth=3)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics = ['accuracy'])

    H = model.fit_generator(aug.flow(X_train, y_train, batch_size=64), validation_data=(X_test, y_test), epochs = 40,

                            steps_per_epoch = len(X_train)/64, verbose=0

                           )

    p = [args['model'], f'model{i}.model']

    model.save(os.path.sep.join(p))

    preds = model.predict(X_test, batch_size=64)

    predictions.append(preds)

    report = classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=labelNames)

    p = [args['output'], f'model{i}.txt']

    with open(os.path.sep.join(p), 'w') as f:

          f.write(report)

    p = [args['output'], f'model{i}.png']

    plt.figure(figsize=(18,12))

    sns.lineplot(x=np.arange(0,40), y=H.history['loss'], label = 'Train_Loss')

    sns.lineplot(x=np.arange(0,40), y=H.history['val_loss'], label = 'Val_Loss')

    sns.lineplot(x=np.arange(0,40), y=H.history['accuracy'], label = 'Train_Accuracy')

    sns.lineplot(x=np.arange(0,40), y=H.history['val_accuracy'], label = 'Val_Accuracy')

    plt.title(f'Training Loss and Accuracy for model {i}')

    plt.xlabel('# of Epochs')

    plt.ylabel('Loss/Accuracy')

    plt.legend(loc='best')

    plt.savefig(os.path.sep.join(p))

    plt.close()          
predictions = np.average(predictions,axis=0)

print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))
# from IPython.display import FileLink

# for i in np.arange(0, args['num_models']):

#     FileLink(fr'model{i}.txt') 
# from IPython.display import FileLink

# FileLink(r'model5.png')
# import glob
# modelPaths = os.path.sep.join([args['model'],'*.model'])

# modelPaths = list(glob.glob(modelPaths))

# models = []
# modelPaths
# from keras.models import load_model

# for i, modelPath in enumerate(modelPaths):

#     print(f'Loading Model: {i+1}/{len(modelPaths)}')

#     models.append(load_model(modelPath))
# models
# predictions=[]

# for model in models:

#     predictions.append(model.predict(X_test, batch_size=64))



# predictions = np.average(predictions,axis=0)

# print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))