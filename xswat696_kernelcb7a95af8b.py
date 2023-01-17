# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from matplotlib import pyplot

from keras.utils.np_utils import to_categorical 

from sklearn.model_selection import train_test_split

import keras

import random 

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard, LambdaCallback

from keras.utils.data_utils import Sequence

from sklearn.metrics import confusion_matrix



import cv2

from tensorflow.python.keras.utils.data_utils import Sequence



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop



from imgaug import augmenters as iaa

import imgaug as ia
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))

train, X_test, Y_test = pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv'), pd.read_csv('../input/sample_submission.csv')

# Any results you write to the current directory are saved as output.

Y_train = train["label"]

X_train = train.drop("label",axis = 1) 

Y_test = Y_test['Label']



X_train = X_train / 255.0

X_test = X_test / 255.0



X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)



X_train, X_pub, Y_train, Y_pub = train_test_split(X_train, Y_train, test_size=0.1, random_state=123, stratify=Y_train)



Y_train = to_categorical(Y_train, num_classes = 10)

Y_pub = to_categorical(Y_pub, num_classes = 10)

Y_test = to_categorical(Y_test, num_classes = 10)
class DataGenerator(keras.utils.Sequence):



    def __init__(self, x_set, y_set, batchSize, seq_det, tot_epocs):

        self.batchSize = batchSize

        self.seq_det = seq_det

        self.tot_epocs = tot_epocs

        self.seq_det = seq_det

        self.epoc_counter = 1

        self.xTrain = x_set; self.yTrain = y_set

        self.on_epoch_end()

        

    def __len__(self):

        return int(np.ceil(len(self.xTrain) / float(self.batchSize)))



    def __getitem__(self, index):

        indexes = random.sample(range(len(self.xTrain)), self.batchSize)

        if (self.epoc_counter % 2) == 0:

            # alternate augmented and not augmented images

            random_augmented_images, random_augmented_labels = self.do_augmentation(self.seq_det, self.xTrain[indexes], self.yTrain[indexes])

        else:

            random_augmented_images, random_augmented_labels = self.xTrain[indexes], self.yTrain[indexes]

        return np.array(random_augmented_images), np.array(random_augmented_labels)

    

    def do_augmentation(self,seq_det, X_train, labels):

        X_train_aug = [(x[:,:,:] * 255.0).astype(np.uint8) for x in X_train]

        X_train_aug = seq_det.augment_images(X_train_aug)

        X_train_aug = [(x[:,:,:].astype(np.float64)) / 255.0 for x in X_train_aug]

        return np.array(X_train_aug), np.array(labels)



    def on_epoch_end(self):

        self.epoc_counter += 1



def do_augmentation(seq_det, X_train):

    X_train_aug = [(x[:,:,:] * 255.0).astype(np.uint8) for x in X_train]

    X_train_aug = seq_det.augment_images(X_train_aug)

    X_train_aug = [(x[:,:,:].astype(np.float64)) / 255.0 for x in X_train_aug]

    return np.array(X_train_aug)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([

    

        iaa.Fliplr(0.5),  

        iaa.Flipud(0.2),  

        sometimes(iaa.Affine(

            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},

            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},

            rotate=(-10, 10),  

            shear=(-5, 5),  

            order=[0, 1],

            cval=(0, 255),  

            mode=ia.ALL

        )),

       iaa.SomeOf((0, 5),

                   [sometimes(iaa.Superpixels(p_replace=(0, 1.0),

                                                     n_segments=(20, 200))),

                       iaa.OneOf([

                               iaa.GaussianBlur((0, 1.0)),

                               iaa.AverageBlur(k=(3, 5)),

                               iaa.MedianBlur(k=(3, 5)),

                       ]),

                   

                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),

                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                       iaa.SimplexNoiseAlpha(iaa.OneOf([

                               iaa.EdgeDetect(alpha=(0.5, 1.0)),

                               iaa.DirectedEdgeDetect(alpha=(0.5, 1.0),

                                                      direction=(0.0, 1.0)),

                       ])),

                       iaa.AdditiveGaussianNoise(loc=0,

                                                 scale=(0.0, 0.01 * 255),

                                                 per_channel=0.5),

                       iaa.OneOf([

                               iaa.Dropout((0.01, 0.05), per_channel=0.5),

                               iaa.CoarseDropout((0.01, 0.03),

                                                 size_percent=(0.01, 0.02),

                                                 per_channel=0.2),

                       ]),

                   

                

                   

                       iaa.OneOf([

                               iaa.Multiply((0.9, 1.1), per_channel=0.5),

                               iaa.FrequencyNoiseAlpha(

                                       exponent=(-1, 0),

                                       first=iaa.Multiply((0.9, 1.1),

                                                          per_channel=True),

                                       second=iaa.ContrastNormalization(

                                               (0.9, 1.1))

                               )

                       ]),



                       sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5),

                                                           sigma=0.25)),

                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),

                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))

                   

        ],random_order=True)

])



# test augmentation

set_test = iaa.Sequential([

                    iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}),

                    iaa.EdgeDetect(alpha=(0.0, 1.0))

])



seq_test_det = set_test.to_deterministic()

seq_det = seq.to_deterministic()
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))



model.compile(optimizer=Adam(lr=0.0005), loss = "categorical_crossentropy", metrics=["accuracy"])
BATCH_SIZE = 60

EPOCHS = 20



params = {'batchSize': 64,

          'seq_det': seq_det,

          'tot_epocs': EPOCHS}



training_generator = DataGenerator(X_train, Y_train, **params)



history = model.fit_generator(generator = training_generator,

                              epochs=EPOCHS,

                              validation_data=(X_pub, Y_pub),

                              verbose = 1, 

                              use_multiprocessing=False

                             )
Y_pred_augm = history.model.predict(do_augmentation(seq_test_det,X_pub))

Y_pred = history.model.predict(X_pub)



Y_pred_augm_classes = np.argmax(Y_pred_augm,axis = 1) 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(Y_pub,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

confusion_mtx
# agumented data cm (one can try to play a bit with augmenting test images, it might predict 

# correctly some images but with my settings overall it is worse - needed more time trying)

confusion_mtx = confusion_matrix(Y_true, Y_pred_augm_classes)

confusion_mtx
results = history.model.predict(X_test)

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission_mnist.csv",index=False)