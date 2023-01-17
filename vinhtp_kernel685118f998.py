!pip install efficientnet
import pandas as pd

import numpy as np

import cv2

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
PATH = '../input/losers' 

train_df = pd.read_csv('../input/losers/train.csv',dtype={'category':str})

test_df = pd.read_csv('../input/losers/test.csv')



train_df['filename'] = '../input/losers/train/train/'+train_df['category']+'/'+ train_df['filename']

test_df['filename'] = '../input/losers/test/test/'+ test_df['filename']
test_df
from keras_preprocessing.image import ImageDataGenerator



train_data_gen= ImageDataGenerator(rescale=1/255)

IMG_SIZE=300

BATCH_SIZE=16
test_generator=train_data_gen.flow_from_dataframe(test_df,directory=None,

                                                      target_size=(IMG_SIZE,IMG_SIZE),

                                                      x_col="filename",

                                                      y_col=None,

                                                      class_mode=None,

                                                      shuffle=False,

                                                      batch_size=BATCH_SIZE)
from keras.models import Sequential,Model

from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization,Input,GlobalAveragePooling2D,GlobalMaxPooling2D,concatenate

from keras.layers import GlobalAveragePooling2D

from keras.callbacks import ReduceLROnPlateau

import efficientnet.keras as efn 

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score

import keras
keras.backend.clear_session()

model =efn.EfficientNetB3(weights = 'imagenet', include_top=False, input_shape = (IMG_SIZE,IMG_SIZE,3))



x = model.output

x = Dropout(0.5)(x)

x = Conv2D(1024,1,activation='relu')(x)

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation="relu")(x)

x = Dense(64, activation="sigmoid")(x)

predictions = Dense(42, activation="softmax")(x)



model = Model(inputs=model.input, outputs=predictions)
model.load_weights('../input/product-detection/model_weights.h5') 
model.summary()
y_test=model.predict(test_generator,steps=test_generator.n/BATCH_SIZE)
y_test.shape
y_classes = y_test.argmax(axis=-1)
y_classes
sub_df=pd.read_csv('../input/losers/test.csv')

sub_df.head()
y_test[:,]
sub_df['category']=y_classes
sub_df
sub_df.to_csv('submission.csv',index=False)