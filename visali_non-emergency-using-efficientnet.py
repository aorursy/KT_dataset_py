!pip install -U efficientnet
from keras import applications

from keras import callbacks

from keras.models import Sequential
import efficientnet.keras as efn 



model = efn.EfficientNetB7(weights='imagenet')
import cv2

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import json

import os

from tqdm import tqdm, tqdm_notebook

from keras.models import Model

from keras.layers import Dropout, Flatten, Dense

from keras import optimizers
train_dir = "../input/emergency-vehicles-identification/Emergency_Vehicles/train"

test_dir = "../input/emergency-vehicles-identification/Emergency_Vehicles/test"

train_df = pd.read_csv('../input/emergency-vehicles-identification/Emergency_Vehicles/train.csv')

train_df.head()
im = cv2.imread("../input/emergency-vehicles-identification/Emergency_Vehicles/train/1002.jpg")

plt.imshow(im)
eff_net = efn.EfficientNetB7(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(rescale=1./255)

batch_size=150
train_df.emergency_or_not=train_df.emergency_or_not.astype(str)
train_generator=datagen.flow_from_dataframe(dataframe=train_df[:1150],directory=train_dir,x_col='image_names',

                                            y_col='emergency_or_not',class_mode='binary',batch_size=batch_size,

                                            target_size=(32,32))





validation_generator=datagen.flow_from_dataframe(dataframe=train_df[1151:],directory=train_dir,x_col='image_names',

                                                y_col='emergency_or_not',class_mode='binary',batch_size=50,

                                                target_size=(32,32))
from keras.layers import Dense

from keras.optimizers import Adam



efficient_net = efn.EfficientNetB7(

    weights='imagenet',

    input_shape=(32,32,3),

    include_top=False,

    pooling='max'

)



model = Sequential()

model.add(efficient_net)

model.add(Dense(units = 120, activation='relu'))

model.add(Dense(units = 120, activation = 'relu'))

model.add(Dense(units = 1, activation='sigmoid'))

model.summary()
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
%%time

# Train model

history = model.fit_generator(

    train_generator,

    epochs = 50,

    steps_per_epoch = 15,

    validation_data = validation_generator,

    validation_steps = 7

)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1,len(acc) + 1)



plt.plot(epochs,acc,'bo',label = 'Training Accuracy')

plt.plot(epochs,val_acc,'b',label = 'Validation Accuracy')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()



plt.plot(epochs,loss,'bo',label = 'Training loss')

plt.plot(epochs,val_loss,'b',label = 'Validation Loss')

plt.title('Training and Validation Loss')

plt.legend()



plt.show()
test_df = pd.read_csv('../input/emergency-vehicles-identification/Emergency_Vehicles/test.csv')



test_datagen = ImageDataGenerator(

    rescale=1/255

)



test_generator = test_datagen.flow_from_dataframe(

    dataframe = test_df,

    directory = test_dir,

    x_col="image_names",

    target_size=(32,32),

    batch_size=1,

    shuffle=False,

    class_mode=None

)
preds = model.predict_generator(

    test_generator,

    steps=len(test_generator.filenames)

)
preds
image_ids = [name.split('/')[-1] for name in test_generator.filenames]

predictions = preds.flatten()

data = {'image_names': image_ids, 'emergency_or_not':predictions} 

submission = pd.DataFrame(data)

print(submission.head())
submission['emergency_or_not'] = submission['emergency_or_not'].apply(lambda x: 1 if x > 0.75 else 0)
submission.to_csv('submission_effnet.csv',index=False)