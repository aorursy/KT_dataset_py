import numpy as np

import cv2

import pandas as pd

import os

import matplotlib.pyplot as plt

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Activation, Flatten, Dropout, BatchNormalization

from tensorflow.python.keras import optimizers, regularizers

from tensorflow.python.keras.applications import InceptionResNetV2

from keras.applications.inception_resnet_v2 import preprocess_input

from tensorflow.python.keras.models import Model

from tensorflow.python.keras.optimizers import sgd

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array, array_to_img
df_train=pd.read_csv('../input/labels.csv')

test_df=pd.read_csv('../input/sample_submission.csv')

image_size = 299
model = Sequential()

model.add(InceptionResNetV2(include_top=False, pooling='avg', weights="imagenet"))

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(120, activation='softmax'))

model.layers[0].trainable = False

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
df_test=pd.read_csv("../input/sample_submission.csv")
df_test.head()
df_train.head()
df_train['id'] = '../input/train/'+df_train['id']+'.jpg'

df_test['id']='../input/test/'+df_test['id']+'.jpg'
plt.imshow(np.array(cv2.imread(df_train['id'][0])))
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255., 

                             horizontal_flip=True,width_shift_range = 0.2,height_shift_range = 0.2,

                             validation_split=0.15)



test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,rescale=1./255.)


train_generator=datagen.flow_from_dataframe(dataframe=df_train,x_col="id",y_col="breed",has_ext=False,

                                            subset="training",batch_size=16,seed=1998,shuffle=True,

                                            class_mode="categorical",target_size=(image_size, image_size))



valid_generator=datagen.flow_from_dataframe(dataframe=df_train,x_col="id",y_col="breed",has_ext=False,

                                            subset="validation",batch_size=1,seed=1998,shuffle=True,

                                            class_mode="categorical",target_size=(image_size, image_size))



x,y=train_generator.next()
for i in range(0,5):

    image = x[i]

    image=image/255

    plt.imshow(array_to_img(image))

    plt.show()
test_generator=test_datagen.flow_from_dataframe(dataframe=df_test,x_col="id",y_col=None,has_ext=False,batch_size=1,

                                            seed=42,shuffle=False,class_mode=None,target_size=(image_size, image_size))
model.fit_generator(generator=train_generator,steps_per_epoch=train_generator.n,

                    validation_data=valid_generator,validation_steps=train_generator.n,epochs=2)
pred=model.predict_generator(test_generator,verbose=1)
labels = (train_generator.class_indices)

labels = list(labels.keys())

df = pd.DataFrame(data=pred,columns=labels)



columns = list(df)

columns.sort()

df = df.reindex(columns=columns)



filenames = test_df["id"]

df["id"]  = filenames



cols = df.columns.tolist()

cols = cols[-1:] + cols[:-1]

df = df[cols]

df.head(5)



df.to_csv("submission6.csv",index=False)