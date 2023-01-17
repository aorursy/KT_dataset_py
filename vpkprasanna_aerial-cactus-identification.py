import pandas as pd

import os

from keras import optimizers

from keras import layers,models

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Conv2D, Flatten

from keras.models import Sequential, Model

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten

print(os.listdir("../input"))

import numpy as np
train_dataframe = pd.read_csv("../input/train.csv")

train_dataframe["has_cactus"] = np.where(train_dataframe["has_cactus"] == 1, "yes", "no")

print(train_dataframe.head())
base_dir = "../input/"

train_dir = os.path.join(base_dir,"train/train")

test_dir = os.path.join(base_dir, "test/test")

df_test=pd.read_csv('../input/sample_submission.csv')



testing_dir = os.path.join(base_dir, "test")

batch_size=150
train_datagen = ImageDataGenerator(rescale=1/255,validation_split=0.10,rotation_range=40,

    width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,

    zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
train_generator = train_datagen.flow_from_dataframe(dataframe = train_dataframe,directory = train_dir,x_col="id",

    y_col="has_cactus",target_size=(150,150),subset="training",batch_size=250,shuffle=True,class_mode="binary"

)



valid_generator = train_datagen.flow_from_dataframe(dataframe = train_dataframe,directory = train_dir,x_col="id",

    y_col="has_cactus",target_size=(150,150),subset="validation",batch_size=125,shuffle=True,class_mode="binary"

)
test_datagen = ImageDataGenerator(

    rescale=1/255

)
test_generator = test_datagen.flow_from_directory(testing_dir,target_size=(150,150),batch_size=1,

    shuffle=False,class_mode=None)
model=models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(Dropout(0.2))

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))

model.summary()



model.compile(loss='binary_crossentropy',optimizer=optimizers.rmsprop(),metrics=['acc'])
epochs=10

history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data=valid_generator,validation_steps=50)
preds = model.predict_generator(

    test_generator,

    steps=len(test_generator.filenames))
df=pd.DataFrame({'id':df_test['id'] })

df['has_cactus']=preds

df.to_csv("submission.csv",index=False)
df.head()