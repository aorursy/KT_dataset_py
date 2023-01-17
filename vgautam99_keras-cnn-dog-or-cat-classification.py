import os

import random

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# DEFINE CONSTANTS



FAST_RUN = False

IMAGE_WIDTH = 128

IMAGE_HEIGHT = 128

IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS = 3
# JOIN DIRECTORYS



train_dir = os.path.join('../input/train/train/')

no_of_train_data = len(os.listdir(train_dir))



print('no of train data: {}'.format(no_of_train_data))
# CREAT CSV FILES



categories = list()

filenames = os.listdir(train_dir)

for filename in filenames:

    name = filename.split('.')[0]

    if name=='dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
df.head()
df.tail()
# PLOT VALUE_COUNT GRAPH



df['category'].value_counts().plot.bar(color=['r','b'])
# CREATE MODEL ARCHITECTURE



model = Sequential()



model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

          

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()
earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',

                                           patience=2,

                                           verbose=1,

                                           factor=0.5,

                                           min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]
df['category'] = df['category'].replace({0:'cat', 1:'dog'})
df.head()
train_df, validation_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validation_df = validation_df.reset_index(drop=True)
train_df['category'].value_counts().plot.bar(color=['r', 'b'])
validation_df['category'].value_counts().plot.bar(color=['b', 'r'])
total_train = train_df.shape[0]

total_validate = validation_df.shape[0]

batch_size = 64
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    directory='../input/train/train/', 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validation_df, 

    directory='../input/train/train/', 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
shape = list()

plt.figure(figsize=(12,12))

for i in range(0, 15):

    plt.subplot(5, 3, i+1)

    for x_batch, _ in train_generator:

        img = x_batch[0]

        shape.append(img.shape)

        plt.imshow(img)

        break

plt.tight_layout()

plt.show()
# FIT MODEL



epochs=3 if FAST_RUN else 50

history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
model.save_weights('model.h5')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(history.history['loss'], color='b', label="Training loss")

ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(np.arange(1, epochs, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(history.history['acc'], color='b', label="Training accuracy")

ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(1, epochs, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()
test_filenames = os.listdir('../input/test1/test1/')

test_df = pd.DataFrame(

{

    'filename': test_filenames

})
no_samples = test_df.shape[0]

print(no_samples)
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df,

    directory='../input/test1/test1/',

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)
predict = model.predict_generator(test_generator, steps=np.ceil(no_samples/batch_size))
print(predict)
test_df['category'] = np.argmax(predict, axis=-1)
test_df
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
label_map
test_df['category'] = test_df['category'].replace(label_map)
test_df
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })
test_df['category'].value_counts().plot.bar(color=['r', 'b'])
sample_test = test_df.head(18)

sample_test.head()

plt.figure(figsize=(12, 15))

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img("../input/test1/test1/"+filename, target_size=IMAGE_SIZE)

    plt.subplot(6, 3, index+1)

    plt.imshow(img)

    if category == 1:

        name = 'dog'

    else:

        name = 'cat'

    plt.xlabel(name + '(' + "{}".format(category) + ')' )

plt.tight_layout()

plt.show()