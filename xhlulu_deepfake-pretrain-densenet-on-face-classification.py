import numpy as np

import cv2

import pandas as pd

from tqdm.notebook import tqdm 

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers

from tensorflow.keras.applications import DenseNet121

from tensorflow.keras.callbacks import Callback, ModelCheckpoint

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
original_fake_paths = []



for dirname, _, filenames in tqdm(os.walk('/kaggle/input/1-million-fake-faces/')):

    for filename in filenames:

        original_fake_paths.append([os.path.join(dirname, filename), filename])
save_dir = '/kaggle/tmp/fake/'



if not os.path.exists(save_dir):

    os.makedirs(save_dir)
fake_paths = [save_dir + filename for _, filename in original_fake_paths]
for path, filename in tqdm(original_fake_paths):

    img = cv2.imread(path)

    img = cv2.resize(img, (224, 224))

    cv2.imwrite(os.path.join(save_dir, filename), img)
train_fake_paths, test_fake_paths = train_test_split(fake_paths, test_size=20000, random_state=2019)



fake_train_df = pd.DataFrame(train_fake_paths, columns=['filename'])

fake_train_df['class'] = 'FAKE'



fake_test_df = pd.DataFrame(test_fake_paths, columns=['filename'])

fake_test_df['class'] = 'FAKE'
real_dir = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/'

eval_partition = pd.read_csv('/kaggle/input/celeba-dataset/list_eval_partition.csv')



eval_partition['filename'] = eval_partition.image_id.apply(lambda st: real_dir + st)

eval_partition['class'] = 'REAL'
real_train_df = eval_partition.query('partition in [0, 1]')[['filename', 'class']]

real_test_df = eval_partition.query('partition == 2')[['filename', 'class']]
train_df = pd.concat([real_train_df, fake_train_df])

test_df = pd.concat([real_test_df, fake_test_df])
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)



train_gen = datagen.flow_from_dataframe(

    train_df,

    target_size=(224, 224),

    batch_size=64,

    class_mode='binary',

    subset='training'

)



val_gen = datagen.flow_from_dataframe(

    train_df,

    target_size=(224, 224),

    batch_size=64,

    class_mode='binary',

    subset='validation'

)
datagen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(

    test_df,

    target_size=(224, 224),

    batch_size=64,

    class_mode='binary'

)
densenet = DenseNet121(

    weights='/kaggle/input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    input_shape=(224,224,3)

)



for layer in densenet.layers:

    layer.trainable = False
def build_model(densenet):

    model = Sequential()

    model.add(densenet)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.5))

    

    model.add(layers.Dense(256, activation='relu'))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.5))

    

    model.add(layers.Dense(1, activation='sigmoid'))

    

    model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(lr=0.0005),

        metrics=['accuracy']

    )

    

    return model
model = build_model(densenet)

model.summary()
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)



train_history_step1 = model.fit_generator(

    train_gen,

    validation_data=val_gen,

    steps_per_epoch=len(train_gen),

    validation_steps=len(val_gen),

    callbacks=[checkpoint],

    epochs=7

)
model.load_weights('model.h5')

for layer in model.layers:

    layer.trainable = True



train_history_step2 = model.fit_generator(

    train_gen,

    validation_data=val_gen,

    steps_per_epoch=len(train_gen),

    validation_steps=len(val_gen),

    callbacks=[checkpoint],

    epochs=3

)
pd.DataFrame(train_history_step1.history).to_csv('history1.csv')

pd.DataFrame(train_history_step2.history).to_csv('history2.csv')