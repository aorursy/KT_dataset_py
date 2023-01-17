# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

 

from tqdm import tqdm

import cv2

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

%matplotlib inline

style.use('fivethirtyeight')

sns.set(style='whitegrid', color_codes=True)



from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping



import random

import uuid

import shutil



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
random.seed(42)

np.random.seed(42)
PATH = "../input"

IMAGE_SIZE = (32, 32)



train_dir=f'{PATH}/train/train'

test_dir=f'{PATH}/test/test'



train_img = os.listdir(train_dir)

test_img = os.listdir(test_dir)



train_df =pd.read_csv(f'{PATH}/train.csv')

test_df = pd.read_csv(f'{PATH}/sample_submission.csv')



# Create new images for the imbalanced data

train_new_dir = 'train'



# Make sure the folder is empty

if os.path.exists(train_new_dir):

    shutil.rmtree(train_new_dir)

else:

    os.makedirs(train_new_dir)



print(f"The number of rows in train and test set are {len(train_df)} and {len(test_df)}")
train_df.head()
test_df.head()
sns.countplot(train_df['has_cactus'])
fig = plt.figure(figsize=(10, 8))

for idx, img in enumerate(np.random.choice(train_img, 20)):

    ax = fig.add_subplot(4, 20//4, idx+1, xticks=[], yticks=[])

    im = cv2.imread(f'{train_dir}/{img}')

    plt.imshow(im)

    lab = train_df.loc[train_df['id'] == img, 'has_cactus'].values[0]

    ax.set_title(f'has_cactus: {lab}')
train_new = []

if len(os.listdir(train_new_dir))==0:

    for idx, row in train_df[train_df['has_cactus']==0].iterrows():

        # get image

        img = cv2.imread(f'{train_dir}/{row["id"]}')

        # flip image

        for i in range(2):

            f = cv2.flip(img, i)

            img_id = uuid.uuid4().hex + '.jpg'

            cv2.imwrite(f'{train_new_dir}/{img_id}', f)

            train_new.append({'id': img_id, 'has_cactus': 0}) 
print(f'{len(os.listdir(train_new_dir))} new images generated')
# copy the read-only files to new dir

!cp ../input/train/train/*.jpg train/
print(f"The number of rows in train and test set are {len(os.listdir(train_new_dir))} and {len(test_df)}")
if len(train_df) == len(os.listdir(train_dir)):

    train_df=train_df.append(train_new, ignore_index=True)

    sns.countplot(train_df['has_cactus'])
# splitting data into train and validation

train, valid = train_test_split(train_df, stratify=train_df.has_cactus, test_size=0.33, random_state=2019)
BATCH_SIZE = 8



train_gen=ImageDataGenerator(

    rescale=1./255, 

    rotation_range=10,  

    zoom_range = 0.1, 

    width_shift_range=0.1,  

    height_shift_range=0.1,  

    fill_mode='nearest'

)  



train_generator=train_gen.flow_from_dataframe(

    x_col='id',                                  

    y_col='has_cactus',

    dataframe=train, 

    directory=train_new_dir, 

    class_mode='other',

    color_mode='rgb',

    batch_size=BATCH_SIZE,

    target_size=IMAGE_SIZE,

    shuffle=True,

    seed=2019

)



valid_generator=train_gen.flow_from_dataframe(

    x_col='id',                                  

    y_col='has_cactus',

    dataframe=valid, 

    directory=train_new_dir, 

    class_mode='other',

    color_mode='rgb',

    batch_size=BATCH_SIZE,

    target_size=IMAGE_SIZE,

    shuffle=True,

    seed=2019

)
drop_rate=0.2



model = Sequential()



model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(32,32,3)))

model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=3,activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(drop_rate))



model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(drop_rate))



model.add(GlobalAveragePooling2D())

model.add(BatchNormalization())

model.add(Dropout(drop_rate))



model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=Adam(), metrics=['accuracy'])
file_path = 'best_weights.h5'



callbacks = [

    ModelCheckpoint(file_path, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max'),

    ReduceLROnPlateau(monitor = 'val_loss', factor = 0.8, patience = 4, verbose = 1, mode = 'min', min_lr = 1e-8),

    EarlyStopping(monitor = 'val_loss', min_delta = 1e-10, patience = 32, verbose = 1, restore_best_weights = True)

]
epochs=128

history=model.fit_generator(train_generator,

                            steps_per_epoch=train_generator.n//train_generator.batch_size,

                            epochs=epochs,

                            verbose = 1,

                            shuffle=True,

                            validation_data=valid_generator,

                            validation_steps=valid_generator.n//valid_generator.batch_size,

                            callbacks = callbacks)
history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
# Make sure not output too many files

if os.path.exists(train_new_dir):

    shutil.rmtree(train_new_dir)
model.load_weights(file_path)
test_gen=ImageDataGenerator(

    rescale=1./255, 

)  



test_generator=test_gen.flow_from_dataframe(

    x_col='id',                                  

    y_col='has_cactus',

    dataframe=test_df, 

    directory=test_dir, 

    class_mode='other',

    color_mode='rgb',

    batch_size=1,

    target_size=IMAGE_SIZE,

    shuffle=False

)
test_generator.reset()

pred = model.predict_generator(test_generator, verbose=1, steps=test_generator.n)



pred[pred>0.99]=1

pred[pred<0.01]=0
test_df['has_cactus'] = pred

test_df.to_csv('submission.csv', index = False)