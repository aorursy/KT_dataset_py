!pip install efficientnet

import efficientnet
import json

import math

import os



import cv2

import tensorflow as tf

from PIL import Image

import numpy as np

import keras

from keras import layers

from keras.applications import MobileNetV2

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, load_model

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, concatenate, Input, add

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score

import scipy

from tqdm import tqdm
!mkdir ../tmp

!unzip -q ../input/recursion-2019-load-resize-and-save-images/train.zip -d ../tmp

!unzip -q ../input/recursion-2019-load-resize-and-save-images/test.zip -d ../tmp 
load_path = '../tmp/train'

save_path = '../tmp/train_resized'

if not os.path.exists(save_path):

    os.makedirs(save_path)



for code in tqdm(os.listdir(load_path)):

    path = f'{load_path}/{code}'

    

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (300, 300))

    

    cv2.imwrite(f'{save_path}/{code}.png', img)
load_path = '../tmp/test'

save_path = '../tmp/test_resized'

if not os.path.exists(save_path):

    os.makedirs(save_path)



for code in tqdm(os.listdir(load_path)):

    path = f'{load_path}/{code}'

    

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (300, 300))

    

    cv2.imwrite(f'{save_path}/{code}.png', img)
train_df = pd.read_csv('../input/recursion-cellular-image-classification/train.csv')

test_df = pd.read_csv('../input/recursion-cellular-image-classification/test.csv')



train_df['category'] = train_df['experiment'].apply(lambda x: x.split('-')[0])

test_df['category'] = test_df['experiment'].apply(lambda x: x.split('-')[0])



train_target_df = pd.get_dummies(train_df['sirna'])



print(train_df.shape)

print(test_df.shape)

print(train_target_df.shape)



train_df.head()
train_idx, val_idx = train_test_split(

    train_df.index, test_size=0.15, random_state=2019

)



print(train_idx.shape)

print(val_idx.shape)
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',

                 base_path = '../input/recursion-cellular-image-classification-224-jpg/train/train',

                 batch_size=32, dim=(224, 224), n_channels=3, ext='png',

                 rotation_range=0, fill_mode='nearest', swap=False,

                 vertical_flip=False, horizontal_flip=False, rescale=1/255.,

                 n_classes=5, random_state=2019, shuffle=True):

        self.dim = dim

        self.batch_size = batch_size

        self.df = df

        self.mode = mode

        self.base_path = base_path

        self.rotation_range=rotation_range

        self.target_df = target_df

        self.list_IDs = list_IDs

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.ext = ext

        self.rescale = rescale

        self.vertical_flip = vertical_flip

        self.horizontal_flip = horizontal_flip

        self.random_state = random_state

        self.swap = swap

        

        self.fill_mode = self.__compute_fill_mode(fill_mode)

        

        np.random.seed(self.random_state)

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        

        X = self.__generate_X(list_IDs_batch)

        

        if self.mode == 'fit':

            y = self.__generate_y(list_IDs_batch)

            return X, y

        

        elif self.mode == 'predict':

            return X

        else:

            raise AttributeError('The parameter mode should be set to "fit" or "predict".')

        

    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)

    

    def __generate_X(self, list_IDs_batch):

        'Generates data containing batch_size samples'

        # Initialization

        X_1 = np.empty((self.batch_size, *self.dim, self.n_channels))

        X_2 = np.empty((self.batch_size, *self.dim, self.n_channels))

        

        # Generate data

        for i, ID in enumerate(list_IDs_batch):

            code = self.df['id_code'].iloc[ID]

            

            img_path_1 = f"{self.base_path}/{code}_s1.{self.ext}"

            img_path_2 = f"{self.base_path}/{code}_s2.{self.ext}"

            

            img1 = self.__load_image(img_path_1)

            img2 = self.__load_image(img_path_2)

            

            if self.swap and np.random.rand() > 0.5:

                img1, img2 = img2, img1

            

            # Store samples

            X_1[i,] = img1

            X_2[i,] = img2



        return [X_1, X_2]

    

    def __generate_y(self, list_IDs_batch):

        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        

        for i, ID in enumerate(list_IDs_batch):

            sirna = self.target_df.iloc[ID]

            y[i, ] = sirna

        

        return y

    

    def __load_image(self, img_path):

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img = self.rescale * img.astype(np.float32)



        return img

    

    def __compute_fill_mode(self, fill_mode):

        convert_cv2 = {

            'nearest': cv2.BORDER_REPLICATE,

            'reflect': cv2.BORDER_REFLECT,

            'wrap': cv2.BORDER_WRAP,

            'constant': cv2.BORDER_CONSTANT

        }

        

        return convert_cv2[fill_mode]

    

    def __random_transform(img):

        if np.random.rand() > 0.5 and self.vertical_flip:

            img = cv2.flip(img, 0)

        if np.random.rand() > 0.5 and self.horizontal_flip:

            img = cv2.flip(img, 1)

        

        # Random Rotation

        rotation = self.rotation_range * np.random.rand()

        

        rows,cols = img.shape[:2]

        M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation,1)

        img = cv2.warpAffine(img,M,(cols,rows), borderMode=self.fill_mode)

        

        return img
BATCH_SIZE = 32

train_generator = DataGenerator(

    train_idx, 

    df=train_df,

    target_df=train_target_df,

    batch_size=BATCH_SIZE, 

    vertical_flip=True,

    horizontal_flip=True,

    swap=True,

    dim=(300, 300),

    base_path='../tmp/train_resized',

    rotation_range=15,

    n_classes=train_target_df.shape[1]

)



val_generator = DataGenerator(

    val_idx, 

    df=train_df,

    target_df=train_target_df,

    batch_size=BATCH_SIZE, 

    vertical_flip=True,

    horizontal_flip=True,

    swap=True,

    dim=(300, 300),

    base_path='../tmp/train_resized',

    rotation_range=15,

    n_classes=train_target_df.shape[1]

)



test_generator = DataGenerator(

    test_df.index, 

    df=test_df,

    batch_size=1, 

    shuffle=False,

    mode='predict',

    n_classes=train_target_df.shape[1],

    dim=(300, 300),

    base_path='../tmp/test_resized'

)
def build_model(n_classes, input_shape=(224, 224, 3)):

    # First load mobilenet

    backbone = efficientnet.EfficientNetB3(

        weights='imagenet', 

        include_top=False,

        input_shape=input_shape

    )

    

    im_inp_1 = Input(shape=input_shape)

    im_inp_2 = Input(shape=input_shape)



    x1 = backbone(im_inp_1)

    x2 = backbone(im_inp_2)



    x1 = GlobalAveragePooling2D()(x1)

    x2 = GlobalAveragePooling2D()(x2)



    out = add([x1, x2])

    out = Dropout(0.5)(out)



    out = Dense(n_classes, activation='softmax')(out)



    model = Model(inputs=[im_inp_1, im_inp_2], outputs=out)

    

    model.compile(Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model
model = build_model(

    input_shape=(300, 300, 3),

    n_classes=train_target_df.shape[1]

)

model.summary()
checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_loss', 

    verbose=0, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)



history = model.fit_generator(

    train_generator,

    validation_data=val_generator,

    callbacks=[checkpoint],

    use_multiprocessing=False,

    workers=1,

    verbose=2,

    epochs=20

)
with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
categories = train_df['category'].unique()

output_df = []



for category in categories:

    # Retrieve desired category

    category_df = train_df[train_df['category'] == category]

    cat_test_df = test_df[test_df['category'] == category].copy()

    

    print('\n' + '=' * 40)

    print("CURRENT CATEGORY:", category)

    print('-' * 40)

    

    train_idx, val_idx = train_test_split(

        category_df.index, 

        random_state=2019,

        test_size=0.15

    )

    

    # Create new generators

    train_generator = DataGenerator(

        train_idx, 

        df=train_df,

        target_df=train_target_df,

        batch_size=BATCH_SIZE, 

        vertical_flip=True,

        horizontal_flip=True,

        swap=True,

        rotation_range=15,

        dim=(300, 300),

        base_path='../tmp/train_resized',

        n_classes=train_target_df.shape[1]

    )



    val_generator = DataGenerator(

        val_idx, 

        df=train_df,

        target_df=train_target_df,

        batch_size=BATCH_SIZE, 

        vertical_flip=True,

        horizontal_flip=True,

        swap=True,

        rotation_range=15,

        dim=(300, 300),

        base_path='../tmp/train_resized',

        n_classes=train_target_df.shape[1]

    )



    test_generator = DataGenerator(

        cat_test_df.index, 

        df=test_df,

        batch_size=1, 

        shuffle=False,

        mode='predict',

        n_classes=train_target_df.shape[1],

        dim=(300, 300),

        base_path='../tmp/test_resized'

    )



    # Restore previously trained model

    model.load_weights('model.h5')

    model.compile(

        Adam(0.0001), 

        loss='categorical_crossentropy', 

        metrics=['accuracy']

    )



    # Train model only on data for specific category

    checkpoint = ModelCheckpoint(

        f'model_{category}.h5', 

        monitor='val_loss', 

        verbose=0, 

        save_best_only=True, 

        save_weights_only=False,

        mode='auto'

    )



    history_category = model.fit_generator(

        train_generator,

        validation_data=val_generator,

        callbacks=[checkpoint],

        use_multiprocessing=False,

        workers=1,

        verbose=2,

        epochs=10

    )



    # Make prediction and add to output dataframe

    y_pred = model.predict_generator(

        test_generator,

        workers=2,

        use_multiprocessing=True,

        verbose=1

    )



    cat_test_df['sirna'] = y_pred.argmax(axis=1)

    output_df.append(cat_test_df[['id_code', 'sirna']])



    # Save history

    with open(f'history_{category}.json', 'w') as f:

        json.dump(history_category.history, f)
output_df = pd.concat(output_df)

output_df.to_csv('submission.csv', index=False)