import sys

IN_COLAB = 'google.colab' in sys.modules

!nvidia-smi -L
# colab에서 구동하는 경우 서버의 구글 드라이브 파일을 다운받는다.



import os

#input.zip : https://drive.google.com/open?id=1Cb9kjJ40Sc7hs3TtDREGjdytH7479PKu

#model.h5 : https://drive.google.com/open?id=1CnF0Ailc2s8ob0JieXrhTK4u1YEr_HaD

#model_predict_missing_mask.h5 : https://drive.google.com/open?id=1Sr6D8utBeOEnQ3BUGEdCCwPkYiPfY_uM

#fold_train.zip : https://drive.google.com/open?id=1UqJaSbwpHhTcbM-zdSLljWMMlNnNZG_X



def download_file_gd(file_id, fpathname, unzip=False):

    from google_drive_downloader import GoogleDriveDownloader as gdd

    if os.path.exists(fpathname) == False:

        gdd.download_file_from_google_drive(file_id=file_id, dest_path=fpathname, unzip=unzip, showsize=False)

    else:

        print(fpathname, ": already downloaded")



files = {

    "1Cb9kjJ40Sc7hs3TtDREGjdytH7479PKu" : "./input/severstal-steel-defect-detection/input.zip", 

    "1CnF0Ailc2s8ob0JieXrhTK4u1YEr_HaD" : "./model.h5", 

    "1Sr6D8utBeOEnQ3BUGEdCCwPkYiPfY_uM" : "./input/severstal-steel-defect-detection-data-files/model_predict_missing_mask.h5", 

    "1UqJaSbwpHhTcbM-zdSLljWMMlNnNZG_X" : "./input/severstal-steel-defect-detection-data-files/fold_train.zip",



}



if IN_COLAB:

    for f in files:

        print(f, files[f])

        download_file_gd(file_id=f, fpathname=files[f], unzip=(files[f].find(".zip") >= 0))

        

    # unzip train/test zip file

    import zipfile

    zipfile.ZipFile("./input/severstal-steel-defect-detection/train_images.zip").extractall("./input/severstal-steel-defect-detection/train_images")

    zipfile.ZipFile("./input/severstal-steel-defect-detection/test_images.zip").extractall("./input/severstal-steel-defect-detection/test_images")        



import os

import json

import gc



import cv2

import keras

from keras import backend as K

from keras import layers

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, load_model, Sequential

from keras.layers import Input

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.optimizers import Adam, Nadam

from keras.callbacks import Callback, ModelCheckpoint

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from pathlib import Path

import shutil



INPUT_PATH = "./input"

if IN_COLAB == False:

    INPUT_PATH = "../input"



DF_TRAIN_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection/train.csv")

DF_TEST_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection/sample_submission.csv")



TRAIN_IMAGE_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection/train_images")

TEST_IMAGE_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection/test_images")

DATA_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection-data-files")



GENERATE_WEIGHTS = True



EPOCHS = 30



USE_CALLBACK = True



CHANNELS = 3



K_FOLDS = 4



ASSIGNED_FOLD_JOBS = [x for x in range(K_FOLDS)]

    

if IN_COLAB == False:    

    data_dir_path = "../input/severstal-steel-defect-detection-data-files"

    if os.path.exists(data_dir_path):

        for fname in os.listdir(data_dir_path):

            filepath = os.path.join(data_dir_path, fname)

            print(filepath)

            if os.path.isfile(filepath):

                if GENERATE_WEIGHTS == True:

                    if fname.find("h5") > 0:

                        continue

                destfilepath = os.path.join("./", fname)

                print("copy file ", filepath, " to ", destfilepath)

                shutil.copy(filepath, destfilepath)

                

train_df = pd.read_csv(DF_TRAIN_PATH)

'''

image 파일명과 ClassId가 _로 연결되어 있어서 분리해서 별도 column으로 만든다.

'''

train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

train_df['isNan'] = train_df['EncodedPixels'].isna()

train_df.head()
train_nan_df = train_df.groupby(by='ImageId', axis=0).agg('sum')

train_nan_df.reset_index(inplace=True)

train_nan_df.rename(columns={'isNan' : 'missingCount'}, inplace=True)

train_nan_df['missingCount'] = train_nan_df['missingCount'].astype(np.int32)

train_nan_df['allMissing'] = (train_nan_df['missingCount'] == 4).astype(int)



train_nan_df.head(16)
train_nan_df['missingCount'].hist()

train_nan_df['missingCount'].value_counts()
import ssl

from keras.models import model_from_json

from sklearn.utils import shuffle

from sklearn.model_selection import StratifiedKFold, KFold



def make_fold_csv(input_df):

    #input_df.to_csv("train_nan_df.csv")

    skfold = StratifiedKFold(n_splits=K_FOLDS, random_state=2019, shuffle=True)



    for fold_index, (train_idx, val_idx) in enumerate(skfold.split(X=input_df['ImageId'], y=input_df['allMissing'])):

        

        # train/val 데이터 나눔

        dataframe_train = input_df.iloc[train_idx, :].reset_index()

        dataframe_val = input_df.iloc[val_idx, :].reset_index()

        

        df_train_filename = ("fold_%d_train.csv" % (fold_index))

        df_val_filename = ("fold_%d_val.csv" % (fold_index))



        dataframe_train = shuffle(dataframe_train).reset_index(drop=True)

        dataframe_val = shuffle(dataframe_val).reset_index(drop=True)



        dataframe_train.to_csv(df_train_filename, index=False)

        dataframe_val.to_csv(df_val_filename, index=False)    



#make_fold_csv(train_nan_df)
BATCH_SIZE = 16



INPUT_WIDTH = 800

INPUT_HEIGHT = 128



def get_total_batch(num_samples, batch_size):    

    if (num_samples % batch_size) > 0 :

        return (num_samples // batch_size) + 1

    else :

        return num_samples // batch_size



def create_datagen():

    return ImageDataGenerator(

        zoom_range=0.1,  # set range for random zoom

        # set mode for filling points outside the input boundaries

        fill_mode='constant',

        cval=0.,

        rotation_range=10,

        height_shift_range=0.1,

        width_shift_range=0.1,

        horizontal_flip=True,

        vertical_flip=True,

        rescale=1/255.)



def create_flow(df, dir, datagen):

    return datagen.flow_from_dataframe(

        df,

        directory=dir,

        x_col='ImageId', 

        y_col='allMissing', 

        class_mode='raw',

        target_size=(INPUT_HEIGHT, INPUT_WIDTH),

        batch_size=BATCH_SIZE)

    

    
# https://github.com/titu1994/keras-efficientnets



! pip install keras_efficientnets
from keras_efficientnets import EfficientNetB0, EfficientNetB4



def build_model():

    efficientnet = EfficientNetB4((INPUT_HEIGHT, INPUT_WIDTH, 3), classes=1, include_top=False, weights='imagenet')

    model = Sequential()

    model.add(efficientnet)



    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))

    

    model.compile(

        loss='binary_crossentropy',

        optimizer=Nadam(),

        metrics=['acc'])

    

    return model





# model = build_model()

# model.summary()
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LambdaCallback



from keras.backend import clear_session

import gc



# Reset Keras Session

def clear_memory():

    clear_session()

    for i in range(20):

        gc.collect()





def get_callbacks(model_save_filepath):    

    checkpoint = ModelCheckpoint(

        model_save_filepath,

        monitor='val_acc',

        verbose=1,

        save_best_only=True,

        save_weights_only=False,

        mode='auto')

    

    es = EarlyStopping(monitor='val_acc', min_delta=0, patience = 3, verbose=1, mode='max')

    rl = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 2, min_lr=0.0000001, mode='min')



    return [checkpoint, es, rl]

model = build_model()

model_save_filename = ("model_predict_missing_mask.h5")

model_save_filepath = os.path.join("./", model_save_filename)



data_generator = create_datagen()

train_gen = create_flow(train_nan_df, TRAIN_IMAGE_PATH, data_generator)

val_gen = create_flow(train_nan_df, TRAIN_IMAGE_PATH, data_generator)



history = model.fit_generator(train_gen,

                    validation_data=val_gen, 

                    epochs=10,

                    callbacks=get_callbacks(model_save_filepath))



history_df = pd.DataFrame(history.history)

history_df[['val_loss']].plot()

history_df[['val_acc']].plot()   



clear_memory()
from IPython.display import FileLinks

FileLinks('.') # input argument is specified folder