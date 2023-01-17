import numpy as np 

import pandas as pd 

from glob import glob 

from skimage.io import imread 

import os

import shutil

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input, Concatenate, GlobalMaxPooling2D

from keras.models import Model

from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint

from keras.optimizers import Adam

!pip install livelossplot

from livelossplot import PlotLossesKeras

from keras.applications.vgg19 import VGG19

from keras.applications.vgg16 import VGG16
TRAINING_LOGS_FILE = "training_logs.csv"

MODEL_FILE = "histopathologic_cancer_detector.h5"

ROC_PLOT_FILE = "roc.png"

KAGGLE_SUBMISSION_FILE = "kaggle_submission.csv"

INPUT_DIR = '../input/'
SAMPLE_COUNT = 50000

TRAINING_RATIO = 0.9

IMAGE_SIZE = 96

EPOCHS = 12

BATCH_SIZE = 128

VERBOSITY = 1

TESTING_BATCH_SIZE = 1000
# Data setup

training_dir = INPUT_DIR + 'train/'

data_frame = pd.DataFrame({'path': glob(os.path.join(training_dir,'*.tif'))})

data_frame['id'] = data_frame.path.map(lambda x: x.split('/')[3].split('.')[0]) 

labels = pd.read_csv(INPUT_DIR + 'train_labels.csv')

data_frame = data_frame.merge(labels, on = 'id')

negatives = data_frame[data_frame.label == 0].sample(SAMPLE_COUNT)

positives = data_frame[data_frame.label == 1].sample(SAMPLE_COUNT)

data_frame = pd.concat([negatives, positives]).reset_index()

data_frame = data_frame[['path', 'id', 'label']]

data_frame['image'] = data_frame['path'].map(imread)



training_path = '../training1'

validation_path = '../validation1'



for folder in [training_path, validation_path]:

    for subfolder in ['0', '1']:

        path = os.path.join(folder, subfolder)

        os.makedirs(path, exist_ok=True)



training, validation = train_test_split(data_frame, train_size=TRAINING_RATIO, stratify=data_frame['label'])



data_frame.set_index('id', inplace=True)



for images_and_path in [(training, training_path), (validation, validation_path)]:

    images = images_and_path[0]

    path = images_and_path[1]

    for image in images['id'].values:

        file_name = image + '.tif'

        label = str(data_frame.loc[image,'label'])

        destination = os.path.join(path, label, file_name)

        if not os.path.exists(destination):

            source = os.path.join(INPUT_DIR + 'train', file_name)

            shutil.copyfile(source, destination)
# Augmentation of data

training_data_generator = ImageDataGenerator(rescale=1./255,

                                             

                                             horizontal_flip=True,

                                             vertical_flip=True,

                                             rotation_range=180,

                                             zoom_range=[1, 1.5], 

                                             width_shift_range=0.3,

                                             height_shift_range=0.3,

                                             shear_range=0.3,

channel_shift_range=0.3)
training_generator = training_data_generator.flow_from_directory(training_path,

                                                                 target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                                                 batch_size=BATCH_SIZE,

                                                                 class_mode='binary')

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_path,

                                                                              target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                                                              batch_size=BATCH_SIZE,

                                                                              class_mode='binary')
#model

input_shape = (96, 96, 3)

inputs = Input(input_shape)

v16 = VGG16(include_top=False, input_shape=input_shape)

v19 = VGG19(include_top=False, input_shape=input_shape) 

outputs = Concatenate(axis=-1)([GlobalAveragePooling2D()(v16(inputs)),GlobalAveragePooling2D()(v19(inputs))])

#outputs = GlobalAveragePooling2D()(dense(inputs))

outputs = Dropout(0.5)(outputs)

outputs = Dense(1, activation='sigmoid')(outputs)



model = Model(inputs, outputs)

model.compile(optimizer=Adam(lr=0.0001, decay=0.00001),

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()        
#  Training

history = model.fit_generator(training_generator,

                              steps_per_epoch=len(training_generator), 

                              validation_data=validation_generator,

                              validation_steps=len(validation_generator),

                              epochs=EPOCHS,

                              verbose=VERBOSITY,shuffle=True,

                              callbacks=[PlotLossesKeras(),

                                         ModelCheckpoint(MODEL_FILE,

                                                         monitor='val_acc',

                                                         verbose=VERBOSITY,

                                                         save_best_only=True,

                                                         mode='max'),

                                         CSVLogger(TRAINING_LOGS_FILE,

                                                   append=False,

                                                   separator=';')])

model.load_weights(MODEL_FILE)
#Test on validation data

roc_validation_generator = ImageDataGenerator(rescale=1./255,shear_range=0.3,zoom_range=[1,1.5],

        horizontal_flip=True,

        rotation_range=10., 

        width_shift_range = 0.3, 

        height_shift_range = 0.3).flow_from_directory(validation_path,target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                                                                  batch_size=BATCH_SIZE,

                                                                                  class_mode='binary',

                                                                                  shuffle=False)

predictions = model.predict_generator(roc_validation_generator, steps=len(roc_validation_generator), verbose=VERBOSITY)

false_positive_rate, true_positive_rate, threshold = roc_curve(roc_validation_generator.classes, predictions)

area_under_curve = auc(false_positive_rate, true_positive_rate)



plt.plot([0, 1], [0, 1], 'k--')

plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.3f}'.format(area_under_curve))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.savefig(ROC_PLOT_FILE, bbox_inches='tight')

plt.show()
# Kaggle testing

tta_steps = 5

from tqdm import tqdm

test_datagen = ImageDataGenerator(

        shear_range=0.1,

        zoom_range=[1,1.5],

        horizontal_flip=True,

        rotation_range=10.,

        fill_mode='reflect', 

        width_shift_range = 0.1, 

        height_shift_range = 0.1)

testing_files = glob(os.path.join(INPUT_DIR+'test/','*.tif'))

submission = pd.DataFrame()

for index in range(0, len(testing_files), TESTING_BATCH_SIZE):

    predicted_labels=[]

    data_frame = pd.DataFrame({'path': testing_files[index:index+TESTING_BATCH_SIZE]})

    data_frame['id'] = data_frame.path.map(lambda x: x.split('/')[3].split(".")[0])

    data_frame['image'] = data_frame['path'].map(imread)

    if(index==57000):

        TESTING_BATCH_SIZE=548;

    images = np.stack(data_frame.image, axis=0)

    for i in tqdm(range(tta_steps)):

        preds = model.predict_generator(test_datagen.flow(images/255, batch_size=TESTING_BATCH_SIZE, shuffle=False), steps = len(images)/TESTING_BATCH_SIZE)

        predicted_labels.append(preds)

    predictions = np.max(predicted_labels,axis=0)

    data_frame['label'] = predictions

    submission = pd.concat([submission, data_frame[["id", "label"]]])

submission.to_csv(KAGGLE_SUBMISSION_FILE, index=False, header=True)