# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.applications import MobileNetV2

from keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input

from keras.models import Model

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping

import cv2
def visualize_dataset(dataset_paths):

    cols = 4

    rows = 3

    fig = plt.figure(figsize=(10, 8))

    plt.rcParams.update({"font.size": 14})

    grid = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig)



    np.random.seed(45)

    rand = np.random.randint(0, len(dataset_paths), size=(rows * cols))



    for i in range(rows * cols):

        fig.add_subplot(grid[i])

        lable = dataset_paths[rand[i]].split(os.path.sep)[-2]

        plt.title('"{:s}"'.format(lable))

        plt.axis(False)

        img = load_img(dataset_paths[rand[i]])

        plt.imshow(img)



    plt.show()



dataset_paths = glob.glob("/kaggle/input/ocr-local-dataset/dataset_characters_local/**/*.jpg")

visualize_dataset(dataset_paths)
def preprocessing(dataset_paths):

    X = []

    labels = []



    for image_path in tqdm(dataset_paths):

        label = image_path.split(os.path.sep)[-2]

        img = load_img(image_path, target_size=(128, 128))

        img = img_to_array(img)



        X.append(img)

        labels.append(label)



    X = np.array(X, dtype="float16")

    labels = np.array(labels)



    ### One-hot encoding on labels

    le = LabelEncoder()

    le.fit(labels)

    labels = le.transform(labels)

    y = to_categorical(labels)



    ### Save label file

    np.save('license_character_classes_v2.npy', le.classes_)



    ### Seperate train(90%) and test(10%) dataset

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)



    ### Initialize data augmentation

    image_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, fill_mode="nearest")



    return train_x, train_y, test_x, test_y, image_gen, y



train_x, train_y, test_x, test_y, image_gen, y = preprocessing(dataset_paths)
def model(output_shape, lr=1e-4, decay=1e-4/25, train=False):

    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(128, 128, 3)))



    headModel = baseModel.output

    headModel = AveragePooling2D(pool_size=(3, 3))(headModel)

    headModel = Flatten(name="flatten")(headModel)

    headModel = Dense(128, activation="relu")(headModel)

    headModel = Dropout(0.5)(headModel)

    headModel = Dense(output_shape, activation="softmax")(headModel)



    model = Model(inputs=baseModel.input, outputs=headModel)



    ### Create model

    if train:

        ### define trainable layer

        for layer in baseModel.layers:

            layer.trainable = True



        ### Compile model

        optimizer = Adam(lr=lr, decay=decay)

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])



    return model
INIT_LR = 1e-4

EPOCHS = 80

ocr_model = model(y.shape[1], lr=INIT_LR, decay=INIT_LR/EPOCHS,train=True)
BATCH_SIZE = 64



my_checkpointer = [EarlyStopping(monitor='val_loss', patience=5, verbose=0), ModelCheckpoint(filepath="License_character_recognition_weight_v2.h5", verbose=1, save_weights_only=True)]



ocr_model_data = ocr_model.fit(

    image_gen.flow(train_x, train_y, batch_size=BATCH_SIZE),

    steps_per_epoch=len(train_x) // BATCH_SIZE,

    validation_data=(test_x, test_y),

    validation_steps=len(test_x) // BATCH_SIZE,

    epochs=EPOCHS,

    callbacks=my_checkpointer)
def visualize_metrics(model_data):

    fig = plt.figure(figsize=(14, 5))

    grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    fig.add_subplot(grid[0])

    plt.plot(model_data.history['accuracy'], label='training accuracy')

    plt.plot(model_data.history['val_accuracy'], label='test accuracy')

    plt.title('Accuracy')

    plt.xlabel('epochs')

    plt.ylabel('accuracy')

    plt.legend()



    fig.add_subplot(grid[1])

    plt.plot(model_data.history['loss'], label='training loss')

    plt.plot(model_data.history['val_loss'], label='test loss')

    plt.title('Loss')

    plt.xlabel('epochs')

    plt.ylabel('loss')

    plt.legend()



    plt.show()
visualize_metrics(ocr_model_data)
model_json = ocr_model.to_json()

with open("MobileNets_character_recognition_v2.json", "w") as json_file:

    json_file.write(model_json)