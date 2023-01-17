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
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
train_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/train'
validation_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/val'
test_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/test'
from tensorflow.keras.applications.densenet import preprocess_input
BATCH_SIZE = 64
IMG_SHAPE  = 100


image_gen_train = ImageDataGenerator(preprocessing_function=preprocess_input)
# image_gen_train = ImageDataGenerator(rescale=1./255)

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),
                                                     class_mode='binary')


test_image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
# test_image_generator = ImageDataGenerator(rescale=1./255) 

test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=test_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE,IMG_SHAPE), 
                                                              class_mode='binary')


validation_image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
# validation_image_generator = ImageDataGenerator(rescale=1./255)  

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE,IMG_SHAPE), 
                                                              class_mode='binary')
# names =  {0: "NORMAL", 1: "PNEUMONIA"}
# train_images, train_labels = next(train_data_gen)

# plt.figure(figsize = (5,5))
# plt.imshow(train_images[3], cmap='gray')
# plt.title(f"Real label: {names[train_labels[3]]}")

# plt.figure(figsize = (5,5))
# plt.imshow(train_images[1], cmap='gray')
# plt.title(f"Real label: {names[train_labels[1]]}")
cpt_filename = 'checkpoint_best.hdf5'

mcp = tf.keras.callbacks.ModelCheckpoint(filepath=cpt_filename, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
input_tensor = tf.keras.Input(shape=(IMG_SHAPE,IMG_SHAPE,3))

base_model = tf.keras.applications.DenseNet201(
                                            input_tensor = input_tensor, 
                                            include_top = False, 
                                            pooling = 'average'
                                            )
model = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1920, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
            ])

model.compile(
              optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
%time

EPOCHS = 100

history = model.fit(
                    train_data_gen,
                    validation_data=test_data_gen,
                    epochs = EPOCHS,   
                    callbacks=[mcp]
                    )
def plotLearningCurve(history,epochs):
    epochRange = range(1,epochs+1)
    plt.plot(epochRange,history.history['accuracy'])
    plt.plot(epochRange,history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Validation'],loc='upper left')
    plt.show()

    plt.plot(epochRange,history.history['loss'])
    plt.plot(epochRange,history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train','Validation'],loc='upper left')
    plt.show()
plotLearningCurve(history, EPOCHS)
model_eval = tf.keras.models.load_model(cpt_filename)

# результат предсказания в процентах
print("Loss val_data_gen - " , model_eval.evaluate_generator(val_data_gen)[0])
print("Accuracy val_data_gen - " , model_eval.evaluate_generator(val_data_gen)[1]*100 , "%")

print("Loss test_data_gen - " , model_eval.evaluate_generator(test_data_gen)[0])
print("Accuracy test_data_gen - " , model_eval.evaluate_generator(test_data_gen)[1]*100 , "%")
image_gen_train = ImageDataGenerator(
                                      preprocessing_function=preprocess_input,
                                      featurewise_center=False,  # set input mean to 0 over the dataset
                                      samplewise_center=False,   # set each sample mean to 0
                                      featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                      samplewise_std_normalization=False,   # divide each input by its std
                                      zca_whitening=False,      # apply ZCA whitening
                                      rotation_range = 30,      # randomly rotate images in the range (degrees, 0 to 180) / максимальный угол поворота
                                      zoom_range = 0.2,         # Randomly zoom image / картинка будет увеличена или уменьшена не более чем на 20% 
                                      width_shift_range=0.1,    # randomly shift images horizontally (fraction of total width) / смещение максимум на 20% ширины по горизонтали
                                      height_shift_range=0.1,   # randomly shift images vertically (fraction of total height) / смещение максимум на 20% высоты по вертикали
                                      horizontal_flip = True,   # randomly flip images / случайное отражение по горизонтали
                                      vertical_flip=False,      # randomly flip images / случайное отражение по вертикали
                                      fill_mode="nearest",       # чем заполнять пробелы -- сначала выберем черный цвет, а потом изменим на "nearest"
                                      )     

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),
                                                     class_mode='binary')
model_DD = model_eval
%time

EPOCHS = 100

history = model_DD.fit(
                        train_data_gen,
                        validation_data=test_data_gen,
                        epochs = EPOCHS,   
                        callbacks=[mcp]
                        )
model_eval = tf.keras.models.load_model(cpt_filename)

# результат предсказания в процентах
print("Loss val_data_gen - " , model_eval.evaluate_generator(val_data_gen)[0])
print("Accuracy val_data_gen - " , model_eval.evaluate_generator(val_data_gen)[1]*100 , "%")

print("Loss test_data_gen - " , model_eval.evaluate_generator(test_data_gen)[0])
print("Accuracy test_data_gen - " , model_eval.evaluate_generator(test_data_gen)[1]*100 , "%")
def show_PNEUMONIA_NORMAL(images, labels, predicted_labels=None):
    names =  {0: "NORMAL", 1: "PNEUMONIA"}   #train_data_gen.class_indices
    plt.figure(figsize=(15,15))
    for i in range(16):
        plt.subplot(4,4, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.gray)
        if predicted_labels is not None:
            title_obj = plt.title(f"Real: {names[labels[i]]}.\n Pred: {names[predicted_labels[i]]}")
            if labels[i] != predicted_labels[i]:
                plt.setp(title_obj, color='r')
        else:
            plt.title(f"Real label: {names[labels[i]]}")
sample_val_images, sample_val_labels = next(val_data_gen)
predicted = model_eval.predict_classes(sample_val_images).flatten()
show_PNEUMONIA_NORMAL(sample_val_images, sample_val_labels, predicted)