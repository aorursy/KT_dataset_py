import matplotlib.pyplot as plt

from matplotlib.image import imread 



def show_rice_image(image_name):

    filename = '/kaggle/input/three-rice-species/New_Rice_Dataset/train/' + image_name

    # load image pixels

    image = imread(filename)

    plt.figure(figsize=(8,8))

    # plot raw pixel data

    plt.imshow(image)

    # show the figure

    plt.show()
show_rice_image('Tai-Nong No.81/Tai-Nong No.81_55.jpg')
show_rice_image('Tai-Nong No.82/Tai-Nong No.82_35.jpg')
show_rice_image('Tai-Nong No.83/Tai-Nong No.83_45.jpg')
train_dir = '/kaggle/input/three-rice-species/New_Rice_Dataset/train'

validation_dir = '/kaggle/input/three-rice-species/New_Rice_Dataset/validation'

test_dir = '/kaggle/input/three-rice-species/New_Rice_Dataset/test'
from keras.applications import VGG16



rice_conv_base = VGG16(weights='imagenet',

                       include_top=False,

                       input_shape=(224, 224, 3))
rice_conv_base.summary()
rice_conv_base.trainable = True



set_trainable = False

for layer in rice_conv_base.layers:

    if layer.name == 'block5_conv1':

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False
from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Flatten())

# 以 relu 函數作為 activation function

model.add(layers.Dense(2048, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(3, activation='softmax'))
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-5), 

              # learning rate 設為 0.00001

              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator

import os

import numpy as np



# 以下設定資料擴增的方法：含隨機圖片旋轉(0到180度)、左右平移、水平與鉛直翻轉

train_datagen = ImageDataGenerator(

    rotation_range=180,

    #width_shift_range = 0.2,

    #height_shift_range = 0.2,

    horizontal_flip=True,

    vertical_flip=True)



batch_size = 20



# 以下定義如何將訓練集的影像轉換為張量

def train_extract_features(directory, sample_count):

    features = np.zeros(shape=(sample_count, 7,7,512))

    labels = np.zeros(shape=(sample_count))

    generator = train_datagen.flow_from_directory(

        directory,

        target_size=(224, 224),

        batch_size=batch_size,

        class_mode='binary')

    i = 0

    for inputs_batch, labels_batch in generator:

        features_batch = rice_conv_base.predict(inputs_batch)

        features[i * batch_size : (i + 1) * batch_size] = features_batch

        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i += 1

        if i * batch_size >= sample_count:

            break

    return features, labels



datagen = ImageDataGenerator()



# 以下定義如何將驗證集、測試集的影像轉換為張量

def extract_features(directory, sample_count):

    features = np.zeros(shape=(sample_count, 7,7,512))

    labels = np.zeros(shape=(sample_count))

    generator = datagen.flow_from_directory(

        directory,

        target_size=(224, 224),

        batch_size=batch_size,

        class_mode='binary')

    i = 0

    for inputs_batch, labels_batch in generator:

        features_batch = rice_conv_base.predict(inputs_batch)

        features[i * batch_size : (i + 1) * batch_size] = features_batch

        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i += 1

        if i * batch_size >= sample_count:

            break

    return features, labels
train_features, train_labels = train_extract_features(train_dir, 1200)

validation_features, validation_labels = extract_features(validation_dir, 150)

test_features, test_labels = extract_features(test_dir, 150)



train_features = np.reshape(train_features, (1200, 7,7,512))

validation_features = np.reshape(validation_features, (150, 7,7,512))

test_features = np.reshape(test_features, (150, 7,7,512))
new_validation_labels = validation_labels

new_train_labels = train_labels

new_test_labels = test_labels
from keras.utils import to_categorical



new_validation_labels = to_categorical(new_validation_labels)

new_train_labels = to_categorical(new_train_labels)

new_test_labels = to_categorical(new_test_labels)
history = model.fit(

    train_features, new_train_labels,

    epochs=50,

    batch_size=50,

    validation_data=(validation_features, new_validation_labels))
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'b', label='Training accuracy')

plt.plot(epochs, val_acc, 'r', label='Validation accuracy')

plt.title('Training and validation accuracy for 3 rice species')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss for 3 rice species')

plt.legend()



plt.show()
def smooth_curve(points, factor=0.8):

    smoothed_points = []

    for point in points:

        if smoothed_points:

            previous = smoothed_points[-1]

            smoothed_points.append(previous * factor + point * (1 - factor))

        else:

            smoothed_points.append(point)

    return smoothed_points



plt.plot(epochs,

         smooth_curve(acc), 'b', label='Smoothed training accuracy')

plt.plot(epochs,

         smooth_curve(val_acc), 'r', label='Smoothed validation accuracy')

plt.title('Training and validation accuracy for 3 rice species')

plt.legend()



plt.figure()



plt.plot(epochs,

         smooth_curve(loss), 'b', label='Smoothed training loss')

plt.plot(epochs,

         smooth_curve(val_loss), 'r', label='Smoothed validation loss')

plt.title('Training and validation loss for 3 rice species')

plt.legend()



plt.show()
score, acc = model.evaluate(test_features, new_test_labels, 

                            batch_size=20, verbose = 1)

print('測試的準確率:', acc*100)