# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.
import os
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
test_zip = '/kaggle/input/dogs-vs-cats/test1.zip'
train_zip = '/kaggle/input/dogs-vs-cats/train.zip'
zip_ref = zipfile.ZipFile(test_zip, 'r')
zip_ref.extractall('/kaggle/temp')
zip_ref.close()

zip_ref = zipfile.ZipFile(train_zip, 'r')
zip_ref.extractall('/kaggle/temp')
zip_ref.close()
print(len(os.listdir('/kaggle/temp/train')))
print(len(os.listdir('/kaggle/temp/test1')))
os.listdir("/kaggle/temp")
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

count=0
plt.figure(figsize=(10,10))
for i in np.random.randint(1, 500, size=25):
    count += 1
    image_path = '/kaggle/temp/train/dog.'+ str(i) +'.jpg'
    img = image.load_img(image_path, target_size=(150, 150))
    x = image.img_to_array(img)

    x = tf.expand_dims(x, axis=0)
    x = x / 255.0
    plt.subplot(5, 5, count)
    plt.axis('off')
    plt.imshow(x[0])
orig_train_data_dir = "/kaggle/temp/train"
orig_test_data_dir = "/kaggle/temp/test1"

try:
    base_dir = "/kaggle/Cats_vs_Dogs/"
    os.mkdir(base_dir)

    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)

    train_dogs_dir = os.path.join(train_dir, 'dogs')
    train_cats_dir = os.path.join(train_dir, 'cats')
    os.mkdir(train_dogs_dir)
    os.mkdir(train_cats_dir)

    test_dogs_dir = os.path.join(test_dir, 'dogs')
    test_cats_dir = os.path.join(test_dir, 'cats')
    os.mkdir(test_dogs_dir)
    os.mkdir(test_cats_dir)

    fnames = ['cat.{}.jpg'.format(i) for i in range(6250)] 
    for fname in fnames:
        src = os.path.join(orig_train_data_dir, fname) 
        dst = os.path.join(train_cats_dir, fname) 
        shutil.copyfile(src, dst)
        
    fnames = ['dog.{}.jpg'.format(i) for i in range(6250)] 
    for fname in fnames:
        src = os.path.join(orig_train_data_dir, fname) 
        dst = os.path.join(train_dogs_dir, fname) 
        shutil.copyfile(src, dst)
except:
    pass        
fnames = os.listdir(orig_test_data_dir) 
for fname in fnames:
    src = os.path.join(orig_test_data_dir, fname) 
    dst = os.path.join(test_dir, fname) 
    shutil.copyfile(src, dst)

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir, 
        target_size=(150, 150),
        batch_size=250,
        class_mode = 'binary',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=250,
        class_mode='binary',
        subset='validation')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=250)
for data_batch, labels_batch in train_generator:
    print(data_batch.shape)
    print(labels_batch.shape)
    break
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.applications import Xception


conv_base = Xception(weights="/kaggle/input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False, input_shape=(150, 150, 3))
conv_base.summary()
batch_size = 250

def extract_features(generator, sample_count):
    features = np.zeros(shape=(sample_count, 5, 5, 2048))
    labels = np.zeros(shape=(sample_count))
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i+1) * batch_size] = features_batch
        labels[i * batch_size : (i+1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_generator, 10000) 
validation_features, validation_labels = extract_features(validation_generator, 2500)
from tensorflow.keras.optimizers import Adam

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.999):
            self.model.stop_training = True
            
callback = myCallback()

model = models.Sequential([
    layers.Flatten(input_shape=(5, 5, 2048)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-5),metrics=['accuracy'])
history = model.fit(train_features, train_labels, epochs=50, batch_size=batch_size, validation_data=(validation_features, validation_labels),callbacks=[callback])
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()

plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
test_features = conv_base.predict(test_generator)
pred = model.predict(test_features)
ans=[]
for i in range(pred.shape[0]):
    if pred[i] > 0.5:
        ans.append(1)
    else:
        ans.append(0)
