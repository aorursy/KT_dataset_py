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
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
batch_size = 40
img_height = 200
img_width = 200
training_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/train',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size

)
validation_ds =  tf.keras.preprocessing.image_dataset_from_directory(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/val',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size)
testing_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/test',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size)
class_names = training_ds.class_names
plt.figure(figsize=(10, 10))
for images, labels in training_ds.take(1):
  for i in range(12):
    ax = plt.subplot(3, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title('train_set: '+class_names[labels[i]])
    plt.grid(True)
AUTOTUNE = tf.data.experimental.AUTOTUNE
training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)
testing_ds = testing_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
## Defining Cnn
MyCnn = tf.keras.models.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(2, activation= 'softmax')
])
MyCnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
## lets train our CNN
retVal = MyCnn.fit(training_ds,validation_data= validation_ds,epochs = 5)
plt.plot(retVal.history['loss'], label = 'training loss')
plt.plot(retVal.history['accuracy'], label = 'training accuracy')
plt.legend()
AccuracyVector = []
plt.figure(figsize=(20, 20))
for images, labels in testing_ds.take(1):
    predictions = MyCnn.predict(images)
    predlabel = []
    prdlbl = []
    
    for mem in predictions:
        predlabel.append(class_names[np.argmax(mem)])
        prdlbl.append(np.argmax(mem))
    
    AccuracyVector = np.array(prdlbl) == labels
    for i in range(40):
        ax = plt.subplot(10, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title('Pred: '+ predlabel[i]+' actl:'+class_names[labels[i]] )
        plt.axis('off')
        plt.grid(True)
## this is for retriving labels form batched testing_ds
y = np.concatenate([y for x, y in testing_ds], axis=0)
PredictionsOnEntireTestingSet = MyCnn.predict(testing_ds)
vect = []
for mem in PredictionsOnEntireTestingSet:
    vect.append(np.argmax(mem))
AccuracyVector = (np.array(y) == np.array(vect))

matchings = 0
for mem in AccuracyVector:
    if  mem == True:
        matchings = matchings + 1 

print ('Accuracy on training set: {}'.format(100 *matchings/len(y)))