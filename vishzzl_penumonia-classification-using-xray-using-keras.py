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
train_datagen= ImageDataGenerator(rescale=1./255)
train_dr=r"/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train"
train_generator=train_datagen.flow_from_directory(
    train_dr,
    target_size=(300,300),
    batch_size=128,
    class_mode='binary')
test_datagen= ImageDataGenerator(rescale=1./255)
test_dr=r"/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test"
test_genrator=test_datagen.flow_from_directory(
    test_dr,
    target_size=(300,300),
    batch_size=128,
    class_mode='binary')
val_datagen= ImageDataGenerator(rescale=1./255)
val_dr=r"/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val"
val_genrator=val_datagen.flow_from_directory(
    val_dr,
    target_size=(300,300),
    batch_size=128,
    class_mode='binary')
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dense(1,activation='sigmoid')
    
])
model.summary()
from tensorflow.keras.optimizers import RMSprop
model.compile(loss="binary_crossentropy",
             optimizer=RMSprop(lr=0.001),
             metrics=['acc'])
history = model.fit_generator(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,
      validation_data = val_genrator,
      validation_steps=5)
import matplotlib.pyplot as plt
plt.imshow(train_generator[0][0][0])
plt.imshow(train_generator[1][0][0])
plt.show()
import numpy as np
from keras.preprocessing import image
path=r"../input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0011-0001-0002.jpeg"
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0]>0.5:
    print(  " has pneumonia ")
else:
    print( " does not have pneumonia")
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
    
 
  # predicting images
    path = '/content/' + fn
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0]>0.5:
        print(fn + " has pneumonia ")
    else:
        print(fn + " does not have pneumonia")
 
