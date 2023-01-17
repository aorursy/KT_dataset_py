from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
from PIL import ImageFile
import random
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
ImageFile.LOAD_TRUNCATED_FILES = True
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

filenames = os.listdir("../input/nnfl-lab-1/training/training/")
categories = []
for filename in filenames:
    category = filename.split('_')[0]
    if(category == 'chair'):
        categories.append(0)
    if(category == 'kitchen'):
        categories.append(1)
    if(category == 'knife'):
        categories.append(2)
    if(category == 'saucepan'):
        categories.append(3)
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
IMG_SIZE = 256
from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()
# 1st Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=(3,3),padding="Same",activation="relu" , input_shape = (IMG_SIZE,IMG_SIZE,3)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
# 2nd Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(3,3),padding="Same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(filters=128, kernel_size=(3,3),padding="Same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
# 4th Convolutional Layer
model.add(Conv2D(filters=256,kernel_size = (3,3),padding="Same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
# 5th Convolutional Layer
model.add(Conv2D(filters=512,kernel_size = (3,3),padding="Same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
#6th Convolutional Layer
model.add(Conv2D(filters=1024,kernel_size = (3,3),padding="Same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.5))
model.add(BatchNormalization())
# Add output layer
model.add(Dense(4,activation="softmax"))

model.summary() # print summary my model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy']) #compile model

df["category"] = df["category"].replace({0: 'chair', 1: 'kitchen', 2: 'knife', 3: 'saucepan'}) 
df.head()
train_df, validate_df = train_test_split(df, test_size=0.25, random_state=69, stratify = df['category'])
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
train_df
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "../input/nnfl-lab-1/training/training/", 
    x_col='filename',
    y_col='category',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='categorical',
    batch_size=64
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "../input/nnfl-lab-1/training/training/", 
    x_col='filename',
    y_col='category',
    target_size=(IMG_SIZE,  IMG_SIZE),
    class_mode='categorical',
    batch_size=64
)
validation_generator
epochs=33
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
test_filenames = os.listdir("../input/nnfl-lab-1/testing/testing/")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "../input/nnfl-lab-1/testing/testing/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=64,
    shuffle=False
)
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/64))
test_df['category'] = np.argmax(predict, axis=-1)

submission_df = test_df.copy()
submission_df['id'] = submission_df['filename']
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)
from IPython.display import HTML
import pandas as pd 
import numpy as np
import base64 
def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 
    csv = df.to_csv(index=False) 
    b64 = base64.b64encode(csv.encode()) 
    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(submission_df)
model.save_weights("model.h5")
