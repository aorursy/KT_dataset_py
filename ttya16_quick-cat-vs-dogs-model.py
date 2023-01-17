import numpy as np

import pandas as pd





from tensorflow import keras

from tensorflow.keras.applications import DenseNet121

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import zipfile





zip_files = ['test1', 'train']

# Will unzip the files so that you can see them..

for zip_file in zip_files:

    with zipfile.ZipFile("../input/dogs-vs-cats/{}.zip".format(zip_file),"r") as z:

        z.extractall(".")

        print("{} unzipped".format(zip_file))
TRAIN_IMG_LIST = os.listdir('../working/train')

TEST_IMG_LIST = os.listdir('../working/test1')

print(len(TRAIN_IMG_LIST))

print(len(TEST_IMG_LIST))
IMG_SIZE = 200



import cv2

img_idx = 1

img_path = '../working/train/' + str(TRAIN_IMG_LIST[img_idx])

img_label = TRAIN_IMG_LIST[img_idx][:3]

sample_img = cv2.imread(img_path)

sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

sample_img = cv2.resize(sample_img, (IMG_SIZE, IMG_SIZE))

print(sample_img.shape)

plt.imshow(sample_img)

plt.title(img_label)

plt.show()
train_df = pd.DataFrame()



imgpaths = []

labels = []



for imgfile in TRAIN_IMG_LIST:

    label =imgfile.split('.')[0]

    full_path = '../working/train/' + imgfile

    

    imgpaths.append(full_path)

    labels.append(label)

    

train_df['image_path'] = imgpaths

train_df['label'] = labels



train_df.head()
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

val_df.shape
#MODEL

densenet = DenseNet121(

    weights = '../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top = False,

    input_shape = (IMG_SIZE, IMG_SIZE, 3)



)



model = keras.Sequential()

# model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))

# model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

# model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

# model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

# model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))

# model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

# model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))

# model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))



model.add(densenet)



model.add(keras.layers.GlobalAveragePooling2D())

model.add(keras.layers.Dense(256, activation='relu'))

model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(2, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

    rotation_range=10,

    rescale = 1./255.,

    horizontal_flip = True,

    width_shift_range = 0.1,

    height_shift_range = 0.1

)



train_datagenerator = train_datagen.flow_from_dataframe(

    dataframe = train_df,

    x_col = 'image_path',

    y_col = 'label',

    target_size = (IMG_SIZE, IMG_SIZE),

    class_mode = 'categorical',

    batch_size = 100

)
val_datagen = ImageDataGenerator(

    rescale = 1./255.

)



val_datagenerator = val_datagen.flow_from_dataframe(

    dataframe = val_df,

    x_col = 'image_path',

    y_col = 'label',

    target_size = (IMG_SIZE, IMG_SIZE),

    class_mode = 'categorical',

    batch_size = 100

)
history = model.fit_generator(train_datagenerator,

                              epochs = 30,

                              validation_data = val_datagenerator,

                              validation_steps = val_df.shape[0]//100,

                              steps_per_epoch = train_df.shape[0]//100

                             )
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
train_datagenerator.class_indices
model.save('catvsdog_model.h5')