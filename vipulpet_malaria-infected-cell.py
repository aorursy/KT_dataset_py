import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
data = '/kaggle/input/cell-images-for-detecting-malaria/cell_images'
!pip install split-folders
import split_folders
split_folders.ratio(data, output="", seed=1337, ratio=(.9, .1))
os.rmdir('/kaggle/working/train/cell_images')
os.rmdir('/kaggle/working/val/cell_images')
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_path = '/kaggle/working/train'
test_path = '/kaggle/working/val'
infected = '/Parasitized/'
uninfected = '/Uninfected/'
image_gen = ImageDataGenerator(rotation_range=20,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              zoom_range=0.1,
                              shear_range=0.1,
                              horizontal_flip=True,
                              fill_mode='nearest')
ran = image_gen.random_transform(imread(train_path+infected+os.listdir(train_path+infected)[0]))
plt.imshow(ran)
image_gen.flow_from_directory(train_path)
batch_size = 16
image_shape = (130, 130, 3)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
          
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
early_stop = EarlyStopping(monitor = 'val_loss', patience=3)
train_img_gen = image_gen.flow_from_directory(train_path,
                             target_size=image_shape[:2],
                             color_mode='rgb',
                             batch_size=batch_size,
                             class_mode='binary')

test_img_gen = image_gen.flow_from_directory(test_path,
                             target_size=image_shape[:2],
                             color_mode='rgb',
                             batch_size=batch_size,
                             class_mode='binary',
                             shuffle=False)
train_img_gen.class_indices
results = model.fit_generator(train_img_gen, epochs=20,
                              validation_data=test_img_gen,
                             callbacks=[early_stop])
model.save("model")
his = pd.DataFrame(model.history.history)
his
his[['accuracy','val_accuracy']].plot()
his[['loss','val_loss']].plot()
model.evaluate_generator(test_img_gen)
model.metrics_names
pred = model.predict_generator(test_img_gen)
predict = pred >= 0.4
predict
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing import image
classification_report(test_img_gen.classes, predict)
confusion_matrix(test_img_gen.classes, predict)
im = test_path+infected+os.listdir(test_path+infected)[500]
im = image.load_img(im)
im
im_arr = image.img_to_array(im)
im_arr = np.expand_dims(im_arr, axis=0)
im_arr.shape
model.predict(im_arr)
