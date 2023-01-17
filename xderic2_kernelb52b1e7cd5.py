import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Input, Activation, Dropout, Flatten, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import os
from fastai.vision import *
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
source_dir = Path('../input/100-bird-species/')
source_dir.ls()
train_dir = os.path.join(source_dir, 'train')
valid_dir = os.path.join(source_dir, 'valid')
test_dir = os.path.join(source_dir, 'test')

img_src = (ImageList.from_folder(source_dir)
                .split_by_folder(train='train', valid='valid')
                .label_from_folder()
                .add_test_folder('test')
                .transform(get_transforms(), size=224))

bird_data = img_src.databunch(bs=32).normalize(imagenet_stats)
bird_data.show_batch()
data_gen = ImageDataGenerator(rescale=1./255, )
train_data = data_gen.flow_from_directory(train_dir, target_size=(224,224))
valid_data = data_gen.flow_from_directory(valid_dir, target_size=(224,224))
test_data = data_gen.flow_from_directory(test_dir, target_size=(224,224))
backend.clear_session()
conv_base = InceptionV3 (weights = 'imagenet', 
                  include_top = False,
                  input_shape = (224, 224, 3))
conv_base.trainable = False

model = Sequential()
model.add(conv_base)
model.add(BatchNormalization())
model.add(Dropout(0.02))
model.add(keras.layers.Flatten())
model.add(Activation('relu'))
model.add(Dense(190))
model.add(Activation('softmax'))
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.001, decay=1e-6, momentum=0.9),
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])
history = model.fit_generator(
    train_data,
    steps_per_epoch=766,
    epochs=50,
    validation_data=valid_data,
    validation_steps=29,
    verbose = 1,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience = 4, restore_best_weights = True)])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss values vs epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()