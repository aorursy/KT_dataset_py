# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50

from keras.models import Sequential, Model

from keras.layers import Dense, Input, Dropout

from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
def summarize_diagnostics(history):

  # plot loss

  plt.subplot(211)

  plt.title('Cross Entropy Loss')

  plt.plot(history.history['loss'], color='blue', label='train')

  plt.plot(history.history['val_loss'], color='orange', label='test')

  # plot accuracy

  plt.subplot(212)

  plt.title('Classification Accuracy')

  plt.plot(history.history['accuracy'], color='blue', label='train')

  plt.plot(history.history['val_accuracy'], color='orange', label='test')

  plt.show() 
datapath = "../input/stanford-car-dataset-by-classes-folder/car_data/car_data/"

!ls $datapath'train'
train_datagen = ImageDataGenerator(

    rescale = 1.0/255.0,

    rotation_range=30,

    horizontal_flip=True,

    validation_split= .25)

train_it = train_datagen.flow_from_directory(

    datapath + 'train', 

    target_size=(256, 256), 

    class_mode='categorical', 

    batch_size=32, 

    shuffle=True,

    subset='training')

val_it = train_datagen.flow_from_directory(

    datapath + 'train', 

    target_size=(256, 256), 

    class_mode='categorical', 

    batch_size=32, 

    shuffle=True,

    subset='validation')
# !curl https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
len(train_it.class_indices)
model = Sequential()

model.add(ResNet50(

    weights = 'imagenet',

    include_top = False,

    pooling = 'avg'

    ))

model.add(Dense(len(train_it.class_indices), activation='softmax'))

model.layers[0].trainable = False
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# hist = model.fit_generator(

#     train_it,

#     steps_per_epoch = len(train_it),

#     validation_data = val_it,

#     validation_steps = len(val_it),

#     epochs = 5

# )
img = Input(shape = (256, 256, 3))

res_model = ResNet50(

    weights = 'imagenet',

    include_top = False, 

    input_tensor = img, 

    input_shape = None, 

    pooling = 'avg'

    )

final_layer = res_model.layers[-1].output

output_layer = Dense(len(train_it.class_indices), activation = 'softmax')(final_layer)

model2 = Model(input = img, output = output_layer)
model2.summary()
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# hist2 = model2.fit_generator(

#     train_it,

#     steps_per_epoch = len(train_it),

#     validation_data = val_it,

#     validation_steps = len(val_it),

#     epochs = 5

# )
train_it.samples/32
img2 = Input(shape = (256, 256, 3))

res_model2 = ResNet50(

    weights = 'imagenet',

    include_top = False, 

    input_tensor = img2, 

    input_shape = None, 

    pooling = 'avg'

    )

final_layer2 = res_model2.layers[-1].output

final_layer2 = Dropout(.2)(final_layer2)

output_layer2 = Dense(len(train_it.class_indices), activation = 'softmax')(final_layer2)

model3 = Model(input = img2, output = output_layer2)
for layer in model3.layers[:-1]:

    layer.trainable = False
model3.summary()
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# hist3 = model3.fit_generator(

#     train_it,

#     steps_per_epoch = len(train_it),

#     validation_data = val_it,

#     validation_steps = len(val_it),

#     epochs = 5

# )
for layer in model.layers:

    print(layer.trainable)
for layer in model2.layers:

    print(layer.trainable)
for layer in model3.layers:

    print(layer.trainable)
# summarize_diagnostics(hist)
# summarize_diagnostics(hist2)
# summarize_diagnostics(hist3)
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from keras.optimizers import SGD
# define cnn model

def define_vgg4model():

  m = Sequential()

  m.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(256, 256, 3)))

  m.add(MaxPooling2D((2, 2)))

  m.add(Dropout(0.2))

  m.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

  m.add(MaxPooling2D((2, 2)))

  m.add(Dropout(0.2))

  m.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

  m.add(MaxPooling2D((2, 2)))

  m.add(Dropout(0.2))

  m.add(Flatten())

  m.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

  m.add(Dense(len(train_it.class_indices), activation='softmax'))

  # compile m

  opt = SGD(lr=0.001, momentum=0.9)

  m.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

  return m
#define model

model4 = define_vgg4model()

model4.summary()
#fit model

# hist4 = model4.fit_generator(train_it, steps_per_epoch=len(train_it),

# 	validation_data=val_it, validation_steps=len(val_it), epochs=20, verbose=1)
# summarize_diagnostics(hist4)
from keras.applications.vgg16 import VGG16
# define cnn model

def define_model():

	# load model

	model = VGG16(include_top=False, input_shape=(256, 256, 3))

	# mark loaded layers as not trainable

	for layer in model.layers:

		layer.trainable = False

	# add new classifier layers

	flat1 = Flatten()(model.layers[-1].output)

	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)

	output = Dense(len(train_it.class_indices), activation='sigmoid')(class1)

	# define new model

	model = Model(inputs=model.inputs, outputs=output)

	# compile model

	opt = SGD(lr=0.001, momentum=0.9)

	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

	return model
model = define_model()

model.summary()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
#fit model

hist5 = model.fit_generator(train_it, steps_per_epoch=len(train_it),

	validation_data=val_it, validation_steps=len(val_it), epochs=5, verbose=1, callbacks=[es])
summarize_diagnostics(hist5)
test_datagen = ImageDataGenerator(rescale = 1.0/255.0)

test_it = test_datagen.flow_from_directory(

    datapath + 'test', 

    target_size=(256, 256), 

    class_mode='categorical', 

    batch_size=32, 

    shuffle=True)
# _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)

# print('> %.3f' % (acc * 100.0))
model6 = ResNet50(

    weights = 'imagenet',

    include_top = False, 

    input_shape = [256, 256, 3]

    )

# mark loaded layers as not trainable

for layer in model.layers:

    layer.trainable = False

    

# final_layer2 = res_model2.layers[-1].output

flat1 = Flatten()(model6.output) 

class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)

output_layer6 = Dense(len(train_it.class_indices), activation = 'softmax',kernel_initializer='he_uniform')(class1)

model6 = Model(input = model6.inputs, output = output_layer6)

# compile model

opt = SGD(lr=0.001, momentum=0.9)

model6.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model6.summary()
# patient early stopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
#fit model

hist6 = model6.fit_generator(train_it, steps_per_epoch=len(train_it),

	validation_data=val_it, validation_steps=len(val_it), epochs=5, verbose=1, callbacks=[es])
summarize_diagnostics(hist6)