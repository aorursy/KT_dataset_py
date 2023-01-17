# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input

from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate

from tensorflow.keras.layers import LeakyReLU, ELU, Add, ReLU

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import utils

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
def get_cb(esc = False, mpc = False, tbc = False, rpl=False, model_fname = 'model_v0.0.1', wait_for_tf_cb=15):

    cb = []

    if esc:

        cb.append(EarlyStopping(monitor='val_acc', patience=5, verbose=1))

    if mpc:

        cb.append(ModelCheckpoint(filepath=model_fname +'.best.hdf5', save_best_only=True, monitor='val_acc', mode='max', verbose=1))

    if tbc:

        cb.append(TensorBoardColab(startup_waiting_time=wait_for_tf_cb))

    if rpl:

        cb.append(ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, verbose=1, mode='max', min_lr=0.00000001))



    return cb
model_fname = 'myConv_model'



# Гиперпараметры

NB_CLASSES = 10

IMAGE_SIZE = 32

IMAGE_CHANNELS = 3



TRAIN_BATCH_SIZE = 64

VALID_BATCH_SIZE = 32

EPOCHS = 100

VALID_SPLIT = 0.1



# ширина

k = 16

# глубина

N = 4



# Названия классов из набора данных CIFAR-10

classes = ['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']
data = np.load('../input/train.npz')

x_train = data['x']

y_train = data['y']



x_test = np.load('../input/test.npy')



y_train = utils.to_categorical(y_train, NB_CLASSES)



x_train_norm = x_train / 255.0

x_test_norm = x_test / 255.0
datagen = ImageDataGenerator(

        #rotation_range=0,

        zoom_range=0.1,

        width_shift_range=0.1,

        height_shift_range=0.1,

        horizontal_flip=True,

        fill_mode='reflect',

        validation_split=VALID_SPLIT

  )



train_generator = datagen.flow(x_train_norm, y_train, 

                               shuffle=True,

                               batch_size=TRAIN_BATCH_SIZE,

                               subset="training")



valid_generator = datagen.flow(x_train_norm, y_train, 

                               shuffle=False,

                               batch_size=VALID_BATCH_SIZE,

                               subset="validation")
def convBlock(kernels, kernel_size, layers, use_max_pooling, input): 

    out = input



    for i in range(layers):

      out = Conv2D(kernels, kernel_size=kernel_size, activation=None, padding='same')(out)

      out = BatchNormalization()(out)

      out = ELU()(out)

    

    if use_max_pooling:

      out = MaxPooling2D(pool_size=(2, 2))(out)

    

    out = Dropout(0.25)(out)

    return out
def wideResBlock(nb_layer, width, input):

  kernels = 8 * (2**nb_layer) * width 

  

  main = Conv2D(filters=kernels, kernel_size=(3, 3), padding="same", use_bias=False)(input)

  main = BatchNormalization()(main)

  main = ELU()(main)

  

  main = Dropout(0.25)(main)



  main = Conv2D(filters=kernels, kernel_size=(3, 3), padding="same", use_bias=False)(main)

  main = BatchNormalization()(main)

  main = ELU()(main)

  

  if input.shape[-1] == kernels:

    shortcut = input

  else:

    shortcut = Conv2D(filters=kernels, kernel_size=(1, 1), padding="same", use_bias=False)(input)

    shortcut = BatchNormalization()(shortcut)

  

  return Add()([main, shortcut])
input_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

input = Input(shape=input_shape)



out = BatchNormalization()(input)



# input block

out = convBlock(kernels=32, kernel_size=(3, 3), layers=1, use_max_pooling=False, input=out)



# first N blocks

for i in range(N):

  out = wideResBlock(nb_layer=1, width=k, input=out)



out = MaxPooling2D(pool_size=(2, 2))(out)



# second N blocks

for i in range(N):

  out = wideResBlock(nb_layer=2, width=k, input=out)



out = MaxPooling2D(pool_size=(2, 2))(out)



# third N blocks

for i in range(N):

  out = wideResBlock(nb_layer=3, width=k, input=out)



out1 = GlobalMaxPooling2D()(out)

out2 = GlobalAveragePooling2D()(out)

out = Concatenate()([out1, out2])



out = Dropout(0.4)(out)



out = Dense(256, activation=None)(out)

out = BatchNormalization()(out)

out = ELU()(out)

out = Dropout(0.4)(out)



out = Dense(NB_CLASSES, activation='softmax')(out)



model = Model(inputs=input, outputs=out)
optimizer = Adam(lr=0.001)

#optimizer = Adam(lr=1e-05)



model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

print(model.summary())


callbacks = get_cb(True, True, False, True, model_fname=f'{model_fname}.callback')
history = model.fit_generator(generator=train_generator,

                              validation_data=valid_generator,

                              epochs=EPOCHS, 

                              verbose=1,

                              callbacks=callbacks)
# Загрузим веса с наилучшей аккуратностью на проверочной выборке

model.load_weights('myConv_model.callback.best.hdf5')

predictions = model.predict(x_test_norm)
predictions = np.argmax(predictions, axis=1)

out = np.column_stack((range(1, predictions.shape[0]+1), predictions))
np.savetxt('submission.csv', out, header="Id,Category", 

            comments="", fmt="%d,%d")