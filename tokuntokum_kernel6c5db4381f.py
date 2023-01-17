import os

import keras

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D, MaxPool2D, LeakyReLU, Input,Dense, Dropout, Activation, Flatten

from keras.models import Sequential

from keras.utils import to_categorical

import keras.preprocessing.image as Image



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

# 学習データ、テストデータを読み込み

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv').to_numpy()



train.head()

x_train = train.drop('label', axis=1).to_numpy()

y_train = train['label'].to_numpy()

y_train = to_categorical(y_train, num_classes = 10) 

print(x_train.shape, y_train.shape)





#データを 縦28 ×  横28 ×　１(RGBだと3, 今回は白黒なので1) の形に整形

x_train = x_train.reshape(-1, 28, 28, 1)

test = test.reshape(-1, 28, 28, 1)



x_train = x_train.astype('float32')

test = test.astype('float32')



# 正規化

x_train /= 255

test /= 255



#データ水増し

datagen = Image.ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(x_train)





#opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

#opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) #Adamaxを指定

#opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

#opt = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

#opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

EPOCHS = 100



# トレーニングデータを使って学習####################################

# ニューラルネットワークの層構成(今回はCNNという深層学習の１つを使用)

model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





# model.add(GlobalAveragePooling2D())

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))





# 学習に関する色々な設定

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

#model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['accuracy'])



batch_size = 192



# 実際に学習を開始、epochsは学習回数を表す

#history = model.fit(x_train, y_train, batch_size=64, epochs=EPOCHS, )

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),batch_size, epochs=EPOCHS, )

##############################################################





# テストデータを使って推論

preds = model.predict(test)



# 提出

df = pd.DataFrame(np.argmax(preds, axis=1), columns=['Label'])

df.insert(0, 'ImageId', df.index + 1)

df.to_csv('submission.csv', index=False)