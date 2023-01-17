# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv



np.random.seed(2)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import keras items.



import keras

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.optimizers import RMSprop

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
# trainとtestの読み込み

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# このmnistのデータはすでにバイナリ化されています



y_train = train['label']

x_train = train.drop(labels=['label'] ,axis=1)

del train

x_train.isnull().any().describe()

test.isnull().any().describe()
# 0~1に正規化

x_train = x_train / 255.0

test = test / 255.0
# 28 * 28 のチャネル1（grayscale）にreshape

x_train = x_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)

NUM_CLASSES = 10

y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
# 分割する　random_stateはseedのため固定すればなんでもok

random_seed = 2

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed)
# 画像をおためしで眺める

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



print(x_train.shape)

test_show = plt.imshow(x_train[0][:,:,0])
# kerasでモデル構築

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28, 28, 1)))

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(NUM_CLASSES, activation='softmax'))
# コンパイル、コールバック等パラメータを定義する

model.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])



reduce_lr = ReduceLROnPlateau(monitor='val_acc',

                              patience=3,

                              verbose=1,

                              factor=0.5,

                              min_lr=0.00001)

epochs = 30

batch_size = 86
# augmantation.



params = {

    'rotation_range':10,

    'zoom_range':0.1,

    'width_shift_range':0.1,

    'height_shift_range':0.1,

    'featurewise_center':False,

    'samplewise_center':False,

    'featurewise_std_normalization':False,

    'samplewise_std_normalization':False,

    'zca_whitening':False,

    'horizontal_flip':False,

    'vertical_flip':False

}



datagen = ImageDataGenerator(**params)



datagen.fit(x_train)
# 学習



hist = model.fit_generator(

    datagen.flow(x_train, y_train, batch_size=batch_size),

    epochs=epochs,

    validation_data=(x_test, y_test),

    verbose=1,

    steps_per_epoch=x_train.shape[0] // batch_size,

    callbacks=[reduce_lr]

)
# testを使って結果予測

results = model.predict(test)



# 10ラベルから一番でかいのを選ぶ（axis=1はたくさんのファイルが存在する為、一層落としてそれぞれのデータから見る）

results = np.argmax(results, axis=1)



#pandasで整形

results = pd.Series(results, name='Label')



results
submission = pd.concat([pd.Series(range(1,28001), name="ImageId"), results], axis=1)

submission.to_csv('mnist_datagen.csv', index=False)