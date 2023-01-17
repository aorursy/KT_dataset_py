import pandas as pd

import cv2

import os



train_csv = pd.read_csv('../input/train.csv')



def load_imgs(path):

    imgs = {}

    for f in os.listdir(path):

        fname = os.path.join(path, f)

        imgs[f] = cv2.imread(fname)

    return imgs



img_train = load_imgs('../input/train/train/')

img_test = load_imgs('../input/test/test/')

import numpy as np



X_train = []

Y_train = []



for _, row in train_csv.iterrows():

    X_train.append(img_train[row['id']])

    Y_train.append(int(row['has_cactus']))



X_train = np.array(X_train)

Y_train = np.array(Y_train)



X_test = np.array([img_test[f] for f in img_test])



print('Training data shape:', X_train.shape, '=>', Y_train.shape)
# Validation split

idxs = np.random.permutation(17500)



x_train = X_train[idxs[:12000]]

y_train = Y_train[idxs[:12000]]



x_valid = X_train[idxs[12000:]]

y_valid = Y_train[idxs[12000:]]
from keras.layers import *

from keras.models import Sequential



def add_conv_level(model, **args):

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', **args))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D())



def make_model():

    model = Sequential()

    

    for i in range(3):

        if i:

            add_conv_level(model)

        else:

            add_conv_level(model, input_shape=(32,32,3))

    

    model.add(Flatten())

    

    model.add(Dropout(0.4))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    

    return model

    

model = make_model()

model.summary()
from keras.optimizers import *



model.compile(optimizer=RMSprop(lr=1e-4),

              loss='binary_crossentropy',

              metrics=['accuracy'])
model.fit(x_train, y_train,

          epochs=16, batch_size=32,

          validation_data = (x_valid, y_valid))
model = make_model()



model.compile(optimizer=RMSprop(lr=1e-4),

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.fit(X_train, Y_train, epochs=16, batch_size=32)



# Save the model for future usage

model.save('model.h5')
pred = model.predict(X_test)





df = pd.DataFrame({

    'id': [f for f in img_test],

    'has_cactus': [int(x[0] >= 0.5) for x in pred]

})



print(df)
df.to_csv('predictions.csv', index=False)