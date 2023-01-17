import random

import numpy as np

from keras import metrics, optimizers, Sequential

from keras.layers import Dense

from keras.losses import mean_squared_error

from sklearn.model_selection import train_test_split





def cube(x):

    return x ** 3 + random.random()





x = np.linspace(1.0, 5.0, num=1000).reshape((1000, 1))

cube_fun = np.vectorize(cube)

y = cube_fun(x)

BATCH_SIZE = 128

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,shuffle=True)





def alexnet_model():

    alexnet = Sequential([

        Dense(10, activation='relu', input_dim=1),

        Dense(10),

        Dense(1)])

    return alexnet





def train_model(model):

    model.compile(optimizer=optimizers.SGD(), loss=mean_squared_error)

    model.fit(X_train, y_train,

              batch_size=BATCH_SIZE,

              epochs=100,

              verbose=1,

              validation_data=(X_test, y_test), callbacks=[])





model = alexnet_model()

train_model(model)

def train_model(model):

    model.compile(optimizer=optimizers.RMSprop(), loss=mean_squared_error)

    model.fit(X_train, y_train,

              batch_size=BATCH_SIZE,

              epochs=100,

              verbose=1,

              validation_data=(X_test, y_test), callbacks=[])

model = alexnet_model()

train_model(model)
def train_model(model):

    model.compile(optimizer=optimizers.SGD(lr=0.00001), loss=mean_squared_error)

    model.fit(X_train, y_train,

              batch_size=BATCH_SIZE,

              epochs=100,

              verbose=1,

              validation_data=(X_test, y_test), callbacks=[])

model = alexnet_model()

train_model(model)