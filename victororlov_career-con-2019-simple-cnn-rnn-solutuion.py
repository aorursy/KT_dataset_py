# I'll put  all the nesessary libraries here

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

from scipy import signal

import os

import seaborn as sns

from tqdm import tqdm

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import StratifiedShuffleSplit



from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Activation, Conv1D, Flatten, Dropout, TimeDistributed, Bidirectional, MaxPooling1D, SpatialDropout1D, GlobalAveragePooling1D

from tensorflow.python.keras.layers import GlobalMaxPooling1D, BatchNormalization

from tensorflow.keras.layers import GRU, LSTM, CuDNNGRU, CuDNNLSTM

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras import Input, Model
# Disabling the warnings

import warnings

warnings.filterwarnings("ignore")

matplotlib.rcParams['figure.figsize'] = (15.0, 5.0)
# Downloading the training set

df_train = pd.read_csv('../input/X_train.csv')

target_train = pd.read_csv('../input/y_train.csv')



# Lets see whats inside

df_train.head()
plt.figure(figsize=(15,25))

for i, column in enumerate(df_train.columns[3:]):

    plt.subplot(10,1,i+1)

    plt.plot(df_train[column][:128])

    plt.title(column)

    plt.grid()
# I am declaring a function so we could use it for the test set later as well

def create_X(df, nrows):

    # The data will be organized into 3D numpy array with shape (series id, measurement number, channel)

    X = np.zeros((nrows,128,10))

    X[:,:,0] = df['orientation_X'].values.reshape((-1,128))

    X[:,:,1] = df['orientation_Y'].values.reshape((-1,128))

    X[:,:,2] = df['orientation_Z'].values.reshape((-1,128))

    X[:,:,3] = df['orientation_W'].values.reshape((-1,128))



    X[:,:,4] = df['angular_velocity_X'].values.reshape((-1,128))

    X[:,:,5] = df['angular_velocity_Y'].values.reshape((-1,128))

    X[:,:,6] = df['angular_velocity_Z'].values.reshape((-1,128))



    X[:,:,7] = df['linear_acceleration_X'].values.reshape((-1,128))

    X[:,:,8] = df['linear_acceleration_Y'].values.reshape((-1,128))

    X[:,:,9] = df['linear_acceleration_Z'].values.reshape((-1,128))

    

    # Detrending each signal

    for i in range(10):

        X[:,:,i] = signal.detrend(X[:,:,i])

        

    # Scaling groups of channels corresponding to different physical quantities to standard deviation

    X[:,:,[0,1,2,3]] = X[:,:,[0,1,2,3]]/X[:,:,[0,1,2,3]].std()

    X[:,:,[4,5,6]] = X[:,:,[4,5,6]]/X[:,:,[4,5,6]].std()

    X[:,:,[7,8,9]] = X[:,:,[7,8,9]]/X[:,:,[7,8,9]].std()



    return X



X = create_X(df_train, 3810)

_ = plt.plot(X[10])
target_train.head()
# I'm using the LabelEncoder to turn text labels into vector of integers...

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(target_train.surface.values.reshape(-1,1))

labels = le.transform(target_train.surface.values.reshape(-1,1))

# ... and than to binary class matrix

y = to_categorical(labels)
_ = plt.hist(labels)
sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2)

train_index, test_index = next(sss.split(X, y))

X_train, X_val = X[train_index], X[test_index]

y_train, y_val = y[train_index], y[test_index]
# I'm creating a generator so we can endlessly feed batches of random samples for our model to train

def data_generator(batch_size):

    while True:

        batch_index = np.random.choice(train_index, batch_size)

        X_out = X[batch_index]            

        yield X_out, y[batch_index]



train_gen = data_generator(64)

next(train_gen)[1].shape
# I took the function below from this kernel, very helpful:

# https://www.kaggle.com/mayer79/rnn-starter-for-huge-time-series

def history_plot(history, what):

    x = history.history[what]

    val_x = history.history['val_' + what]

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs = np.asarray(history.epoch) + 1

    

    plt.subplot(1,2,1)

    plt.plot(epochs, x, 'b', label = "Training " + what)

    plt.plot(epochs, val_x, 'r', label = "Validation " + what)

    plt.grid()

    plt.title("Training and validation " + what)

    plt.xlabel("Epochs")

    plt.legend()

    

    plt.subplot(1,2,2)

    plt.plot(epochs, loss, 'b', label = "Training loss")

    plt.plot(epochs, val_loss, 'r', label = "Validation loss")

    plt.grid()

    plt.title("Training and validation " + what)

    plt.xlabel("Epochs")

    plt.legend()

    plt.show()

    return None
model = Sequential()

# Again I chose 32 units as a starting point just because starting examples in "Deep Learning with Python" by François Chollet. 

# I dont really know any hard theory behind this decision.

model.add(CuDNNGRU(32, input_shape=(128,10))) 

model.add(Dense(9, activation='softmax'))

model.summary()

model.compile(optimizer=RMSprop(), loss="categorical_crossentropy", metrics=['accuracy'])



history = model.fit_generator(train_gen,epochs=10, steps_per_epoch=500, validation_data=(X_val, y_val),verbose=False)

history_plot(history, what='acc')
model = Sequential()

# And again, the number of filter I used and the kernel size is a bit of guesswork

model.add(Conv1D(filters=150,kernel_size=10,activation='relu', input_shape=(128,10)))

# I'm using MaxPolling to reduce dimentionality of convolution layer before feeding it to GRU

model.add(MaxPooling1D(2))

# I have also added dropout to prevent overfitting

model.add(Dropout(0.5))

model.add(CuDNNGRU(32))

model.add(Dropout(0.5))

model.add(Dense(9, activation='softmax'))

model.summary()



model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=3, min_lr=0.001)

history = model.fit_generator(train_gen,epochs=50, steps_per_epoch=1000, validation_data=(X_val, y_val),

                             callbacks=[reduce_lr])
history_plot(history, what='acc')
def aug_generator(batch_size):

    while True:

        batch_index = np.random.choice(train_index, batch_size)

        X_out = X[batch_index]

        if (np.random.randint(0,2)):

            # randomly reversing all the samples in the batch

            X_out = np.flip(X[batch_index],axis=1)

        if (np.random.randint(0,2)):

            # randomly inversing all the samples in the batch

            X_out = - X[batch_index]

            

        yield X_out, y[batch_index]



train_gen = aug_generator(64)
model = Sequential()

model.add(Conv1D(filters=150,kernel_size=10,activation='relu', input_shape=(128,10)))

model.add(MaxPooling1D(2))

model.add(Dropout(0.5))

model.add(CuDNNGRU(32))

model.add(Dropout(0.5))

model.add(Dense(9, activation='softmax'))

model.summary()



model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=3, min_lr=0.001)

history = model.fit_generator(train_gen,epochs=50, steps_per_epoch=1000, validation_data=(X_val, y_val),

                             callbacks=[reduce_lr])
history_plot(history, what='acc')
df_test = pd.read_csv('../input/X_test.csv')

submission = pd.read_csv('../input/sample_submission.csv')

X_test = create_X(df_test, 3816)
y_test = np.argmax(model.predict(X_test, verbose=1),axis=1)

preds = le.inverse_transform(y_test.reshape(-1,1))

submission['surface'] = preds

submission.to_csv('submission.csv',index=False)

print(preds)