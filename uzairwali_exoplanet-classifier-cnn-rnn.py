# Run on GPU kernel



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.models import Model

from keras.layers import Input, Dense, LSTM, concatenate, Activation, CuDNNLSTM, MaxPool1D, Flatten

from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Bidirectional

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from keras.optimizers import Adam

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
raw_data = np.loadtxt('../input/exoTrain.csv', skiprows=1, delimiter=',')

x_train = raw_data[:, 1:]

y_train = raw_data[:, 0, np.newaxis] - 1.

raw_data = np.loadtxt('../input/exoTest.csv', skiprows=1, delimiter=',')

x_test = raw_data[:, 1:]

y_test = raw_data[:, 0, np.newaxis] - 1.

del raw_data
x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1,1)) / 

           np.std(x_train, axis=1).reshape(-1,1))

x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / 

          np.std(x_test, axis=1).reshape(-1,1))
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train,

                                                  test_size=0.3, random_state=123)
np.set_printoptions(threshold=np.inf)

X_train_r = np.expand_dims(X_train, axis=2)

X_val_r = np.expand_dims(X_val, axis=2)

x_test = np.expand_dims(x_test, axis=2)
def batch_generator(x_train, y_train, batch_size=32):

    """

    Gives equal number of positive and negative samples, and rotates them randomly in time

    """

    half_batch = batch_size // 2

    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')

    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')

    

    yes_idx = np.where(y_train[:,0] == 1.)[0]

    non_idx = np.where(y_train[:,0] == 0.)[0]

    

    while True:

        np.random.shuffle(yes_idx)

        np.random.shuffle(non_idx)

    

        x_batch[:half_batch] = x_train[yes_idx[:half_batch]]

        x_batch[half_batch:] = x_train[non_idx[half_batch:batch_size]]

        y_batch[:half_batch] = y_train[yes_idx[:half_batch]]

        y_batch[half_batch:] = y_train[non_idx[half_batch:batch_size]]

    

        for i in range(batch_size):

            sz = np.random.randint(x_batch.shape[1])

            x_batch[i] = np.roll(x_batch[i], sz, axis = 0)

     

        yield x_batch, y_batch
ip = Input(shape=(3197, 1))
# LSTM

x = Permute((2, 1))(ip)

x = CuDNNLSTM(16, return_sequences=True)(x)

x = CuDNNLSTM(32, return_sequences=True)(x)

x = CuDNNLSTM(64, return_sequences=True)(x)

x = CuDNNLSTM(128)(x)

x = Dropout(0.25)(x)
y = Conv1D(filters=16, kernel_size=11, activation='relu')(ip)

y = MaxPool1D(strides=4)(y)

y = BatchNormalization()(y)

y = Conv1D(filters=32, kernel_size=11, activation='relu')(y)

y = MaxPool1D(strides=4)(y)

y = BatchNormalization()(y)

y = Conv1D(filters=64, kernel_size=11, activation='relu')(y)

y = MaxPool1D(strides=4)(y)

y = BatchNormalization()(y)

y = Conv1D(filters=128, kernel_size=11, activation='relu')(y)

y = MaxPool1D(strides=4)(y)

y = Flatten()(y)

y = Dropout(0.25)(y)

y = Dense(64, activation='relu')(y)
#Concatenate

x = concatenate([x, y])

x = Dense(32, activation='relu')(x)

out = Dense(1, activation='sigmoid')(x)

model = Model(ip, out)

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(batch_generator(X_train_r, y_train, 32), 

                           validation_data=(X_val_r, y_val), 

                           epochs=100,

                           steps_per_epoch=X_train_r.shape[1]//32, verbose=0)
y_pred = model.predict(x=x_test)
non_idx = np.where(y_test[:,0] == 0.)[0]

yes_idx = np.where(y_test[:,0] == 1.)[0]

plt.plot([y_pred[i] for i in yes_idx], 'bo')

plt.show()

plt.plot([y_pred[i] for i in non_idx], 'ro')

plt.show()
#ROC Area under curve

y_true = (y_test[:, 0] + 0.5).astype("int")

fpr, tpr, thresholds = roc_curve(y_true, y_pred)

plt.plot(thresholds, 1.-fpr)

plt.plot(thresholds, tpr)

plt.show()

crossover_index = np.min(np.where(1.-fpr <= tpr))

crossover_cutoff = thresholds[crossover_index]

crossover_specificity = 1.-fpr[crossover_index]

print("Crossover at {0:.2f} with specificity {1:.2f}".format(crossover_cutoff, crossover_specificity))

plt.plot(fpr, tpr)

plt.show()

print("ROC area under curve is {0:.2f}".format(roc_auc_score(y_true, y_pred)))
predthr = np.where(y_pred > 0.5, 1, 0)

print(classification_report(y_test, predthr))
print(confusion_matrix(y_test, predthr))