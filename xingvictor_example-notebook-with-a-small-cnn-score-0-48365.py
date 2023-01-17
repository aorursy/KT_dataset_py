from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout, SpatialDropout2D, SeparableConv2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
X_train_original = np.load('/kaggle/input/rcmemulators/X_train.dat', allow_pickle=True)
X_train_original.shape
temp_mpl_gcm = X_train_original[:,4,6,15]
y_train = pd.read_csv('/kaggle/input/rcmemulators/Y_train_mpl.csv')
Y_train_temp = np.asarray(y_train.tempé - temp_mpl_gcm)
Y_train_temp
X_train, X_valid, y_train, y_valid = train_test_split(X_train_original, Y_train_temp, test_size = 0.2, random_state = 42)
X_train = X_train.reshape(X_train.shape[0], -1)
X_valid = X_valid.reshape(X_valid.shape[0], -1)
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_valid = sc_x.transform(X_valid)
X_train = X_train.reshape(X_train.shape[0], 11, 11, 19)
X_valid = X_valid.reshape(X_valid.shape[0], 11, 11, 19)
# Things that did not work: using less input channels, RFs, regularizing a bigger model (l2, Dropout after Dense, SpatialDropout2D, BatchNorm), SeparableConv2D
# Good regularization with smaller network and smaller BS (128 -> 16)
def step_decay(epoch, lr):
    lrate = lr
    if epoch%10 == 0 and epoch > 1:
        lrate /= 2
    return lrate
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', input_shape = (11,11,19), activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (3,3), strides=(2, 2), padding = 'same', activation = 'relu'))
model.add(Flatten())
model.add(Dense(8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1))
model.summary()
myopt = Adam(lr=1e-3)
model.compile(loss='mean_squared_error', optimizer = myopt, metrics=['mae'])
sched = LearningRateScheduler(step_decay, verbose=1)
ckpt = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True, verbose=1)
model.fit(x=X_train, y=y_train,
          validation_data=(X_valid, y_valid),
          batch_size=16,
          epochs=30,
          callbacks=[ckpt, sched])
losses = pd.DataFrame(model.history.history)
losses.plot()
# You can load a previous model if it is better than the current one
# model = load_model('weights_21706.h5')
X_test_origin = np.load('/kaggle/input/rcmemulators/X_test.dat', allow_pickle=True)
temp_mpl_gcm=X_test_origin[:,4,6,15]
temp_mpl_gcm
X_test = X_test_origin.reshape(X_test_origin.shape[0], -1)
X_test = sc_x.transform(X_test)
X_test = X_test.reshape(X_test.shape[0], 11, 11, 19)
predictions = model.predict(X_test)
predictions
predictions.mean()
predictions.std()
pred_train = model.predict(X_train)
pred_train.mean()
pred_train.std()
y_test = predictions[:, 0] + X_test_origin[:,4,6,15]
y_test.shape
y_test
res = pd.read_csv('/kaggle/input/rcmemulators/samplesub.csv')
res.tempé = y_test
print(res)
res.to_csv('./submission.csv', index=False)
