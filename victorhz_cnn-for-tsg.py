import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers import Dropout, BatchNormalization, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dot, Add, Multiply
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from sklearn.model_selection import KFold
import gc
import os
print(os.listdir("../input"))
print(os.listdir("../input/image-to-arrays-tsg/"))
import matplotlib.pyplot as plt
%matplotlib inline
X_train = np.load('../input/image-to-arrays-tsg/X_train.npy')
y_train = np.load('../input/image-to-arrays-tsg/Y_train.npy')
X_test = np.load('../input/image-to-arrays-tsg/X_test.npy')
y_train = y_train.reshape(4000, 101*101)
X_train.shape, y_train.shape
names = os.listdir('../input/tgs-salt-identification-challenge/test/images/')
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), strides=(2,2), activation='relu', input_shape=(101,101,3)))
model.add(Dropout(0.5))
model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), strides=(2,2), activation='relu'))
model.add(Dropout(0.5))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(101, kernel_size=(2,2), strides=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Reshape((101,12,12)))
model.add(Conv2D(64, kernel_size=(1,2), strides=(1,2), activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Reshape((64,101,6)))
model.add(Conv2D(8, kernel_size=(2,3), strides=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(101*101, activation='sigmoid'))
#model.add(Reshape((101,101)))
model.summary()
from keras import metrics
from keras import losses
model.compile(loss=losses.mean_squared_error, metrics=[metrics.mean_absolute_error], optimizer='adam')
folds = KFold(n_splits=200, shuffle=True, random_state=42)
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
    trn_x, trn_y = X_train[trn_idx], y_train[trn_idx]
    val_x, val_y = X_train[val_idx], y_train[val_idx]
    print( 'Fitting neural network...' )
    model.fit(trn_x, trn_y, batch_size=100, validation_data = (val_x, val_y), epochs=10, verbose=1)
    break
y_val = np.round(model.predict(val_x))
del X_train ,y_train, folds
gc.collect()
plt.imshow(np.abs(val_y[0].reshape(101,101)-y_val[0].reshape(101,101)))
plt.imshow(np.abs(val_y[1].reshape(101,101)-y_val[1].reshape(101,101)))
plt.imshow(np.abs(val_y[2].reshape(101,101)-y_val[2].reshape(101,101)))
print(val_y[2].reshape(101,101)-y_val[2].reshape(101,101))
plt.imshow(np.abs(val_y[3].reshape(101,101)-y_val[3].reshape(101,101)))
plt.imshow(np.abs(val_y[4].reshape(101,101)-y_val[4].reshape(101,101)))
del y_val
gc.collect()
y_test = np.round(model.predict(X_test)).reshape((18000, 101, 101))
del X_test
gc.collect()
t = 0
for i in range(len(y_test)):
    xi = y_test[i].tolist().count(1)
    if xi != 0:
        t+=1
        #print(xi)
print(t)
i_ = y_test[0]
out = []
out1 = []
for _ in range(len(y_test)):
    en = y_test[_].reshape(101*101)
    l = []
    encoded = ''
    for i in range(1,len(en)+1):
        if en[i-1] == 1:
            l.append(i)
    for i in range(len(l)):
        if i==0:
            t = 1
            encoded += str(l[0])+' '
        elif l[i]-l[i-1]==1:
            t+=1
        elif l[i]-l[i-1]!=1:
            encoded += str(t)+' '
            t = 1
            encoded += str(l[i])+' '
    if len(l)==0:
        t = ''
    encoded += str(t)
    out.append(encoded)
for _ in range(len(y_test)):
    en = y_test[_].T.reshape(101*101)
    l = []
    encoded = ''
    for i in range(1,len(en)+1):
        if en[i-1] == 1:
            l.append(i)
    for i in range(len(l)):
        if i==0:
            t = 1
            encoded += str(l[0])+' '
        elif l[i]-l[i-1]==1:
            t+=1
        elif l[i]-l[i-1]!=1:
            encoded += str(t)+' '
            t = 1
            encoded += str(l[i])+' '
    if len(l)==0:
        t = ''
    encoded += str(t)
    out1.append(encoded)
df = pd.DataFrame()
df1 = pd.DataFrame()
for i in range(len(names)):
    names[i] = names[i].replace('.png','')
df['id'] = names
df1['id'] = names
df['rle_mask'] = out
df1['rle_mask'] = out1
df.to_csv('sub.csv', index=False)
df.head()
df1.to_csv('sub1.csv', index=False)
df1.head()
y_train = np.array([[0,0,0,0,1,1,0],[1,1,0,0,1,0,1]])
out = []
for _ in range(len(y_train)):
    en = y_train[_]
    l = []
    encoded = ''
    for i in range(len(en)):
        if en[i] == 1:
            l.append(i)
    for i in range(len(l)):
        if i==0:
            t = 1
            encoded += str(l[0])+' '
        elif l[i]-l[i-1]==1:
            t+=1
        elif l[i]-l[i-1]!=1:
            encoded += str(t)+' '
            t = 1
            encoded += str(l[i])+' '
    if len(l)==0:
        t = ''
    encoded += str(t)
    out.append(encoded)
out
