# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
from keras.callbacks import EarlyStopping
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score




# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# This function can help you to prepare balanced dataset
def balance(train, target, sampling_amount):
    degerler = train[target].value_counts()
    if sampling_amount is 'mean' or 'Mean':
        rounded = int(np.round(degerler.values.mean()))
    else:
        rounded = sampling_amount
    for i in range(len(degerler)):
        gg = train[train['open_channels']==i]
        train = train.drop(train[train['open_channels']==i].index)
        if degerler[i]<=degerler.mean():
            labelt_upsampled = gg.sample(rounded, replace=True)
            train = pd.concat([train, labelt_upsampled])
        else:
            labelt_downsampled = gg.sample(rounded)
            train = pd.concat([train,labelt_downsampled])
    return train

train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv', dtype={'time': 'str'})
#labelt = train['open_channels'][1000000:3000000]
#traindata = train['signal'][1000000:3000000]
ttrain = balance(train[:3000000],'open_channels', 'mean')
ttest = train[3000000:]
#train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv', dtype={'time': 'str'})
gg = ttest['open_channels'].value_counts()
gg
labelt = ttrain['open_channels'].values
tlabelt = ttest['open_channels'].values
#ind = train['signal'].index
#fftveri = np.imag(np.fft.fft(train['signal'].values))
#fftverinorm = fftveri/np.max(fftveri)
#signalnorm = pd.DataFrame(train['signal']/train['signal'].max(), columns=['signal'])
#fftverinormp = pd.DataFrame(fftveri, columns=['TFFT'], index = ind)
#traindata = pd.concat([fftverinormp, signalnorm], axis=1)
traindata = ttrain['signal'].values
testdata = ttest['signal']
#traindata = signalnorm['signal'].apply(lambda x: (x**2)*np.exp(-x**2))
#traindata = traindata.values.reshape(len(traindata),1,1)
#testdata = testdata.values.reshape(len(testdata),1,1)
# These functions create sequenced data in appropriate format in order to be able to insert them 7
# into GRU or LSTM

def seqcreator(seq, label, seqlength):
    inpx, inpy = list(), list()
    for i in range(len(seq)):
        steprange = i + seqlength
        if steprange > len(seq)-1:
            break
        inp_seq_x, inp_seq_y = seq[i:steprange], label[i]
        inpx.append(inp_seq_x)
        inpy.append(inp_seq_y)
    return np.array(inpx), np.array(inpy)
    
def seqcreatorsingle(seq, seqlength):
    inpx = list()
    for i in range(len(seq)):
        steprange = i + seqlength
        if steprange > len(seq)-1:
            break
        inp_seq_x = seq[i:steprange]
        inpx.append(inp_seq_x)
    return np.array(inpx)
seqlength = 10
X, y = seqcreator(traindata, labelt, seqlength)
X = X.reshape(len(X), seqlength, 1)
#y = y/max(y)
(X.shape, y.shape)
# Here comes the GRU model for sequenced data
model = tf.keras.Sequential([
    tf.keras.layers.GRU(60, return_sequences=True),
    tf.keras.layers.GRU(60),
    tf.keras.layers.Flatten(),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Dense(11),   
])

callback = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=3)
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['categorical_accuracy'])

model.fit(X, y, batch_size = 50, epochs=5, callbacks = [callback])
# This is CNN architecture to try how effective CNN can be for sequenced data
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(40, 1, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling1D(pool_size=1),
    tf.keras.layers.Conv1D(60, 1, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling1D(pool_size=1),
    tf.keras.layers.Conv1D(60, 1, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling1D(pool_size=1),
    tf.keras.layers.Conv1D(40, 1, activation='relu', padding='same'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(11)    
])

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
model.compile(optimizer='rmsprop',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

model.fit(traindata, labelt, batch_size = 30, epochs=50, validation_data=(testdata, tlabelt), callbacks = [callback])
# Let's check what Gradient Boost method will predict
traindata = traindata.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(traindata, labelt, test_size=0.3)
model = GradientBoostingClassifier(n_estimators=150)
model.fit(X_train, y_train)
predict = model.predict(X_test)
acc = accuracy_score(y_test, predict)
acc

acc = f1_score(y_test, predict, average='weighted')
acc
prediction = model.predict(test)
Xtest = traindata
Xtest = Xtest.reshape(len(Xtest), n_step, 1)
Xtest.shape
yhat = model.predict(Xtest, verbose=0)
(yhat, labelt[101500])
labelt[101500]
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
time = test['time']
test.drop(['time'], axis = 1, inplace = True)
test = test.values
#test = seqcreatorsingle(test, seqlength)
#test = test.reshape(len(test), 1, 1)
#test.shape
#test = test['signal']
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
prediction = probability_model.predict(test)

train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv', dtype={'time': 'str'})
testdeneme = train['signal'][3000000:]
testdeneme = testdeneme.values.reshape(len(testdeneme),1,1)
prediction = model.predict(testdeneme)
#testdeneme.shape
prediction[0]
train['open_channels'][3000000]
ids = np.zeros(shape=(len(testo),1))
for i, z in enumerate(prediction):
    ids[i] = np.argmax(z)
ids = pd.DataFrame(ids, columns=['open_channels'])
ids['time'] = time
ids = ids[['time','open_channels']]
ids
ids['open_channels'].value_counts()
def prediction_csv(array, sample_submission_df, output_csv):
    temp_dataset = pd.DataFrame({'time':sample_submission_df.time,'open_channels':array})
    temp_dataset['time'] = temp_dataset['time'].apply(lambda x: format(x,'.4f'))
    temp_dataset['open_channels'] = temp_dataset['open_channels'].apply(lambda x: int(x))
    temp_dataset.to_csv(output_csv, index=False)
asd = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
prediction_csv(prediction, asd, '/kaggle/working/predict2.csv')
sns.boxplot(y=signal,x=labell)
sns.scatterplot(x=train['time'][2999980:3000000], y=labelt[2999980:3000000])
train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv', dtype={'time': 'str'})
labell = train['open_channels']
signal = train['signal']
train['open_channels'].value_counts()
b = []
for i in range (11):
    gg = train[train['open_channels']==i]
    b.append(gg['signal'].median())

test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
result = []
for j in range(len(test)):
    result.append(np.argmin(np.abs(b-test['signal'][j])))
loca = pd.DataFrame(result, columns=['open_channels'])
loca
asd = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
prediction_csv(loca['open_channels'].values, asd, '/kaggle/working/predict2.csv')
loca['open_channels'].value_counts()
