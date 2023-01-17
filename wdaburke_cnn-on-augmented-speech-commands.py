# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import nltk



from scipy import signal

from scipy.io import wavfile

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

"""for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))"""



# Any results you write to the current directory are saved as output.
sample_rate, samples = wavfile.read('/kaggle/input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset/happy/2173.wav')

frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)



plt.pcolormesh(times, frequencies, spectrogram)

plt.imshow(np.log(spectrogram))

plt.ylabel('Frequency [Hz]')

plt.xlabel('Time [sec]')

plt.show()



Pxx, freqs, bins, im = plt.specgram(samples, Fs=sample_rate)



# add axis labels

plt.ylabel('Frequency [Hz]')

plt.xlabel('Time [sec]')
X = []

y = []



for dirname, _, filenames in os.walk('/kaggle/input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset/'):

    for filename in filenames:

        if dirname.split('/')[-1]:

            sample_rate, samples = wavfile.read(os.path.join(dirname, filename))

            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

            X.append(spectrogram)

            y.append(dirname.split('/')[-1])

X = np.array(X)

X = X.reshape(X.shape + (1,))            
all_labels = nltk.FreqDist(y)

all_labels_df = pd.DataFrame({'Label': list(all_labels.keys()), 'Count': list(all_labels.values())})

num_labels = len(all_labels)

print('num labels ',num_labels)



g = all_labels_df.nlargest(columns="Count", n = 50) 

plt.figure(figsize=(12,15))

ax = sns.barplot(data=g, x= "Count", y = "Label")

#ax.set(ylabel = 'Label')



plt.show()
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()



mlb.fit(pd.Series(y).fillna("missing").str.split(', '))

y_mlb = mlb.transform(pd.Series(y).fillna("missing").str.split(', '))

mlb.classes_
X.shape

y_mlb.shape
X_train, X_valtest, y_train, y_valtest = train_test_split(X,y_mlb,test_size=0.2, random_state=37)

X_val, X_test, y_val, y_test = train_test_split(X_valtest,y_valtest,test_size=0.5, random_state=37)

X_train.shape, X_val.shape,X_test.shape, y_train.shape, y_val.shape,y_test.shape
import keras

from keras import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D,Flatten,Dropout,BatchNormalization
droprate = 0.5



input_shape = (X_train.shape[1],X_train.shape[2],1)

model = Sequential()



model.add(Conv2D(512,kernel_size=(3,3),activation='relu',input_shape=input_shape,padding="same"))

model.add(BatchNormalization())

#model.add(Dropout(droprate))

model.add(MaxPooling2D())



model.add(Conv2D(256,kernel_size=(3,3),activation='relu',border_mode="same"))

model.add(BatchNormalization())

model.add(MaxPooling2D())

#model.add(Dropout(droprate))



model.add(Conv2D(128,kernel_size=(3,3),activation='relu',border_mode="same"))

model.add(BatchNormalization())

model.add(MaxPooling2D())

#model.add(Dropout(droprate))



model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(droprate))

model.add(Dense(30, activation='softmax'))
(None,1,X_train.shape[1],X_train.shape[2])

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])

model.build()

model.summary()
from keras.callbacks import EarlyStopping

epochs = 3

batch_size = 32

callbacks = [

    EarlyStopping(

        monitor='val_acc', 

        patience=4,

        mode='max',

        verbose=1)

]



history = model.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(X_val, y_val),shuffle=True,callbacks=callbacks)



#score = model.evaluate(X_test_CNN, y_test_CNN, verbose=0)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.evaluate(X_test, y_test, verbose=1)

y_pred = model.predict(X_test, batch_size=32, verbose=1)



y_pred = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)

#print(classification_report(y_test, )



report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()

report_df["label"] = list(mlb.classes_) + ["micro avg","macro avg","weighted avg","samples avg"]

report_df.sort_values(by=['f1-score','support'], ascending=False)