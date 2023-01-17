# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        print(filename)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import librosa

import librosa.display

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, LSTM, Bidirectional, GRU

from keras.utils import to_categorical

import os

import math

import json

import random
DATASET_PATH = '/kaggle/input'

JSON_PATH = './myjson.json'

SAMPLE_RATE = sr =  22050

DURATION = 30 #measured in seconds 

SAMPLES_PER_TRACK = SAMPLE_RATE*DURATION
def save_mfcc(dataset_path, json_path, n_mfcc=40, n_fft=2048, hop_length=512, num_segments=5):

    #dictionary to store data

    data = {

        'mapping' : [],

        'mfcc' : [],

        'labels' : []

    }

    

    count = 0

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments) 

    print(num_samples_per_segment)

    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    

    #Loop through all the genres

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        

        #ensure that we're not at the root level

        if dirpath not in dataset_path:



            #save the semantic label

            dirpath_components = dirpath.split('/')

            semantic_label = dirpath_components[-1]

            data['mapping'].append(semantic_label)

            print('\nProcessing {}'.format(semantic_label))

            

            #process files for a specific genre 

            for f in filenames:

                if f.endswith('.wav') and f != 'jazz.00054.wav':

                    #load audio

                    file_path = os.path.join(dirpath,f)



                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE) # len(signal) = 661794

                    #print(signal,sr)

                    #process segments extracting mfcc and storing data

                    for s in range(num_segments):

                        start_sample = num_samples_per_segment * s

                        finish_sample = num_samples_per_segment + start_sample

                        mfcc = librosa.feature.mfcc(signal[start_sample : finish_sample],

                                                   sr = sr,

                                                   n_fft = n_fft,

                                                   n_mfcc = n_mfcc,

                                                   hop_length = hop_length)



                        mfcc = mfcc.T

                        # store mfcc for segment if it has the expected length

                        if len(mfcc) == expected_num_mfcc_vectors_per_segment:

                            print(mfcc.shape)

                            data['mfcc'].append(mfcc.tolist())

                            data['labels'].append(i)

                            print('{}, segment:{}'.format(file_path, s))

                            count += 1

    print(count)

    with open(json_path, 'w') as fp:

        json.dump(data, fp, indent=4)
save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
def load_data(path):

    with open(path, 'r') as fp:

        data = json.load(fp)

        

    #Convert lists into numpy arrays

    inputs = np.array(data['mfcc'])

    targets = np.array(data['labels'])

    

    return inputs, targets
inputs, targets = load_data('./myjson.json')
np.unique(targets, return_counts=True)
v = min(np.unique(targets))

for i in range(len(targets)):

    if targets[i] == v:

        targets[i] = 0

    else:

        new = targets[i] - v

        targets[i] = new
np.unique(targets, return_counts=True)
# If you want to apply Convolution NN the remove the comment from the below line

#inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], inputs.shape[2], 1))
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)
# Adding Noise 

for i in range(inputs_train.shape[0]):

    s = np.random.rand(inputs_train.shape[1], inputs_train.shape[2])

    inputs_train[i] = inputs_train[i] + s

model = Sequential()



model.add(GRU(100, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2])))

model.add(GRU(500))

model.add(Flatten())

model.add(Dense(100, 'relu'))

model.add(Dense(10, 'softmax'))



model.compile(optimizer=tf.keras.optimizers.Adam(),

             loss = 'sparse_categorical_crossentropy',

             metrics=['accuracy'])

model.summary()
history = model.fit(inputs_train, targets_train,

          validation_data=(inputs_test, targets_test),

          epochs = 100,

          batch_size=100)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()