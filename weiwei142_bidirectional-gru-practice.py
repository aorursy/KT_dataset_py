# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_dir = '../input/'

fname = os.path.join(data_dir,'jena_climate_2009_2016.csv')
f = open(fname)

data = f.read()

f.close





lines = data.split('\n')

header = lines[0].split(',')

lines = lines[1:]



print(header)

print(len(lines))
float_data = np.zeros((len(lines),len(header)-1))



for i, line in enumerate(lines):

    values = [float(x) for x in line.split(',')[1:]]

    float_data[i,:] = values



print(float_data.shape)
mean = float_data[:200000].mean(axis=0)

float_data -= mean

std = float_data[:200000].std(axis=0)

float_data /= std

def generator(data, lookback, delay, min_index, max_index,shuffle=False, batch_size=128, step=6):

    if max_index is None:

        max_index = len(data) - delay - 1

    

    i = min_index + lookback

    while True:

        if shuffle:

            rows = np.random.randint(min_index + lookback, max_index, size = batch_size)

        else:

            if i + batch_size >= max_index:

                i = min_index + lookback

            rows = np.arange(i, min(i + batch_size, max_index))

            i += len(rows)

        samples = np.zeros((len(rows),lookback//step,data.shape[1]))

            

        targets = np.zeros((len(rows),))

        #print('Rows:',rows[:3])

        for j, row in enumerate(rows):

            indices = range(rows[j] - lookback, rows[j], step)

            samples[j] = data[indices]

            targets[j] = data[rows[j] + delay][1]

        yield samples[:,::-1,:], targets
lookback = 1440

step = 6

delay = 144

batch_size = 128



train_gen = generator(float_data,

                        lookback = lookback,

                        delay=delay,

                        min_index=0,

                        max_index=200000,

                        shuffle=True,

                        step=step,

                        batch_size=batch_size)



val_gen = generator(float_data,

                        lookback = lookback,

                        delay=delay,

                        min_index=200001,

                        max_index=300000,

                        shuffle=True,

                        step=step,

                        batch_size=batch_size)



val_gen = generator(float_data,

                        lookback = lookback,

                        delay=delay,

                        min_index=300001,

                        max_index=None,

                        shuffle=True,

                        step=step,

                        batch_size=batch_size)



val_steps = (300000-200001-lookback) // batch_size

test_stpes = (len(float_data)-300001-lookback) // batch_size
from keras.models import Sequential

from keras.layers import Bidirectional, Dense, GRU

from keras.optimizers import RMSprop

model = Sequential()

model.add(Bidirectional(GRU(32),input_shape=(None, float_data.shape[-1])))

model.add(Dense(1))

model.compile(optimizer=RMSprop(),loss='mse')

history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps=val_steps)




import matplotlib.pyplot as plt

def show_all(history):

    def show(history,acc,val_acc,label):

        epochs = range(1, len(history.history[acc])+1)

        plt.plot(epochs,history.history[acc],label='Training '+label)

        plt.plot(epochs,history.history[val_acc],label='Validation '+label)

        plt.legend()

        

    plt.figure(figsize=(15,5))

    

    #plt.subplot(121)

    #show(history,'acc','val_acc','acc')

    #plt.subplot(122)

    show(history,'loss','val_loss','loss')

show_all(history)


