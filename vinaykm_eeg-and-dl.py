# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.layers import Embedding

from keras.layers import LSTM, CuDNNLSTM, BatchNormalization, Conv2D, Flatten, MaxPooling2D, Dropout

from keras.optimizers import Adam

from sklearn.metrics import roc_curve, auc, roc_auc_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

load = 1

time_steps = 1000

subsample = 100

x_paths = ["train/subj1_series1_data.csv", "train/subj1_series2_data.csv", "train/subj1_series3_data.csv", "train/subj1_series4_data.csv", \

          "train/subj2_series1_data.csv", "train/subj2_series2_data.csv", "train/subj2_series3_data.csv", "train/subj2_series4_data.csv", \

          "train/subj3_series1_data.csv", "train/subj3_series2_data.csv", "train/subj3_series3_data.csv", "train/subj3_series4_data.csv", \

          "train/subj4_series1_data.csv", "train/subj4_series2_data.csv", "train/subj4_series3_data.csv", "train/subj4_series4_data.csv",\

          "train/subj5_series1_data.csv", "train/subj5_series2_data.csv", "train/subj5_series3_data.csv", "train/subj5_series4_data.csv",\

          "train/subj6_series1_data.csv", "train/subj6_series2_data.csv", "train/subj6_series3_data.csv", "train/subj6_series4_data.csv"]

y_paths = ["train/subj1_series1_events.csv", "train/subj1_series2_events.csv", "train/subj1_series3_events.csv", "train/subj1_series4_events.csv", \

          "train/subj2_series1_events.csv", "train/subj2_series2_events.csv", "train/subj2_series3_events.csv", "train/subj2_series4_events.csv", \

          "train/subj3_series1_events.csv", "train/subj3_series2_events.csv", "train/subj3_series3_events.csv", "train/subj3_series4_events.csv", \

          "train/subj4_series1_events.csv", "train/subj4_series2_events.csv", "train/subj4_series3_events.csv", "train/subj4_series4_events.csv", \

          "train/subj5_series1_events.csv", "train/subj5_series2_events.csv", "train/subj5_series3_events.csv", "train/subj5_series4_events.csv", \

          "train/subj6_series1_events.csv", "train/subj6_series2_events.csv", "train/subj6_series3_events.csv", "train/subj6_series4_events.csv"]



if load:

    x_data = []

    for x_path in x_paths:

        x_path = "../input/" + x_path

        with open(x_path) as file:

            for line in file:

                if line[0] == "i": continue

                line_array = []

                for word in range(len(line.split(','))-1):

                    word+=1

                    line_array.append(int(line.split(',')[word]))

                line_array = np.asarray(line_array)

                x_data.append(line_array)

    x_data = np.asarray(x_data)    

        

  

    y_data = []

    for y_path in y_paths:

        y_path = "../input/" + y_path

        with open(y_path) as file:

            for line in file:

                if line[0] == "i": continue

                line_array = []

                for word in range(len(line.split(','))-1):

                    word+=1

                    line_array.append(int(line.split(',')[word]))

                line_array = np.asarray(line_array)

                y_data.append(line_array)

    y_data = np.asarray(y_data)



print(x_data)

    
model = Sequential()

#model.add(CuDNNLSTM(128, input_shape = (time_steps//subsample, 32)))

model.add(Conv2D(filters = 64, kernel_size = (7,7), padding = "same", activation = "relu", input_shape = (time_steps//subsample, 32, 1)))

model.add(BatchNormalization())

#model.add(MaxPooling2D(pool_size = (3,3)))

model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = "same", activation = "relu", input_shape = (time_steps//subsample, 32, 1)))

model.add(BatchNormalization())

#model.add(MaxPooling2D(pool_size = (3,3)))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu", input_shape = (time_steps//subsample, 32, 1)))

model.add(BatchNormalization())

#model.add(MaxPooling2D(pool_size = (3,3)))

model.add(Flatten())

#model.add(Dropout(0.2))

model.add(Dense(32, activation = "relu"))

model.add(BatchNormalization())

# model.add(Dropout(0.2))

model.add(Dense(6, activation = "sigmoid"))





adam = Adam(lr = 0.001)



model.compile(optimizer = adam, loss = "categorical_crossentropy", metrics = ["accuracy"])



model.summary()



def generator(batch_size):

    while 1:

        

        x_time_data = np.zeros((batch_size, time_steps//subsample, 32))

        yy = []

        for i in range(batch_size):

            random_index = np.random.randint(0, len(x_data)-time_steps)

            x_time_data[i] = x_data[random_index:random_index+time_steps:subsample]

            yy.append(y_data[random_index + time_steps])

        yy = np.asarray(yy)

        yield x_time_data.reshape((x_time_data.shape[0],x_time_data.shape[1], x_time_data.shape[2], 1)), yy



model.fit_generator(generator(32), steps_per_epoch = 5000, epochs = 3)
x_val_path = "../input/train/subj4_series7_data.csv"

y_val_path = "../input/train/subj4_series7_events.csv"

x_val_data = []

with open(x_val_path) as file:

    for line in file:

        if line[0] == "i": continue

        line_array = []

        for word in range(len(line.split(','))-1):

            word+=1

            line_array.append(int(line.split(',')[word]))

        line_array = np.asarray(line_array)

        x_val_data.append(line_array)

x_val_data = np.asarray(x_val_data)    

        

y_val_data = []

with open(y_val_path) as file:

    for line in file:

        if line[0] == "i": continue

        line_array = []

        for word in range(len(line.split(','))-1):

            word+=1

            line_array.append(int(line.split(',')[word]))

        line_array = np.asarray(line_array)

        y_val_data.append(line_array)

y_val_data = np.asarray(y_val_data)



def val_generator():

    while 1:

        batch_size = 1

        x_time_data = np.zeros((batch_size, time_steps//subsample, 32))

        yy = []

        for i in range(batch_size):

            random_index = np.random.randint(0, len(x_val_data)-time_steps)

            x_time_data[i] = x_val_data[random_index:random_index+time_steps:subsample]

            yy.append(y_val_data[random_index + time_steps])

        yy = np.asarray(yy)

        yield x_time_data.reshape((x_time_data.shape[0],x_time_data.shape[1], x_time_data.shape[2], 1)), yy
gen_data = val_generator()

scores = []

num_test = 1000

for i in range(num_test):

    x_test, y_test = next(gen_data)

    while not 1 in y_test:

#         print(y_test)

        x_test, y_test = next(gen_data)



#     print(y_test)

    y_out = model.predict(x_test).reshape((6,1))

#     print(y_out)

    scores.append(roc_auc_score(y_test.reshape((6,1)), y_out))

scores = np.asarray(scores)

print("Mean ", np.mean(scores))