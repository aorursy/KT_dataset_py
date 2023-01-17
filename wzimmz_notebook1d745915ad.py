# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import tflearn 

from tflearn.data_utils import load_csv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


data, labels = load_csv("../input/train.csv", target_column=1,

                        categorical_labels=True, n_classes=2)
print(data)
# Preprocessing function

def preprocess(data, columns_to_ignore):

    # Sort by descending id and delete columns

    for id in sorted(columns_to_ignore, reverse=True):

        [r.pop(id) for r in data]

    for i in range(len(data)):

      # Converting 'sex' field to float (id is 1 after removing labels column)

      data[i][1] = 1. if data[i][1] == 'female' else 0.

    return np.array(data, dtype=np.float32)



# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)

to_ignore=[0,2,5]



# Preprocess data

data = preprocess(data, to_ignore)

print(len(data))
# Build neural network

net = tflearn.input_data(shape=[None, 3])

net = tflearn.fully_connected(net, 32)

net = tflearn.fully_connected(net, 32)

net = tflearn.fully_connected(net, 2, activation='softmax')

net = tflearn.regression(net)
# Define model

model = tflearn.DNN(net)

# Start training (apply gradient descent algorithm)

model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)