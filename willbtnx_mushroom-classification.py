# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf



from tensorflow.keras import datasets, layers, models



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

mushrooms_df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
mushrooms_df.head()
data_c = mushrooms_df.loc[:, mushrooms_df.columns != 'odor'].to_numpy()

labels_c = mushrooms_df[['odor']].to_numpy()
enc_data = OneHotEncoder(handle_unknown='ignore')

enc_data.fit(data_c)



enc_labels = OneHotEncoder(handle_unknown='ignore')

enc_labels.fit(labels_c)



data = enc_data.transform(data_c).todense()

labels = enc_labels.transform(labels_c).todense()
print(data.shape)

print(labels.shape)



N = data.shape[0]
seed = np.random.get_state()

np.random.shuffle(data)

np.random.set_state(seed)

np.random.shuffle(labels)
split_idx = int(N * 0.8)

train_data = data[0:split_idx]

train_labels = labels[0:split_idx]

test_data = data[split_idx:N]

test_labels = labels[split_idx:N]
model = models.Sequential([

    layers.Dense(data.shape[1], input_shape=(data.shape[1],), activation='relu'),

    layers.Dropout(0.2),

    layers.Dense(labels.shape[1])

])

model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])



history = model.fit(train_data, train_labels, epochs=10, 

                    validation_data=(test_data, test_labels))