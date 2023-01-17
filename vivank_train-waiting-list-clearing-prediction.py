# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
TB_Data_2 = pd.read_csv("../input/TB_Data_2.csv")
TB_Data_2.head()
TB_Data_2.describe(include='all')
f = plt.figure(figsize=(19, 15))
plt.matshow(TB_Data_2.corr(), fignum=f.number)
plt.xticks(range(TB_Data_2.shape[1]), TB_Data_2.columns, fontsize=14, rotation=45)
plt.yticks(range(TB_Data_2.shape[1]), TB_Data_2.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
wt = TB_Data_2['Waitlist-cancellations'].value_counts().sort_values().plot(kind = 'barh',lw=2, colormap='Spectral',title="Waitlist-cancellations Percentage per day")
wt.set_xlabel("Number of days the percentage was similar")
wt.set_ylabel("Waitlist cancellation percentage");
ws = TB_Data_2['Astatus'].value_counts().sort_values().plot(kind = 'bar',lw=2, colormap='Spectral',title="Number of waitlisted tickets")
ws.set_xlabel("Ticket status (Here 1=Waitlisted and 0=Confirmed)")
ws.set_ylabel("Number of Tickets Booked");
cl = TB_Data_2['ClassOfTravel'].value_counts().sort_values().plot(kind = 'bar',lw=2, colormap='Spectral',title="Class of Travel")
cl.set_xlabel("Class of Travel")
cl.set_ylabel("Count of Passengers in the Class");
cl = TB_Data_2['BookingStatus'].value_counts().sort_values().plot(kind = 'bar',lw=2, colormap='Spectral',title="Booking Status")
cl.set_xlabel("Waiting List Type")
cl.set_ylabel("Number of Tickets Booked");
# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Astatus')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds
feature_columns = []

#numeric columns
for header in ['Wno', 'Weekend', 'Month-of-journey', 'Days', 'Upcoming-Festival-Weekends', 'Waitlist-cancellations', 'Allotted-seats']:
  feature_columns.append(feature_column.numeric_column(header))

# indicator cols
bs = feature_column.categorical_column_with_vocabulary_list(
      'BookingStatus', ['WL', 'TQWL'])
bs_one_hot = feature_column.indicator_column(bs)
feature_columns.append(bs_one_hot)

ct = feature_column.categorical_column_with_vocabulary_list(
      'ClassOfTravel', ['2A', '3A','SL'])
ct_one_hot = feature_column.indicator_column(ct)
feature_columns.append(ct_one_hot)

# embedding cols
bs_embedding = feature_column.embedding_column(bs, dimension=8)
feature_columns.append(bs_embedding)

ct_embedding = feature_column.embedding_column(ct, dimension=8)
feature_columns.append(ct_embedding)

for i in feature_columns:
    print(i)
    print()
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
train, test = train_test_split(TB_Data_2, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of class of travel:', feature_batch['ClassOfTravel'])
  print('A batch of targets:', label_batch )
model = tf.keras.Sequential([
      feature_layer,
      layers.Dense(64, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(256, activation='relu'),
      layers.Dropout(0.2),
      layers.Dense(512, activation='relu'),
      layers.Dense(1024, activation='relu'),
      layers.Dense(1)
])

model.compile(optimizer='Adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=100)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training set accuracy versus Validation set accuracy')
plt.legend(loc='lower right');

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training set Loss versus Validation set Loss')
plt.legend(loc='lower right');


loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
# save entire model to a HDF5 file
model.save('my_model.h5')