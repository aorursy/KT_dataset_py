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

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

import tensorflow_datasets as tfds
df_true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

df_fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
df_true.head()
df_fake.head()
df_true.drop(['title', 'subject', 'date'], axis = 1, inplace = True)

df_fake.drop(['title', 'subject', 'date'], axis = 1, inplace = True)
df_true['label'] = 1

df_fake['label'] = 0
df = pd.concat([df_true, df_fake], ignore_index = True)
num_examples = len(df)
#Step 1: Create a dataset

ds_raw = tf.data.Dataset.from_tensor_slices((df['text'].values, df['label'].values))

ds_raw = ds_raw.shuffle(num_examples, reshuffle_each_iteration = False)

for example in ds_raw.take(1):

    print(example[0].numpy()[:100])

    print(example[1])
#Train Val Test Splits

print(num_examples)



ds_temp = ds_raw.take(40000)

ds_test_raw = ds_raw.skip(40000)

ds_train_raw = ds_temp.take(38000)

ds_val_raw = ds_temp.skip(38000)
#Step 2: Find unique tokens

from collections import Counter

token_count = Counter()



tokenizer = tfds.features.text.Tokenizer()



for example in ds_train_raw:

    tokens = tokenizer.tokenize(example[0].numpy())

    token_count.update(tokens)
print(len(token_count))
#Step 3: Token Encoding

encoder = tfds.features.text.TokenTextEncoder(token_count)



def encoding_fn(text, label):

    return encoder.encode(text.numpy()), tf.cast(label, tf.int32)



def encoding_fn_eager(text, label):

    return tf.py_function(encoding_fn, [text, label], [tf.int32, tf.int32])

    



ds_train_enc = ds_train_raw.map(encoding_fn_eager)

ds_val_enc = ds_val_raw.map(encoding_fn_eager)

ds_test_enc = ds_test_raw.map(encoding_fn_eager)



for example in ds_train_enc.take(5):

    print(example[0][:10])
#Step 4: batch and padding

ds_train = ds_train_enc.padded_batch(32, ([None], []))

ds_val = ds_val_enc.padded_batch(32, ([None], []))

ds_test = ds_test_enc.padded_batch(32, ([None], []))
#Step 5: build the model

model = tf.keras.Sequential()

#Embedding layer adds dimension for features

model.add(tf.keras.layers.Embedding(input_dim = len(token_count) + 2, output_dim = 20)) #reason for +2 in input_dim: +1 for 0 paddings; +1 for oov token

#result in params of input_dim * output_dim



#add bidirectional(optional) of LSTM

model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))) #LSTM collapsed the sequence dimension by only returning last element in the sequence

model.add(tf.keras.layers.Dense(32, activation = 'relu'))

model.add(tf.keras.layers.Dense(1))



model.summary()
model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), metrics = ['accuracy'])
history = model.fit(ds_train.prefetch(buffer_size = tf.data.experimental.AUTOTUNE), validation_data = ds_val.cache(), epochs = 5)
import matplotlib.pyplot as plt
hist = history.history

epochs = np.arange(5) + 1



fig, ax = plt.subplots(1, 2, figsize = (20, 8))



ax[0].plot(epochs, hist['loss'], '-o', label = 'training_loss')

ax[0].plot(epochs, hist['val_loss'], '--<', label = 'val_loss')

ax[0].legend()

ax[0].set_title('Training vs. Val Loss')



ax[1].plot(epochs, hist['accuracy'], '-o', label = 'training_accuracy')

ax[1].plot(epochs, hist['val_accuracy'], '--<', label = 'val_accuracy')

ax[1].legend()

ax[1].set_title('Training vs. Val Accuracy')
model.evaluate(ds_test)