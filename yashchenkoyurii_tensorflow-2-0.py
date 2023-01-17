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
import tensorflow as tf

from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow_hub as hub
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head()
test.head()
X_train, X_test, y_train, y_test = train_test_split(train[['text']].values, train[['target']], test_size=0.2, random_state=2012)
tokenizer = Tokenizer()



vocab_size = 14000

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')

tokenizer.fit_on_texts(X_train[:,0])
def len_of_set(data):

  lengths = []



  for item in data:

    lengths.append(len(item.split()))

      

  length = np.array(lengths)



  return {

      'max': length.max(), 'min': length.min(), 'avg': np.median(length)

      }



len_of_set(X_train[:, 0])
max_len = 20



X_seq_train = pad_sequences(tokenizer.texts_to_sequences(X_train[:, 0]), padding='post', maxlen=max_len)

X_seq_test = pad_sequences(tokenizer.texts_to_sequences(X_test[:, 0]), padding='post', maxlen=max_len)
sequence = tf.keras.Input(shape=(max_len,))

embdeddings = layers.Embedding(vocab_size, 16)(sequence)



pooling = layers.GlobalAveragePooling1D()(embdeddings)



output = layers.Dense(1, activation='sigmoid')(pooling)



model = tf.keras.Model(inputs=sequence, outputs=output)



model.compile(optimizer=tf.optimizers.Adam(1e-3),

              loss='mae',

              metrics=['accuracy'])



model.summary()
model.fit(X_seq_train, y_train,

        epochs=21,

        validation_data=(X_seq_test, y_test))
sub_padded = pad_sequences(tokenizer.texts_to_sequences(test[['text']].values[:,0]), padding='post', maxlen=max_len)
test['target'] = model.predict(sub_padded)

test['target'] = test['target'].apply(lambda x: int(x > 0.5))
sub = test[['id', 'target']]
sub.head()
sub.to_csv('submission.csv', index=False)