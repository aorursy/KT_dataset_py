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
import pandas as pd



true=pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

false=pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')



true=true.drop('title',axis=1)

true=true.drop('subject',axis=1)

true=true.drop('date',axis=1)

false=false.drop('title',axis=1)

false=false.drop('subject',axis=1)

false=false.drop('date',axis=1)

false['label'] = 0

true['label'] = 1



data=pd.concat([true, false],ignore_index=True)



texts = data['text']

labels = data['label']

x=texts

y=labels



# vectorize the text samples into a 2D integer tensor 



MAX_SEQUENCE_LENGTH = 5000

MAX_NUM_WORDS = 25000

EMBEDDING_DIM = 300

TEST_SPLIT = 0.2



# vectorize the text samples into a 2D integer tensor 



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical



tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)



word_index = tokenizer.word_index

num_words = min(MAX_NUM_WORDS, len(word_index)) + 1

data = pad_sequences(sequences, 

                     maxlen=MAX_SEQUENCE_LENGTH, 

                     padding='pre', 

                     truncating='pre')



print('Found %s unique tokens.' % len(word_index))

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', labels.shape)

from sklearn.model_selection import train_test_split

 

x_train, x_val, y_train, y_val = train_test_split(data, 

                                                  labels.apply(lambda x: 0 if x == 0 else 1), 

                                                  test_size=TEST_SPLIT)



from keras import layers

from keras.models import Sequential

MAX_SEQUENCE_LENGTH = 5000

MAX_NUM_WORDS = 25000

EMBEDDING_DIM = 300

TEST_SPLIT = 0.2

model = Sequential(

    [

        # part 1: word and sequence processing

        layers.Embedding(num_words,

                         EMBEDDING_DIM, 

                         input_length=MAX_SEQUENCE_LENGTH,

                         trainable=True),

        layers.Conv1D(128, 5, activation='relu'),

        layers.GlobalMaxPooling1D(),

        

        # part 2: classification

        layers.Dense(128, activation='relu'),

        layers.Dense(1, activation='sigmoid')

    ])



model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])

history = model.fit(x_train, 

                    y_train,

                    batch_size=128,

                    epochs=1,

                    validation_data=(x_val, y_val))




