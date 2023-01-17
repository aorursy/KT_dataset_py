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
import re

from sklearn.model_selection import train_test_split
df = pd.read_csv(r"/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
df.info()
df.head()
sample_text = df.review[1]

print(sample_text)
print(sample_text)

print()

#lower case 

t = sample_text.lower()

print(t)

print()





#removing_non_ascii_char

t1 = re.sub(r'[^\x00-\x7F]+','', t)

print(t1)

print(len(t1))

print()



#removing html tags 

clean = re.compile('<.*?>')

t2 = re.sub(clean, '', t1)

print(t2)

print(len(t2))

print()



#removing puncuations

t3 = re.sub(r'[^\w\s]', '', t2)

print(t3)

print(len(t3))

print()



#removing numbers 

t4 = re.sub("\d+", "", t3)

print(t4)

print(len(t4))

print()
clean = re.compile('<.*?>')

df.review = df.review.apply(lambda x: x.lower())

df.review = df.review.apply(lambda x: re.sub(r'[^\x00-\x7F]+','', x))

df.review = df.review.apply(lambda x: re.sub(clean, '', x))

df.review = df.review.apply(lambda x: re.sub("\d+", "", x))
polarity = []

for i in df.sentiment:

    if i == "positive":

        polarity.append(1)

    else:

        polarity.append(0)
df["polarity"] = polarity

df.head(15)
X = df.review.values

y = df.polarity.values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X.shape)

print(y.shape)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers.embeddings import Embedding

from keras.layers import Dense

from keras.layers import Flatten

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from keras.layers import Dropout
tokenizer = Tokenizer()

tokenizer.fit_on_texts(x_train)

train_sequences = tokenizer.texts_to_sequences(x_train)

test_sequences = tokenizer.texts_to_sequences(x_test)
count = 0 

for i in range(40000):

    count = count + len(train_sequences[i])

print(count)

print(count/40000)
#parameters: 

MAX_SEQUENCE_LENGTH = 230

EMBEDDING_DIM = 50 #vector dimesnion of a single word - large will be sparse , less will miss the information carried by that word  (hence trade off)

vocab_size = len(tokenizer.word_index) + 1
X_train = pad_sequences(train_sequences, MAX_SEQUENCE_LENGTH, truncating = "post", padding='post')



X_test = pad_sequences(test_sequences, MAX_SEQUENCE_LENGTH,  truncating = "post", padding='post')
embedding_layer = Embedding(input_dim = vocab_size,

                            output_dim = EMBEDDING_DIM,

                            input_length = MAX_SEQUENCE_LENGTH)
model = Sequential()

model.add(embedding_layer)

model.add(Flatten())

model.add(Dense(1024, activation = "relu"))

model.add(Dense(512, activation = "relu"))

model.add(Dropout(0.1))

model.add(Dense(256, activation = "relu"))

model.add(Dense(2, activation='softmax'))



# compile the keras mode

model.compile(optimizer='adam', 

              loss= tf.keras.losses.SparseCategoricalCrossentropy(),

              metrics=['accuracy'])



print(model.summary())



# fit the keras model on the dataset

model_fit = model.fit(X_train, y_train, epochs=10, batch_size = 500, verbose=1, validation_split=0.1)



# evaluate the keras model

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

print('Accuracy: %f' % (accuracy*100))
import matplotlib.pyplot as plt

plt.plot(model_fit.history['accuracy'])

plt.plot(model_fit.history['val_accuracy'])



plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train','test'], loc = 'upper left')

plt.show()



plt.plot(model_fit.history['loss'])

plt.plot(model_fit.history['val_loss'])



plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train','test'], loc = 'upper left')

plt.show()