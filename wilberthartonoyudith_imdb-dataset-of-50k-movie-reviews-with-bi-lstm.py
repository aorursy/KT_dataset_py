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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
df=pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df
df.isnull().sum()#to check is there any null value
df.isna().sum()#checking na value
df['sentiment'].unique()
le = LabelEncoder()
feature = df['review'].values
label = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size = 0.2)#split the data

y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
df['sentiment'] = le.fit_transform(df['sentiment'])
#then we try to remove our special character 

#only store the char and space
def special_chars(text):
        alphanumeric=""
        for character in text:
            if character.isalpha() or character==" ":
                alphanumeric += character
        return alphanumeric
    
#change the special char into whitespace
def tags(text):
     return re.compile(r"<[^>]+>").sub(" ", text)

#change the number char into whitespace
def num(text):
     return "".join(re.sub(r"([0–9]+)"," ",text))

#jalankan fungsi
df.review=df.review.apply(lambda x : tags(x))
df.review=df.review.apply(lambda x : num(x))
df.review=(df.review).apply(special_chars)

#menampilkan data hasil fungsi
df.head()
#before we remove the special char
#A wonderful little production. <br /><br />The...	

#after we remove the special char 

#A wonderful little production The filming te...
df.head(5)
pos = df['sentiment'] == 1
neg = df['sentiment'] == 0
temp = [pos.sum(),neg.sum()]
plt.pie(temp, labels = ['positive', 'negative'],autopct = '%1.1f%%')
plt.show()
#tokenizer
tokenizer = Tokenizer(num_words=10000,oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

#sequencing
training_sequence = tokenizer.texts_to_sequences(X_train)
testing_sequence = tokenizer.texts_to_sequences(X_test)

#padding
train_pad_sequence = pad_sequences(training_sequence,maxlen = 500,truncating= 'post',padding = 'pre')
test_pad_sequence = pad_sequences(testing_sequence,maxlen = 500,truncating= 'post',padding = 'pre')
print('Total Unique Words : {}'.format(len(word_index)))
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip.2
!ls
#embedding
embedded_words = {}
with open ('./glove.6B.200d.txt') as file:
  for line in file:
    words, coeff = line.split(maxsplit=1)
    coeff = np.array(coeff.split(),dtype = float)
    embedded_words[words] = coeff

embedding_matrix = np.zeros((len(word_index) + 1,200))
for word, i in word_index.items():
  embedding_vector = embedded_words.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector
#using the sequential and Bi-LSTM model
model = tf.keras.Sequential([tf.keras.layers.Embedding(len(word_index) + 1,200,weights=[embedding_matrix],input_length=500,
                            trainable=False),
                             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(256,activation = 'relu',),
                             tf.keras.layers.Dense(128,activation = 'relu'),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(1,activation = tf.nn.sigmoid)])
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
#creating callback
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.98):
      print("\n Accuracy == 98%!")
      self.model.stop_training = True
callbacks = myCallback()
num_epochs = 30
history = model.fit(train_pad_sequence,
                    y_train,
                    epochs = num_epochs,
                    validation_data=(test_pad_sequence,y_test),
                    callbacks=[callbacks])