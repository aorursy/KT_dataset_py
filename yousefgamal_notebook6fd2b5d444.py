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
Twitter_df = pd.read_csv('/kaggle/input/twitter-and-reddit-sentimental-analysis-dataset/Twitter_Data.csv')
Reddit_df = pd.read_csv('/kaggle/input/twitter-and-reddit-sentimental-analysis-dataset/Reddit_Data.csv')
Twitter_df.head()
Reddit_df.head()
Twitter_df.dropna(inplace=True)
Reddit_df.dropna(inplace=True)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,RNN,Bidirectional,Embedding, LSTM, GRU, Flatten, Conv1D, Input,Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
# Making the tokinzer ready for using 
tokinzer = Tokenizer()

# Fit the data into the tokinzer: 
tokinzer.fit_on_texts(list(Twitter_df['clean_text']))

text_sequence = tokinzer.texts_to_sequences(list(Twitter_df['clean_text']))
# Global Variables for the model
max_length = max([len(x) for x in text_sequence])
max_words = len(tokinzer.word_index) + 1
pad_sequence = pad_sequences(text_sequence,maxlen = max_length, padding = 'post')
labels = to_categorical(Twitter_df['category'],num_classes=3)
Reddit_labels = to_categorical(Reddit_df['category'],num_classes=3)
reddit_sequence = tokinzer.texts_to_sequences(list(Twitter_df['clean_text']))
reddit_padded = pad_sequences(reddit_sequence,maxlen = max_length, padding = 'post')
def Model():
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length= max_length - 1))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.3))
    model.add(Dense(256,activation = 'relu'))
    model.add(Dense(125,activation = 'relu'))
    model.add(Dense(3,activation = 'softmax'))
    return model
model = Model()
model.compile(optimizer=Adam(),loss=CategoricalCrossentropy(),metrics=['accuracy'])
hist = model.fit(pad_sequence,labels,epochs = 2)
def classifier(text):
    review_seq = tokinzer.texts_to_sequences([text])
    review_padded = pad_sequences(review_seq,maxlen = max_length, padding = 'post')
    prediction = model.predict(review_padded)
    max_classifier = np.argmax(prediction)
    if max_classifier == 0:
        return 0
    elif max_classifier == 1:
        return 1
    elif max_classifier == 2:
        return -1
    
review = "before 2014 hindustan has seen the worst for hindus own maj hindu rashtra who thrashed the rascal faces these anti indian politiciansantinationals urban naxals wait watch after modis win pakistan mein bhi hindu hona garv baat hogiâœŒ"
classifier(review)
# Testing for user: 
user_tweet = input("Enter your tweet: ")
classing = classifier(user_tweet)
print("The class of your tweet is: ", classing)
