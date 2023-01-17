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
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

import re

from wordcloud import WordCloud

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json",lines=True) #Read the file as a json object per line.
df.head()
df.shape #check dimensions
df.isna().sum() #check for any nulls
df2 = df.iloc[:, :2] #article link is not required for analysis. All saracastic articles - TheOnion, rest are - HuffPost
df2.head(3)
plt.figure(figsize=(12,8)) #Checking for class imbalance

ax = sns.countplot(x="is_sarcastic", data=df2)
plt.title('Distribution of News Headlines')
plt.xlabel('Type of Headline')
plt.ylabel('Frequency')

df2['headline'].describe()
pd.set_option('display.max_colwidth', -1) #show all text in col
pd.set_option('display.max_rows', None) #show all rows
df2['headline'].head(50)#observe col to see areas for cleaning
nltk.download('stopwords')
stop = list(stopwords.words('english'))
stop[:20]

def lowercase(text): #convert to lower case
      return text.lower()

def remove_punct(text): #remove punctuations using regex
      return re.sub('[^a-z]+',' ',text)

def remove_stopwords(text):  #remove stopwords to aavoid overfitting
    mylist = []
    for i in text.split():
        if i not in stop:
            mylist.append(i)
    return " ".join(mylist)


def cleanmepls(text): #fn to clean col
  text = lowercase(text)
  text = remove_punct(text)
  text = remove_stopwords(text)
  return text

df2['headline_new'] = df2['headline'].apply(cleanmepls)

df2.head(20) #observe changes
del df2['headline']
plt.figure(figsize = (20,20)) #fig size
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df2[df2['is_sarcastic'] == 0]['headline_new'])) #News that is Not Sarcastic
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20)) #fig size
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df2[df2['is_sarcastic'] == 1]['headline_new'])) #News that is Sarcastic
plt.imshow(wc , interpolation = 'bilinear')
len_not_sarcastic = df2[df2['is_sarcastic']==0]['headline_new'].str.len() #find length of headline (all characters including spaces)
len_sarcastic = df2[df2['is_sarcastic']==1]['headline_new'].str.len()
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5)) #subplots to plot side by side

ax1.hist(len_not_sarcastic,color='red')
ax1.set_title('Not Sarcastic')

ax2.hist(len_sarcastic,color='green')
ax2.set_title('Sarcastic')

fig.suptitle('Length of each headline')
plt.show()
count_not_sarcastic = df2[df2['is_sarcastic']==0]['headline_new'].str.split() #split headline into words
cns=count_not_sarcastic.map(lambda x: len(x)) #find no. of words in a headline

count_sarcastic = df2[df2['is_sarcastic']==1]['headline_new'].str.split()
cs=count_sarcastic.map(lambda x: len(x))
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5)) #subplots to plot side by side

ax1.hist(cns,color='red')
ax1.set_title('Not Sarcastic')

ax2.hist(cs,color='green')
ax2.set_title('Sarcastic')

fig.suptitle('Count of words in each headline')
plt.show() 
avg_not_sarcastic = df2[df2['is_sarcastic']==0]['headline_new'].map(lambda x: [len(i) for i in x.split()]) #length of each word in a news headline
ansc=avg_not_sarcastic.map(lambda x: np.mean(x)) #mean of those lengths

avg_sarcastic = df2[df2['is_sarcastic']==1]['headline_new'].map(lambda x: [len(i) for i in x.split()])
asc=avg_sarcastic.map(lambda x: np.mean(x)) 
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5)) #subplots to plot side by side

ax1.hist(ansc,color='red')
ax1.set_title('Not Sarcastic')

ax2.hist(asc,color='green')
ax2.set_title('Sarcastic')

fig.suptitle('Count of words in each headline')
plt.show() 
embedding_dim = 200
max_length = 20 #20 words per headline max
trunc_type = 'post'
pad_type = 'post'
training_size = len(df2)
test_portion = 0.3
training_size
tokenizer = Tokenizer() #create tokenizer object 
tokenizer.fit_on_texts(df2['headline_new']) #create word index dict

word_index_dict = tokenizer.word_index #get word index dict
vocab_size = len(word_index_dict) + 1

vocab_size
sequence = tokenizer.texts_to_sequences(df2['headline_new']) #convert words to their vector representations

padding = pad_sequences(sequences=sequence, maxlen=max_length, padding=pad_type, truncating=trunc_type) #pad all sequences, to ensure same length

split = int(test_portion*training_size)
split
test_sequences = padding[0:split] 
training_sequences = padding[split:]

labels = df2['is_sarcastic']
test_labels = labels[0:split]
training_labels = labels[split:]
embedding_dict = {} #create embedding dict with words and their 200-D representations
#../input/glove-twitter/glove.twitter.27B.200d.txt
with open('../input/glove6b200d/glove.6B.200d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float64')
        embedding_dict[word] = coefs
embedding_matrix = np.zeros((vocab_size, embedding_dim)) #create embedding matrix to store weights
embedding_matrix.shape
for word, index in word_index_dict.items(): #update embeddings matrix with their GloVe 100-D weights
  embedding_vector = embedding_dict.get(word)
  if embedding_vector is not None:
    embedding_matrix[index] = embedding_vector
test_sequences = np.asarray(test_sequences, dtype='int64') #covert to array
training_sequences = np.asarray(training_sequences, dtype='int64')

test_labels = np.asarray(test_labels)
training_labels = np.asarray(training_labels)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False), #we don't want to update/change the learned weights
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.4, return_sequences=False)), #using LSTM gate to overcome vanishing Gradient problem due to long term dependancies
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics='accuracy')
num_epochs = 10
history = model.fit(x=training_sequences, y=training_labels, validation_data=(test_sequences, test_labels), epochs = num_epochs, verbose = 1)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = [i for i in range(10)]

fig, ax = plt.subplots(1,2)

ax[0].plot(epochs, acc, 'r', label='Training accuracy')
ax[0].plot(epochs, val_acc, 'g', label='Validation accuracy')
ax[0].set_title('Training and validation accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")


ax[1].plot(epochs, loss, 'r', label='Training Loss')
ax[1].plot(epochs, val_loss, 'g', label='Validation Loss')
ax[1].set_title('Training and validation loss')
ax[1].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

fig.set_size_inches(15,7)
plt.show()
num_epochs = 4
history = model.fit(x=training_sequences, y=training_labels, validation_data=(test_sequences, test_labels), epochs = num_epochs, verbose = 1)

print("Training Loss and Accuracy: ")
loss, accuracy = model.evaluate(training_sequences, training_labels, verbose = 1)
print("Validation Loss and Accuracy: ")
loss, accuracy = model.evaluate(test_sequences, test_labels, verbose = 1)