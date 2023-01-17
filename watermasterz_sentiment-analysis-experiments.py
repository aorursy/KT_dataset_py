import json
import tensorflow as tf
import csv
import random
import numpy as np
import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tqdm


import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import pandas as pd

LR = 1e-1
BATCH_SIZE = 1024
embedding_dim = 100
max_length = 30
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size=1600000
test_portion=.2

corpus = []

!wget --no-check-certificate \https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv \/input/training_cleaned.csv
!wget --no-check-certificate \https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt \/input/100d.txt
df  = pd.read_csv("training_cleaned.csv", names=['label','id', 'day', 'querytype', 'user', 'comment'])
print(len(df))
df.tail()
toxiclen = len([d for d in df['label'] if d])
nontoxiclen = len(df) - toxiclen
print(f"Number of toxic comments: {toxiclen}")
print(f"Number of non toxic comments: {nontoxiclen}")
print("The dataset is balanced")
df["label_corrected"] = [1 if l else 0 for l in df["label"]]
corpus = df[["comment", "label_corrected"]]
corpus = np.array(corpus)

print(len(corpus))
print(corpus[-1])

# Expected Output:
# 1600000
# ["is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!", 0]
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

def preprocess(text, stem=False):
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words:
      if stem:
        tokens.append(stemmer.stem(token))
      else:
        tokens.append(token)
  return " ".join(tokens)
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt 


train_labels=[]

train_sentence=[]
random.shuffle(corpus)
for x in tqdm.tqdm(range(training_size), desc='Preprocessing...'):
    sentence=corpus[x][0]
    train_labels.append(corpus[x][1])
    for word in stop_words:
        token=" "+word+" "
        sentence=sentence.replace(token," ")
    train_sentence.append(sentence)
    

print(len(train_sentence))
sentences1=[]
for sentence in tqdm.tqdm(train_sentence, desc="Changing..."):
    for word in sentence:
        word1=stemmer.stem(word)
        token=" "+word+" "
        sentence=sentence.replace(token,word1)
    sentence=sentence.replace("[^a-zA-Z#]", " ")
    sentences1.append(sentence)
train_sentence=sentences1
print(len(train_sentence))
print(train_sentence[0])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_sentence)

word_index = tokenizer.word_index
vocab_size = len(word_index)

sequences = tokenizer.texts_to_sequences(train_sentence)
pad = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


from sklearn.model_selection import train_test_split
x, testx, y, testy = train_test_split(pad, train_labels, test_size=0.2)
x = np.array(x)
y = np.array(y)

testx = np.array(testx)
testy = np.array(testy)
print(x.shape)
print(testx.shape)
print(y[2])
x[2]
np.save("testx.npy", testx)
np.save("testy.npy", testy)
print(vocab_size)
#print(word_index['bad'])

# Note this is the 100 dimension version of GloVe from Stanford
# I unzipped and hosted it on my site to make this notebook easier

# Note2: using 50d version of GloVe as well to compare which one does better
# !wget --no-check-certificate \ https://www.kaggle.com/watts2/glove6b50dtxt  \/content/glove.6B.50d.txt
embeddings_index = {};
with open('../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec') as f:
    for line in tqdm.tqdm(f, desc='loading embeddings....'):
        values = line.split()
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
embedding_dim = 300
embeddings_matrix = np.zeros((vocab_size+1, embedding_dim))
for word, i in tqdm.tqdm(word_index.items(), desc='making embedding matrix....'):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

print('Found %s word vectors.' %len(embeddings_index))

print(len(embeddings_matrix))
# Expected Output
# 138859
def lrs(epoch, lr):
    if epoch < 4:
        return lr
    else:
        return lr/(0.5 * epoch)

cb = tf.keras.callbacks.LearningRateScheduler(lrs)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.SpatialDropout1D(0.2),
   
    #tf.keras.layers.Conv1D(64, 5, activation='relu'),
    #tf.keras.layers.Conv1D(128, 5, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,dropout=0.2,recurrent_dropout=0.2)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.01),metrics=['accuracy'])
Rlr = ReduceLROnPlateau(             factor=0.1,
                                     min_lr = 0.001,
                                     monitor = 'val_loss',
                                     verbose = 1)
model.summary()

num_epochs = 30
history = model.fit(x, y, batch_size=int(x.shape[0]/200), epochs=num_epochs, validation_data=(testx, testy), callbacks=[Rlr])

print("Training Complete")

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])

plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])

plt.figure()


# Expected Output
# A chart where the validation loss does not increase sharply!
model.save("100dLSTM-FastText.h5")
np.save("wordidx.npy", tokenizer.word_index)

file.remove("./100dLSTMCNN-nothing.h5")
