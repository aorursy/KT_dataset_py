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
import csv

import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
# Import tweet text and target:

train_file = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train_sentences = train_file['text']

train_targets = train_file['target']



test_file = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test_sentences = test_file['text']
# add keyword and location

to_add = train_file[['keyword']]

to_add = to_add.fillna("unkown", axis=1)

# to_add = to_add['keyword'].str.cat(to_add['location'], sep = " ; ")

train_sentences = train_sentences.str.cat(to_add, sep = " ; ", join='left')



# add keyword and location

to_add = test_file[['keyword']]

to_add = to_add.fillna("unkown", axis=1)

# to_add = to_add['keyword'].str.cat(to_add['location'], sep = " ; ")

test_sentences = test_sentences.str.cat(to_add, sep = " ; ", join='left')



print(train_sentences[10])

print(test_sentences[10])
# Define hyperparameters:

vocab_size = 50000

embedding_dim = 100

max_length = 100

trunc_type='pre'

padding_type='pre'

oov_tok = "<OOV>"

training_portion = .95
# # Remove stopwords from sentences:

# stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]



# # For training sentences

# for i in range(len(train_sentences)):

#     for word in stopwords:

#         token = " "+word+" "

#         train_sentences[i] = train_sentences[i].replace(token, " ")

    

# # For testing sentences

# for i in range(len(test_sentences)):

#     for word in stopwords:

#         token = " "+word+" "

#         test_sentences[i] = test_sentences[i].replace(token, " ")
# Divide train set into training and validation set

train_size = int(len(train_sentences) * training_portion)



train_sent = train_sentences[:train_size]

train_labels = train_targets[:train_size]



validation_sent = train_sentences[train_size:]

validation_labels = train_targets[train_size:]



print(train_size)

print(len(train_sent))

print(len(train_labels))

print(len(validation_sent))

print(len(validation_labels))
# Tokenize word and sentence encoding for the training set



tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(train_sent)

word_index = tokenizer.word_index

vocab_size = len(word_index)



train_sequences = tokenizer.texts_to_sequences(train_sent)

train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)



print(len(train_sequences[0]))

print(len(train_padded[0]))



print(len(train_sequences[1]))

print(len(train_padded[1]))



print(len(train_sequences[10]))

print(len(train_padded[10]))
# Tokenize word and sentence encoding for the validation set

validation_sequences = tokenizer.texts_to_sequences(validation_sent)

validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)



print(len(validation_sequences))

print(validation_padded.shape)
# Get Glove Embedding:

embedding_index = {}

with open ('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.100d.txt') as f:

    for line in f:

        values = line.split() # split line into cols

        word = values[0] # the word is the fist col

        coefs = np.asarray(values[1:], dtype = 'float32') # the rest of the line are the word vector

        embedding_index[word] = coefs # create the dictionnary word:vectors



# Build embedding matrix that fit with the word WE tokenize in THIS exercise

embedding_matrix = np.zeros((vocab_size+1, embedding_dim)) # Create the 0 matrix with embedding matrix dimmension 

for word,i in word_index.items(): #pick a word and its index our word_index for this exercise

    embedding_vector = embedding_index.get(word) # Extract the vector of the Word from the Glove Embedding

    if embedding_vector is not None: # Discard missing vectors

        embedding_matrix[i] = embedding_vector # Save the vector in our matrix
tf.keras.backend.clear_session()

tf.random.set_seed(51)

np.random.seed(51)



# Build the model

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights = [embedding_matrix], trainable = False),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, dropout=0.2,return_sequences = True)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(25, dropout=0.2)),

    tf.keras.layers.BatchNormalization(),

    #tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(50, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(25, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model.summary()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-6 * 10**(epoch / 20))



optimizer = tf.keras.optimizers.Adam(lr=1e-6)



model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])



history = model.fit(train_padded, train_labels, batch_size=150, epochs=100, 

                    validation_data=(validation_padded, validation_labels),

                    callbacks=[lr_schedule], verbose=2)
import matplotlib.pyplot as plt



lrs = 1e-6 * (10 ** (np.arange(100) / 20))

plt.semilogx(lrs, history.history["loss"])

plt.axis([1e-6, 1,0, 1])
tf.keras.backend.clear_session()

tf.random.set_seed(51)

np.random.seed(51)



# Build the model

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights = [embedding_matrix], trainable = False),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, dropout=0.4,return_sequences = True)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(25, dropout=0.2)),

    tf.keras.layers.BatchNormalization(),

    #tf.keras.layers.GlobalAveragePooling1D(),

#     tf.keras.layers.Dense(50, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),

#     tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(25, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(5, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



optimizer = tf.keras.optimizers.Adam(lr=0.01)



reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,

                              patience=2, min_lr=1e-8, mode='auto', verbose=1)





model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])



print(model.summary())



history = model.fit(train_padded, train_labels, batch_size=450, shuffle=True, epochs=50, 

                    validation_data=(validation_padded, validation_labels),

                    callbacks=[reduce_lr], verbose=1)

import matplotlib.pyplot as plt



def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()

  

plot_graphs(history, "accuracy")

plot_graphs(history, "loss")
# prediction on test sentences::

test_sequences = tokenizer.texts_to_sequences(test_sentences)

test_padded = pad_sequences(test_sequences, padding=padding_type, maxlen=max_length)



preds = model.predict_classes(test_padded)
submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission['target']= preds

submission.to_csv("submission.csv", index=False, header=True)



submission.head(-1)