# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
imdb_dir = '../input/keras-imdb/aclImdb_v1/aclImdb'

train_dir = os.path.join(imdb_dir,'train')
def showdir(path, depth):

    if depth == 0:

        print("root:[" + path + "]")

 

    for item in os.listdir(path):

        if '.' not in item:

            print("|      " * depth + "|--" + item)

 

            newitem = os.path.join(path,item)

            if os.path.isdir(newitem):

                showdir(newitem, depth +1)

showdir('../',0)
labels = []

texts = []



for label_type in ['neg','pos']:

    dir_name = os.path.join(train_dir, label_type)

    for fname in os.listdir(dir_name):

        if fname[-4:] == '.txt':

            f = open(os.path.join(dir_name,fname),encoding='utf-8')

            texts.append(f.read())

            f.close

        if label_type == 'neg':

            labels.append(0)

        else:

            labels.append(1)

print(len(texts))
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



maxlen = 100

train_samples = 200

validation_samples = 10000

max_words = 10000



tokenizer = Tokenizer(num_words = max_words)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)



word_index = tokenizer.word_index



print('Use %s token words'% len(word_index))



data = pad_sequences(sequences,maxlen=maxlen)

labels = np.asarray(labels)

indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]



x_train = data[:train_samples]

y_train = labels[:train_samples]



x_val = data[train_samples:train_samples+validation_samples]

y_val = labels[train_samples:train_samples+validation_samples]
glove_dir = '../input/glove6b'



embeddings_index = {}

f = open(os.path.join(glove_dir,'glove.6B.100d.txt'),encoding='UTF-8')

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:],dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('There are %s word-vector.' % len(embeddings_index))
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))

print(embedding_matrix.shape)

for word, i in word_index.items():

    if i < max_words:

        embeddings_vector = embeddings_index.get(word)

        if embeddings_vector is not None:

            embedding_matrix[i] = embeddings_vector

from keras.models import Sequential

from keras.layers import Embedding, Flatten, Dense



model = Sequential()

model.add(Embedding(max_words,embedding_dim,input_length=maxlen))

model.add(Flatten())

model.add(Dense(32,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.summary()
model.layers[0].set_weights([embedding_matrix])

model.layers[0].trainable = False
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history = model.fit(x_train,y_train,epochs = 10,batch_size = 32,validation_data=(x_val,y_val))

model.save_weights('../working/pre_trained_glove_model.h5')
import matplotlib.pyplot as plt

def show_all(history):

    def show(history,acc,val_acc,label):

        epochs = range(1, len(history.history[acc])+1)

        plt.plot(epochs,history.history[acc],label='Training '+label)

        plt.plot(epochs,history.history[val_acc],label='Validation '+label)

        plt.title('Training and Validation '+label)

        plt.legend()

    plt.figure(figsize=(15,5))

    plt.subplot(121)

    show(history,'acc','val_acc','acc')

    plt.subplot(122)

    show(history,'loss','val_loss','loss')

show_all(history)
model_2 = Sequential()

model_2.add(Embedding(max_words,embedding_dim,input_length=maxlen))

model_2.add(Flatten())

model_2.add(Dense(32,activation='relu'))

model_2.add(Dense(1,activation='sigmoid'))

model_2.summary()
model_2.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history_2 = model_2.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))
show_all(history_2)
labels = []

texts = []

test_dir = os.path.join(imdb_dir,'test')

for label_type in ['neg','pos']:

    dir_name = os.path.join(test_dir, label_type)

    for fname in os.listdir(dir_name):

        if fname[-4:] == '.txt':

            f = open(os.path.join(dir_name,fname),encoding='utf-8')

            texts.append(f.read())

            f.close

        if label_type == 'neg':

            labels.append(0)

        else:

            labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)

x_test = pad_sequences(sequences, maxlen=maxlen)

y_test = np.asarray(labels)

    

model.load_weights('../working/pre_trained_glove_model.h5')

print(model.metrics_names)

model.evaluate(x_test, y_test)