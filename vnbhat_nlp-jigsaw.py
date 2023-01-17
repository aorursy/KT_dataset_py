# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from pprint import pprint  # pretty-printer

from collections import defaultdict

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from gensim.parsing.preprocessing import STOPWORDS



import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from keras.models import Sequential

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation



from keras.layers import Embedding

from keras.initializers import Constant

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
documents = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')



# remove common words and tokenize

texts = [[word for word in document.lower().split() if word not in STOPWORDS]

         for document in documents['comment_text'].values.tolist()]



# remove words that appear only once

frequency = defaultdict(int)

for text in texts:

    for token in text:

        frequency[token] += 1



train_texts = [[token for token in text if frequency[token] > 1]for text in texts]

### test preprocessing



texts = [[word for word in document.lower().split() if word not in STOPWORDS]

         for document in pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')['comment_text'].values.tolist()]



frequency = defaultdict(int)

for text in texts:

    for token in text:

        frequency[token] += 1



test_texts = [[token for token in text if frequency[token] > 1]for text in texts]
embeddings_index = {}

f = open('../input/glove840b300dtxt/glove.840B.300d.txt')

for line in f:

    values = line.split(' ')

    word = values[0] ## The first entry is the word

    coefs = np.asarray(values[1:], dtype='float32') ## These are the vecotrs representing the embedding for the word

    embeddings_index[word] = coefs

f.close()



print('GloVe data loaded')




MAX_NUM_WORDS = 1000

MAX_SEQUENCE_LENGTH = 100

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

train_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = np.asarray(documents['target'])

print(train_data.shape)

print(labels.shape)
## More code adapted from the keras reference (https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py)

# prepare embedding matrix 



## EMBEDDING_DIM =  ## seems to need to match the embeddings_index dimension

EMBEDDING_DIM = embeddings_index.get('a').shape[0]

num_words = min(MAX_NUM_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():

    if i > MAX_NUM_WORDS:

        continue

    embedding_vector = embeddings_index.get(word) ## This references the loaded embeddings dictionary

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector



# load pre-trained word embeddings into an Embedding layer

# note that we set trainable = False so as to keep the embeddings fixed

embedding_layer = Embedding(num_words,

                            EMBEDDING_DIM,

                            embeddings_initializer=Constant(embedding_matrix),

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=False)


## To create and visualize a model



model = Sequential()

model.add(Embedding(num_words, 300, input_length=100, weights= [embedding_matrix], trainable=False))



model.add(Dropout(0.2))

model.add(Conv1D(64, 5, activation='relu'))

model.add(MaxPooling1D(pool_size=4))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
## Fit train data

model.fit(train_data,labels, validation_split=0.3, epochs = 3)
## Model visualization code adapted from: https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa



## Get weights

embds = model.layers[0].get_weights()[0]

## Plotting function

## Visualize words in two dimensions 

tsne_embds = TSNE(n_components=2).fit_transform(embds)



plt.plot(tsne_embds[:,0],tsne_embds[:,1],'.')
test_data = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=MAX_SEQUENCE_LENGTH)

pred = model.predict(test_data)
model.summary()

test_info = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')



new_df = {'id':test_info['id'], 'prediction' : pred}
def extract_pred(pred):

    ans = []

    for i in pred:

        ans.append(i[0])

    return ans
df = pd.DataFrame({"id": test_info["id"], "prediction": extract_pred(pred)})

df.to_csv("submission.csv", index=False)