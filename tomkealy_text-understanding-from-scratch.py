import pandas as pd

import numpy as np

import os

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer

from sklearn.cross_validation import train_test_split

from nltk import sent_tokenize, word_tokenize

import string

from keras.models import Model

from keras.optimizers import SGD

from keras.layers import Input, Dense, Dropout, Flatten

from keras.layers.convolutional import Convolution1D, MaxPooling1D

import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

%matplotlib inline

print(os.getcwd())
texts = pd.read_csv( '../input/train.csv')
texts.head()
authors = texts['author']

texts = texts['text']
def create_vocab_set():

    #https://github.com/johnb30/py_crepe/

    #This alphabet is 69 chars vs. 70 reported in the paper since they include two

    # '-' characters. See https://github.com/zhangxiangxiao/Crepe#issues.



    alphabet = (list(string.ascii_lowercase) + list(string.digits) +

                list(string.punctuation) + ['\n'])

    vocab_size = len(alphabet)

    check = set(alphabet)



    vocab = {}

    reverse_vocab = {}

    for ix, t in enumerate(alphabet):

        vocab[t] = ix

        reverse_vocab[ix] = t



    return vocab, reverse_vocab, vocab_size, check
def encode_data(x, maxlen, vocab, vocab_size, check):

    #https://github.com/johnb30/py_crepe/

    #Iterate over the loaded data and create a matrix of size maxlen x vocabsize

    #In this case that will be 1014x69. This is then placed in a 3D matrix of size

    #data_samples x maxlen x vocab_size. Each character is encoded into a one-hot

    #array. Chars not in the vocab are encoded into an all zero vector.



    input_data = np.zeros((len(x), maxlen, vocab_size))

    for dix, sent in enumerate(x):

        counter = 0

        sent_array = np.zeros((maxlen, vocab_size))

        chars = list(sent.lower().replace(' ', ''))

        for c in chars:

            if counter >= maxlen:

                pass

            else:

                char_array = np.zeros(vocab_size, dtype=np.int)

                if c in check:

                    ix = vocab[c]

                    char_array[ix] = 1

                sent_array[counter, :] = char_array

                counter += 1

        input_data[dix, :, :] = sent_array



    return input_data
maxlen = 140

vocab, reverse_vocab, vocab_size, check = create_vocab_set()

encoded = encode_data(texts, maxlen, vocab, vocab_size, check)
lb = LabelBinarizer()

lb.fit(authors)

targets = lb.transform(authors)
encoded.shape
targets.shape
print(encoded[0])

print(targets[0])
X_train, X_test, y_train, y_test = train_test_split(encoded, targets, test_size=0.2, random_state=1234) 
nb_filter = 256

dense_outputs = 1024

cat_output = 3

batch_size = 80

nb_epoch = 10
inputs = Input(shape=(maxlen, vocab_size), name='input', dtype='float32')

conv0 = Convolution1D(nb_filter=nb_filter, filter_length=7, border_mode='valid', activation='relu', input_shape=(maxlen, vocab_size))(inputs)

conv0 = MaxPooling1D(pool_length=2)(conv0)



conv1 = Convolution1D(nb_filter=nb_filter, filter_length=7, border_mode='valid', activation='relu', input_shape=(maxlen, vocab_size))(conv0)

conv1 = MaxPooling1D(pool_length=2)(conv1)



conv2 = Convolution1D(nb_filter=nb_filter, filter_length=4, border_mode='valid', activation='relu', input_shape=(maxlen, vocab_size))(conv1)



conv3 = Convolution1D(nb_filter=nb_filter, filter_length=4, border_mode='valid', activation='relu', input_shape=(maxlen, vocab_size))(conv2)



conv4 = Convolution1D(nb_filter=nb_filter, filter_length=4, border_mode='valid', activation='relu', input_shape=(maxlen, vocab_size))(conv3)



conv5 = Convolution1D(nb_filter=nb_filter, filter_length=4, border_mode='valid', activation='relu', input_shape=(maxlen, vocab_size))(conv4)



conv6 = Convolution1D(nb_filter=nb_filter, filter_length=4, border_mode='valid', activation='relu', input_shape=(maxlen, vocab_size))(conv5)

conv6 = MaxPooling1D(pool_length=2)(conv6)

conv6 = Flatten()(conv5)



dense0 = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv6))

dense1 = Dropout(0.5)(Dense(dense_outputs, activation='relu')(dense0))



pred = Dense(cat_output, activation='softmax', name='output')(dense1)



model = Model(input=inputs, output=pred)



sgd = SGD(lr=0.01, momentum=0.9)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x=X_train, y=y_train, epochs=5)
y_pred = model.predict(X_test)
y_pred
model.evaluate(X_test, y_test)
a2c = {'EAP': 0, 'HPL' : 1, 'MWS' : 2}

result = pd.read_csv('../input/sample_submission.csv')

for a, i in a2c.items():

    result[a] = y_pred[:, i]

#to_submit=result