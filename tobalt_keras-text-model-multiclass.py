# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# keras model

# for text classification, with multiple classes (multi-class)

# but single label

# character-level tokenization

# with fixed length input

# based on convolutional layers



# using customer complaints dataset

# we classify a narrative text about an issue into a product category

# https://www.kaggle.com/cfpb/us-consumer-finance-complaints



# See also

# https://www.kaggle.com/kadhambari/multi-class-text-classification

# https://www.kaggle.com/anucool007/multi-class-text-classification-bag-of-words
import keras
# utility functions for later



def dict_to_csv(d, path):

    df = pd.DataFrame.from_dict(d, orient='index')

    df.to_csv(path)
# load dataset

df = pd.read_csv('../input/consumer_complaints.csv', usecols=('product', 'consumer_complaint_narrative'))
print(len(df))

df.head()[:5]
# remove nan's

df = df.dropna() # drop row if have nan in any column

print(len(df))

df.head()[:5]
# encode product
# this turns each string into a number (most popular are lowest)

product_encoding = pd.factorize(df['product'])

print(product_encoding)
labels, index = product_encoding

print(labels) # encoding for each product in the dataset

print(index) # index -> string map
# build label <-> index maps to use later

product_to_id = {name: i for i, name in enumerate(index)}

id_to_product = {i: name for i, name in enumerate(index)}

print(product_to_id)

print(id_to_product)

print(len(index)) # number of classes
dict_to_csv(product_to_id, 'labels_index.csv')
# note that the classes are imbalanced

for product in index:

    print(product, len(df.loc[df['product'] == product]))
# one-hot encode

y = keras.utils.to_categorical(labels)

print(len(y))

print(y[0], labels[0])
# encode input with character level tokenization and embeddings



from keras.preprocessing.text import Tokenizer



tok = Tokenizer(num_words=None, # don't limit number of characters

                lower=False, # don't lower

                char_level=True, # character-level tokenization

                oov_token='<OOV>', # token for unknown characters

                                   # FIXME: multi-character token but should be single char?

               )
texts = df['consumer_complaint_narrative'].values

print(texts[:2])
tok.fit_on_texts(texts)
x = tok.texts_to_sequences(texts)

print(x[:2])
df = pd.DataFrame(x)

df.to_csv('x_data.csv')
y = np.array(y)

np.savetxt('y_data.csv', y, fmt="%d", delimiter=",")
# create word index to use later



# word -> index map

word_index = tok.word_index

word_index['<PAD>'] = 0 # set unused index to padding token

print(word_index['<PAD>'], word_index['<OOV>'], word_index[' '])



# index -> word map

reversed_word_index = {v:k for k, v in word_index.items()}

print(reversed_word_index[0], reversed_word_index[1], reversed_word_index[2])
dict_to_csv(word_index, 'word_index.csv')
def vectorized_to_tokens(sample):

    return [reversed_word_index.get(num, '<OOV>') for num in sample]

    

def tokens_to_string(tokens):

    return ''.join(tokens)
print(len(x), len(y))
# pad to fixed length
# find a good length to pad to

lengths = [len(sample) for sample in x]

print(len(lengths))

print(lengths[0])
p = np.percentile(lengths, 95)

print(p)
maxlen = int(p)

print(maxlen)
from keras.preprocessing.sequence import pad_sequences



x = pad_sequences(x,

                  padding='post',

                  truncating='post',

                  value=word_index['<PAD>'],

                  maxlen=maxlen,

                )

print(len(x))

print(len(x[0]))
# split train and test data



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)



print(len(x_train), len(x_test))

print(len(y_train), len(y_test))
print(tokens_to_string(vectorized_to_tokens(x_test[0])))

print(id_to_product[np.argmax(y_test[0])])
# define model



from keras.models import Sequential

from keras.layers import (

    Embedding,

    Conv1D,

    Dense,

    MaxPooling1D,

    AveragePooling1D,

    Flatten,

    GlobalAveragePooling1D,

    Dropout,

)



model = Sequential([

    Embedding(len(word_index), 8, input_length=maxlen),

    Conv1D(128, 15, activation='relu'),

    Dropout(0.2),

    MaxPooling1D(2),

    Conv1D(128, 10, activation='relu'),

    Dropout(0.2),

    AveragePooling1D(2),

    Conv1D(128, 5, activation='relu'),

    Dropout(0.2),

    MaxPooling1D(2),

    GlobalAveragePooling1D(),

    Dense(32, activation='relu'),

    Dense(11, activation='softmax'),

])



model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# train



epochs = 47

batch_size = 512



history = model.fit(x, 

                    y, 

                    epochs=epochs, 

                    batch_size=batch_size,

                    verbose=2, 

                    validation_split=0.3,

                   )



# baseline: 1/11 ~= 0.1 accuracy with random guessing
# test

print(model.evaluate(x_test, y_test))
model.save('keras_text_model_multiclass.h5')