# https://www.kaggle.com/willianbecker/



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding, Flatten

from keras.layers import LSTM

from keras.layers import Conv1D, MaxPooling1D

from sklearn.model_selection import train_test_split

import re

import numpy as np 

import pandas as pd

from nltk.corpus import stopwords

from nltk import word_tokenize

STOPWORDS = set(stopwords.words('english'))
import pandas as pd

df = pd.read_csv("../input/consumer-complaints/consumer_complaints.csv")
print(df.info())
print(df["product"].value_counts())
# texto do usuario

df = df[df["consumer_complaint_narrative"].isnull() == False]
print(df["product"].value_counts())
df.head()
# realiza a limpeza nos dados (lowecase, remocao de caracteres e stopwords)

remove_caracteres = re.compile('[^0-9a-z #+_]')

replace_espaco = re.compile('[/(){}\[\]\|@,;]')

df = df.reset_index(drop=True)



def pre_processamento(text):

    text = text.lower()

    text = remove_caracteres.sub('', text)

    text = replace_espaco.sub(' ', text)

    text = text.replace('x', '')

    text = ' '.join(word for word in text.split() if word not in STOPWORDS)

    return text



df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].apply(pre_processamento)
n_max_palavras = 5000

tamanho_maximo_sent = 250

embedding_dimensions = 100



tokenizer = Tokenizer(num_words=n_max_palavras, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(df['consumer_complaint_narrative'].values)

word_index = tokenizer.word_index

print(' %s tokens unicos.' % len(word_index))
X = tokenizer.texts_to_sequences(df['consumer_complaint_narrative'].values)

X = pad_sequences(X, maxlen=tamanho_maximo_sent)

print("shape X", X.shape)
Y = pd.get_dummies(df["product"]).values

print("shape Y", Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)

print(X_train.shape)

print(X_test.shape)
# PARAMETROS PARA CONVOLUCAO



kernel_size = 5 

filters = 24

pool_size = 4



model = Sequential()

model.add(Embedding(n_max_palavras, embedding_dimensions, input_length=X.shape[1]))

model.add(Conv1D(filters, 

                 kernel_size,

                activation="relu",

                strides=1))

model.add(MaxPooling1D(pool_size=pool_size))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(11))

model.add(Activation("softmax"))



model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



print(model.summary())

model.fit(X_train, Y_train,

         batch_size=256,

         epochs=2,

         validation_data=(X_test, Y_test))



score, acc = model.evaluate(X_test, Y_test, batch_size=256)



print("Acuracia Teste", acc)