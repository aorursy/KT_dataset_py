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
import pandas as pd

data = pd.read_csv('../input/usinlppracticum/imdb_train.csv',delimiter = ",",encoding="latin-1")

data.head() 
data_test = pd.read_csv("../input/usinlppracticum/imdb_test.csv", delimiter=",",header=0,encoding="latin-1")

data_test.head()
data_mas = pd.read_csv('../input/imdb-review-dataset/imdb_master.csv',encoding="latin-1")

data_mas.head()
data_mas = data_mas.drop(['Unnamed: 0','type','file'],axis=1)

data_mas.columns = ["review","sentiment"]

data_mas.head()
data_mas = data_mas[data_mas.sentiment != 'unsup']

data_mas['sentiment'] = data_mas['sentiment'].map({'pos': 1, 'neg': 0})

data_mas.head()
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0}) 

data.head()
data = pd.concat([data, data_mas]).reset_index(drop=True)

data.head()
import string

alphabet = string.ascii_letters+string.punctuation

data.review.str.strip(alphabet).astype(bool).any()
data.review = data.review.str.replace('<br />', ' ')

data_test.review = data_test.review.str.replace('<br />', ' ')

data.head(17)
data.review = data.review.str.replace(r"[^a-zA-Z\s]+", "") 

data.head()



data_test.review = data_test.review.str.replace(r"[^a-zA-Z\s]+", "") 

data_test.head()
data['review'] = data['review'].str.lower()

data.head()



data_test['review'] = data_test['review'].str.lower()

data_test.head()
import re

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords



stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()



def clean_text(text): 

    text = re.sub(r'[^\w\s]','',text, re.UNICODE)

    text = text.lower()

    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]

    text = [lemmatizer.lemmatize(token, "v") for token in text]

    text = [word for word in text if not word in stop_words]

    text = " ".join(text)

    return text



data['c_review'] = data.review.apply(lambda x: clean_text(x))

data.head()

import keras.preprocessing

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Embedding, Dropout, Activation, GRU,Flatten

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model, Sequential

from keras.layers import Convolution1D

from keras import initializers, regularizers, constraints, optimizers, layers
max_features = 8800

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(data['c_review'])

list_tokenized_train = tokenizer.texts_to_sequences(data['c_review'])



maxlen = 130

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

y = data['sentiment']
embed_size = 128

model = Sequential()

model.add(Embedding(max_features, embed_size))

model.add(Bidirectional(LSTM(32, return_sequences = True)))

model.add(Dropout(0.3))

model.add(GlobalMaxPool1D())

model.add(Dense(20,activation = "relu")) 

model.add(Dropout(0.3))

model.add(Dense(1, activation = "sigmoid"))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



model.fit(X_t, y,batch_size=500,epochs = 10, validation_split=0.1 )
data_test.head(20)
data_test['review']=data_test.review.apply(lambda x: clean_text(x)) 

data_test.head()
list_sentences_test = data_test["review"]

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

prediction = model.predict(X_te)

y_pred = (prediction > 0.5)



pred = pd.DataFrame(y_pred.flatten())

data_pred= pd.merge(data_test, pred, left_index=True, right_index=True)
data_pred.columns = ['id','review','sentiment']

data_pred['sentiment'] = data_pred['sentiment'].map({True: 'positive', False: 'negative'})

data_pred_s = data_pred[['id','sentiment']]

data_pred_s.head()
data_pred_s.to_csv (r'submissions_v4.csv', index = None, header=True)