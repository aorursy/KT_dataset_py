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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gc
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import Model
from keras.utils import plot_model

import nltk
nltk.download('punkt')
nltk.download('stopwords')
df=pd.read_csv('../input/dataset-csv/text_emotion.csv')
df.head()
df=df.drop(['tweet_id','author'],axis=1)
df.head()
df["sentiment"].value_counts()
def clean(df):
    line=df['content'].values.tolist()
    all_content=list()
    for text in line:
        text=text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)        
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text) 
        text = re.sub(r"\'ll", " will", text)  
        text = re.sub(r"\'ve", " have", text)  
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"don't", "do not", text)
        text = re.sub(r"did't", "did not", text)
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"couldn't", "could not", text)
        text = re.sub(r"have't", "have not", text)
        pattern=re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-f][0-9a-fA-F]))+')
        text=pattern.sub("",text)
        text=re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]","",text)
        tokens=word_tokenize(text)
        table=str.maketrans('','',string.punctuation)
        stripped=[w.translate(table) for w in tokens]
        words=[word for word in stripped if word.isalpha()]
        stop_word=set(stopwords.words("english"))
        stop_word.discard("not")
        ps=PorterStemmer()
        words=[ps.stem(w) for w in words if not w in stop_word]
        words=' '.join(words)
        all_content.append(words)
    return all_content
all_review=clean(df)
all_review[0:5]
dummies=pd.get_dummies(df.sentiment)
dummies.head()
y = dummies.values
def padding(all_review):   
    max_features = 10000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(all_review))
    list_tokenized_train = tokenizer.texts_to_sequences(all_review)
    totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
    plt.hist(totalNumWords,bins = np.arange(0,30,1))
    maxlen = 18
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    return X_t

X_t=padding(all_review)

maxlen = 18
inp = Input(shape=(maxlen, ))
embed_size = 100
max_features = 10000
x = Embedding(max_features, embed_size)(inp)
x = LSTM(30, return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(15, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(13, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
batch_size = 30
epochs = 10
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)