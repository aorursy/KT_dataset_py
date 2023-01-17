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
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



train_df.head(5)
train_df.isnull().sum()
test_df.isnull().sum()
train_df=train_df.fillna(" ")

train_df.isnull().sum()
test_df = test_df.fillna(" ")

test_df.isnull().sum()
train_df["text"]= train_df["keyword"] + " " + train_df["location"] + " "+train_df["text"]

test_df["text"]= test_df["keyword"] + " " + test_df["location"] + " "+test_df["text"]



train_df=train_df.drop("keyword",axis=1)

train_df=train_df.drop("location",axis=1)



test_df=test_df.drop("keyword",axis=1)

test_df=test_df.drop("location",axis=1)
print(train_df["text"][5])
import re

def text_normalize(sen):

    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence.lower()



X = []

for sen in list(train_df["text"]):

    X.append(text_normalize(sen))

train_df["text"] = X



X = []

for sen in list(test_df["text"]):

    X.append(text_normalize(sen))

test_df["text"] = X



train_df.head()
print(train_df["text"][5])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df["text"],train_df["target"], test_size=0.2, random_state=42)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC,SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



text_clf = Pipeline([('tfidf', TfidfVectorizer()),('clf',LinearSVC(loss="hinge",fit_intercept=False))])

text_clf.fit(X_train, y_train) 

predictions = text_clf.predict(X_test)



print("Accuracy score: ", accuracy_score(y_test, predictions))

print("Precision score: ", precision_score(y_test, predictions))

print("Recall score: ", recall_score(y_test, predictions))
y_pred = text_clf.predict(test_df["text"])



sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')



y_pred = np.round(y_pred).astype(int).reshape(3263)

sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':y_pred})

sub.to_csv('submission.csv',index=False)
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_df=train_df.fillna(" ")

train_df.isnull().sum()



test_df = test_df.fillna(" ")

test_df.isnull().sum()



train_df["text"]= train_df["keyword"] + " " + train_df["location"] + " "+train_df["text"]

test_df["text"]= test_df["keyword"] + " " + test_df["location"] + " "+test_df["text"]



train_df=train_df.drop("keyword",axis=1)

train_df=train_df.drop("location",axis=1)



test_df=test_df.drop("keyword",axis=1)

test_df=test_df.drop("location",axis=1)
import re

# from nltk.corpus import stopwords

# stop_words = set(stopwords.words('english'))

def preprocess_text(sen):

    sentence = re.sub("http[s]*://[^\s]+"," ",sen)

    # Remove punctuations and numbers

    sentence = re.sub('[^a-zA-Z]', ' ', sentence)



    # Single character removal

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    

    # Removing multiple spaces

    sentence = re.sub(r'\s+', ' ', sentence)

    

    return sentence
X = []

sentences = list(train_df["text"])

for sen in sentences:

    X.append(preprocess_text(sen))

y = train_df["target"].values

y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



from keras.preprocessing.text import Tokenizer

from keras_preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(X_train)



X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)



vocab_size = len(tokenizer.word_index) + 1



maxlen = 160



X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove*.zip
from numpy import array

from numpy import asarray

from numpy import zeros



embeddings_dictionary = dict()



glove_file = open('./glove.6B.200d.txt', encoding="utf8")



for line in glove_file:

    records = line.split()

    word = records[0]

    vector_dimensions = asarray(records[1:], dtype='float32')

    embeddings_dictionary[word] = vector_dimensions

glove_file.close()



embedding_matrix = zeros((vocab_size, 200))

for word, index in tokenizer.word_index.items():

    embedding_vector = embeddings_dictionary.get(word)

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector
from keras.layers import Embedding,Dense,GlobalMaxPool1D,Dropout,Flatten,Bidirectional,LSTM

from keras.models import Sequential

model=Sequential([Embedding(vocab_size,200,input_length=maxlen,weights=[embedding_matrix], trainable=False),

                 Bidirectional(LSTM(100,return_sequences=True)),

                 GlobalMaxPool1D(),

                  Dense(64,activation = 'relu'),

                  Dense(1,activation='sigmoid')

                 ])





model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.summary()
# history = model.fit(X_train, y_train, batch_size=256, epochs=15, verbose=1, validation_split=0.2)