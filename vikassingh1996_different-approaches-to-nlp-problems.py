"""Importing libraries"""

import numpy as np

import pandas as pd

from tqdm import tqdm

import time

import re

import string

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

import xgboost as xgb

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.metrics import f1_score



from keras.models import Sequential

from keras.layers.recurrent import LSTM, GRU

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

from keras.preprocessing import sequence, text

from keras.callbacks import EarlyStopping
"""Let's load the data files"""

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
"""Reading train data"""

print(train.shape)

train.head()
"""Reading test data"""

print(test.shape)

test.head()
"""reading submission file"""

sub.head()
xtrain, xvalid, ytrain, yvalid = train_test_split(train.text, 

                                                  train.target,

                                                  random_state=42, 

                                                  test_size=0.1, shuffle=True)
print(xtrain.shape)

print(xvalid.shape)
%%time

def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text





# Applying the cleaning function to both test and training datasets

xtrain = xtrain.apply(lambda x: clean_text(x))

xvalid = xvalid.apply(lambda x: clean_text(x))

xtrain.head(3)
%%time

tokenizer1 = nltk.tokenize.WhitespaceTokenizer()

tokenizer2 = nltk.tokenize.TreebankWordTokenizer()

tokenizer3 = nltk.tokenize.WordPunctTokenizer()

tokenizer4 = nltk.tokenize.RegexpTokenizer(r'\w+')

tokenizer5 = nltk.tokenize.TweetTokenizer()



# appling tokenizer5

xtrain = xtrain.apply(lambda x: tokenizer5.tokenize(x))

xvalid = xvalid.apply(lambda x: tokenizer5.tokenize(x))

xtrain.head(3)
%%time

def remove_stopwords(text):

    """

    Removing stopwords belonging to english language

    

    """

    words = [w for w in text if w not in stopwords.words('english')]

    return words





xtrain = xtrain.apply(lambda x : remove_stopwords(x))

xvalid = xvalid.apply(lambda x : remove_stopwords(x))
%%time

def combine_text(list_of_text):

    combined_text = ' '.join(list_of_text)

    return combined_text



xtrain = xtrain.apply(lambda x : combine_text(x))

xvalid = xvalid.apply(lambda x : combine_text(x))
# Stemmer

stemmer = nltk.stem.PorterStemmer()



# Lemmatizer

lemmatizer=nltk.stem.WordNetLemmatizer()



# Appling Lemmatizer

xtrain = xtrain.apply(lambda x: lemmatizer.lemmatize(x))

xvalid = xvalid.apply(lambda x: lemmatizer.lemmatize(x))
# Appling CountVectorizer()

count_vectorizer = CountVectorizer()

xtrain_vectors = count_vectorizer.fit_transform(xtrain)

xvalid_vectors = count_vectorizer.transform(xvalid)
# Appling TFIDF

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2), norm='l2')

xtrain_tfidf = tfidf.fit_transform(xtrain)

xvalid_tfidf = tfidf.transform(xvalid)
# Fitting a simple Logistic Regression on TFIDF

clf = LogisticRegression(C=1.0)

clf.fit(xtrain_tfidf, ytrain)

#scores = model_selection.cross_val_score(clf, train_tfidf, ytrain, cv=5, scoring="f1")



predictions = clf.predict(xvalid_tfidf)

print('simple Logistic Regression on TFIDF')

print ("f1_score :", np.round(f1_score(yvalid, predictions),5))
# Fitting a simple Logistic Regression on CountVec

clf = LogisticRegression(C=1.0)

clf.fit(xtrain_vectors, ytrain)

#scores = model_selection.cross_val_score(clf, xtrain_vectors, ytrain_vectors, cv=5, scoring="f1")



predictions = clf.predict(xvalid_vectors)

print('simple Logistic Regression on CountVectorizer')

print ("f1_score :", np.round(f1_score(yvalid, predictions),5))
# Fitting a LinearSVC on TFIDF

clf = SVC()

clf.fit(xtrain_tfidf, ytrain)



predictions = clf.predict(xvalid_tfidf)

print('SVC on TFIDF')

print ("f1_score :", np.round(f1_score(yvalid, predictions),5))
# Fitting a LinearSVC on CountVec

clf = SVC()

clf.fit(xtrain_vectors, ytrain)



predictions = clf.predict(xvalid_vectors)

print('SVC on CountVectorizer')

print ("f1_score :", np.round(f1_score(yvalid, predictions),5))
# Fitting a MultinomialNB on TFIDF

clf = MultinomialNB()

clf.fit(xtrain_tfidf, ytrain)



predictions = clf.predict(xvalid_tfidf)

print('MultinomialNB on TFIDF')

print ("f1_score :", np.round(f1_score(yvalid, predictions),5))
# Fitting a MultinomialNB on CountVec

clf = MultinomialNB()

clf.fit(xtrain_vectors, ytrain)



predictions = clf.predict(xvalid_vectors)

print('MultinomialNB on CountVectorizer')

print ("f1_score :", np.round(f1_score(yvalid, predictions),5))
# Fitting a simple xgboost on TFIDF

clf = xgb.XGBClassifier(max_depth=5, n_estimators=300, colsample_bytree=0.8, 

                        subsample=0.5, nthread=10, learning_rate=0.1)

clf.fit(xtrain_tfidf.tocsc(), ytrain)

predictions = clf.predict(xvalid_tfidf.tocsc())



print('XGBClassifier on TFIDF')

print ("f1_score :", np.round(f1_score(yvalid, predictions),5))
# Fitting a simple xgboost on CountVec

clf = xgb.XGBClassifier(max_depth=5, n_estimators=300, colsample_bytree=0.8, 

                        subsample=0.5, nthread=10, learning_rate=0.1)

clf.fit(xtrain_vectors, ytrain)



predictions = clf.predict(xvalid_vectors)

print('XGBClassifier on CountVectorizer')

print ("f1_score :", np.round(f1_score(yvalid, predictions),5))
'''Create a function to tune hyperparameters of the selected models.'''

seed = 44

def grid_search_cv(model, params):

    global best_params, best_score

    grid_search = GridSearchCV(estimator = model, param_grid = params, cv = 5, 

                             verbose = 3,

                             scoring = 'f1', n_jobs = -1)

    grid_search.fit(xtrain_vectors, ytrain)

    best_params = grid_search.best_params_

    best_score = grid_search.best_score_

    

    return best_params, best_score
'''Define hyperparameters of Logistic Regression.'''

LR_model = LogisticRegression()



LR_params = {'penalty':['l1', 'l2'],

             'C': np.logspace(0.1, 1, 4, 8 ,10)}



grid_search_cv(LR_model, LR_params)

LR_best_params, LR_best_score = best_params, best_score

print('LR best params:{} & best_score:{:0.5f}' .format(LR_best_params, LR_best_score))
'''Define hyperparameters of Logistic Regression.'''

SVC_model = SVC()



SVC_params = {'kernel':[ 'linear', 'rbf', 'sigmoid'],

             'C': np.logspace(0.1, 1,10)}



grid_search_cv(SVC_model, SVC_params)

SVC_best_params, SVC_best_score = best_params, best_score

print('SVC best params:{} & best_score:{:0.5f}' .format(SVC_best_params, SVC_best_score))
'''Define hyperparameters of Logistic Regression.'''

NB_model = MultinomialNB()



NB_params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}



grid_search_cv(NB_model, NB_params)

NB_best_params, NB_best_score = best_params, best_score

print('NB best params:{} & best_score:{:0.5f}' .format(NB_best_params, NB_best_score))
#'''For XGBC, the following hyperparameters are usually tunned.'''

#'''https://xgboost.readthedocs.io/en/latest/parameter.html'''



#XGB_model = XGBClassifier(

#            n_estimators=500,

#            verbose = True)





#XGB_params = {'max_depth': (2, 5),

#               'reg_alpha':  (0.01, 0.4),

#               'reg_lambda': (0.01, 0.4),

#               'learning_rate': (0.1, 0.4),

#               'colsample_bytree': (0.3, 1),

#               'gamma': (0.01, 0.7),

#               'num_leaves': (2, 5),

#               'min_child_samples': (1, 5),

#              'subsample': [0.5, 0.8],

#              'random_state':[seed]}



#grid_search_cv(XGB_model, XGB_params)

#XGB_best_params, XGB_best_score = best_params, best_score

#print('XGB best params:{} & best_score:{:0.5f}' .format(XGB_best_params, XGB_best_score))
"""Load the Glove vectors in a dictionay"""

embeddings_index={}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embeddings_index[word]=vectors

f.close()



print('Found %s word vectors.' % len(embeddings_index))
""" Function Creates a normalized vector for the whole sentence"""

def sent2vec(s):

    words = str(s).lower()

    words = word_tokenize(words)

    words = [w for w in words if not w in stopwords.words('english')]

    words = [w for w in words if w.isalpha()]

    M = []

    for w in words:

        try:

            M.append(embeddings_index[w])

        except:

            continue

    M = np.array(M)

    v = M.sum(axis=0)

    if type(v) != np.ndarray:

        return np.zeros(200)

    return v / np.sqrt((v ** 2).sum())
# create sentence vectors using the above function for training and validation set

# create glove features

xtrain_glove = np.array([sent2vec(x) for x in tqdm(xtrain)])

xvalid_glove = np.array([sent2vec(x) for x in tqdm(xvalid)])
# Shape of data after embedding

xtrain_glove.shape,  xvalid_glove.shape
# Fitting a simple xgboost on glove features

clf = xgb.XGBClassifier(max_depth=8, n_estimators=300, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

clf.fit(xtrain_glove, ytrain)



predictions = clf.predict(xvalid_glove)

print('XGBClassifier on GloVe featur')

print ("f1_score :", np.round(f1_score(yvalid, predictions),5))
"""scale the data before any neural net"""

scl = preprocessing.StandardScaler()

xtrain_glove_scl = scl.fit_transform(xtrain_glove)

xvalid_glove_scl = scl.transform(xvalid_glove)
"""create a simple 2 layer sequential neural net"""

model = Sequential()



model.add(Dense(200, input_dim=200, activation='relu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())



model.add(Dense(200, activation='relu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())



model.add(Dense(1))

model.add(Activation('sigmoid'))



# compile the model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xtrain_glove_scl, y=ytrain, batch_size=64, 

          epochs=10, verbose=1, 

          validation_data=(xvalid_glove_scl, yvalid))
predictions = model.predict(xvalid_glove_scl)

predictions = np.round(predictions).astype(int)

print('2 layer sequential neural net on GloVe Feature')

print ("f1_score :", np.round(f1_score(yvalid, predictions),5))
# using keras tokenizer here

token = text.Tokenizer(num_words=None)

max_len = 80



token.fit_on_texts(list(xtrain) + list(xvalid))

xtrain_seq = token.texts_to_sequences(xtrain)

xvalid_seq = token.texts_to_sequences(xvalid)



# zero pad the sequences

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)

xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)



word_index = token.word_index

print('Number of unique words:',len(word_index))
#create an embedding matrix for the words we have in the dataset

embedding_matrix = np.zeros((len(word_index) + 1, 200))

for word, i in tqdm(word_index.items()):

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
# A simple LSTM with glove embeddings and two dense layers

model = Sequential()

model.add(Embedding(len(word_index) + 1,

                     200,

                     weights=[embedding_matrix],

                     input_length=max_len,

                     trainable=False))

model.add(SpatialDropout1D(0.3))

model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))



model.add(Dense(1000, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(1000, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model with early stopping callback

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')



model.fit(xtrain_pad, y=ytrain, batch_size=512, epochs=100, verbose=1, validation_data=(xvalid_pad, yvalid), callbacks=[earlystop])
predictions = model.predict(xvalid_pad)

predictions = np.round(predictions).astype(int)



print('simple LSTM')

print ("f1_score :", np.round(f1_score(yvalid, predictions),5))
# A simple LSTM with glove embeddings and two dense layers

model = Sequential()

model.add(Embedding(len(word_index) + 1,

                     200,

                     weights=[embedding_matrix],

                     input_length=max_len,

                     trainable=False))

model.add(SpatialDropout1D(0.3))

model.add(GRU(100, dropout=0.3, recurrent_dropout=0.3))



model.add(Dense(1000, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(1000, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model with early stopping callback

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')



model.fit(xtrain_pad, y=ytrain, batch_size=512, epochs=100, verbose=1, validation_data=(xvalid_pad, yvalid), callbacks=[earlystop])
predictions = model.predict(xvalid_pad)

predictions = np.round(predictions).astype(int)

print('simple GRU')

print ("f1_score :", np.round(f1_score(yvalid, predictions),5))