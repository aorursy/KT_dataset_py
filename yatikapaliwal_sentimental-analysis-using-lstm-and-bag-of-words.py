import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

import os

import io

import string

import re

from sklearn.metrics import classification_report, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier

from sklearn.svm import SVC
reviews = pd.read_csv('../input/zomato-restaurants-hyderabad/Restaurant reviews.csv')

reviews.head()
reviews.isnull().sum()
reviews.shape
reviews.info()
reviews =reviews.dropna()
reviews.shape
reviews_txt = reviews[['Review', 'Rating']]

reviews_txt.head()
reviews_txt['Rating'] = reviews_txt['Rating'].replace('Like', 5)

reviews_txt['Rating'] = reviews_txt['Rating'].astype('float')
reviews_txt['Rating'] = np.where(reviews_txt['Rating']<4, 0, 1) #0 for BAD rating and 1 for Good rating
reviews_txt.head()
reviews_txt['Rating'].value_counts()
reviews['Restaurant'].value_counts()
reviews.columns
sns.countplot(reviews_txt['Rating']) #0 for bad and 1 for good
cleanup_re = re.compile('[^a-z]+')

def clean(sentence): 

  sentence = str(sentence)

  sentence = sentence.lower()

  sentence = cleanup_re.sub(' ', sentence).strip()

  return sentence

reviews_txt['Review'] = reviews_txt['Review'].apply(clean)
nltk.download('popular')
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
def preprocess(sentence):

  sentence = str(sentence)

  word_tokens = word_tokenize(sentence)

  stop_words = set(stopwords.words('english'))

  sentence = ' '.join([i for i in word_tokens if not i in stop_words])

  return sentence



reviews_txt['Review'] = reviews_txt['Review'].apply(preprocess)
reviews_txt.head()
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

def preprocess4(sentence):

  input_str=word_tokenize(sentence)

  lemmatized_output = ' '.join([lemma.lemmatize(w) for w in input_str])

  return lemmatized_output



reviews_txt['Review'] = reviews_txt['Review'].apply(preprocess4)
reviews_txt.head()
X = reviews_txt['Review']

y = reviews_txt['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)
X_train.shape, X_test.shape
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(X_train)



#transform the train and test dataset

X_train_countvect = count_vect.transform(X_train)

X_test_countvect = count_vect.transform(X_test)
from sklearn.feature_extraction.text import TfidfVectorizer

# word level tf-idf

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(X_train)

xtrain_tfidf =  tfidf_vect.transform(X_train)

xtest_tfidf =  tfidf_vect.transform(X_test)

 

 # ngram level tf-idf 

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram.fit(X_train)

xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)

xtest_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)



# characters level tf-idf

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram_chars.fit(X_train)

xtrain_tfidf_ngram_char =  tfidf_vect_ngram_chars.transform(X_train)

xtest_tfidf_ngram_char =  tfidf_vect_ngram_chars.transform(X_test)
def train_model(classifier, feature_vector_train, label, feature_vector_valid):

    # fit the training dataset on the classifier

    classifier.fit(feature_vector_train, label)

    

    # predict the labels on validation dataset

    predictions = classifier.predict(feature_vector_valid)

    

    return accuracy_score(predictions, y_test)
# Naive Bayes on Count Vectors

accuracy = train_model(MultinomialNB(), X_train_countvect, y_train, X_test_countvect)

print("NB, Count Vectors: ", accuracy)



# Naive Bayes on Word Level TF IDF Vectors

accuracy = train_model(MultinomialNB(), xtrain_tfidf, y_train, xtest_tfidf)

print("NB, WordLevel TF-IDF: ", accuracy)



# Naive Bayes on Ngram Level TF IDF Vectors

accuracy = train_model(MultinomialNB(), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram)

print("NB, N-Gram Vectors: ", accuracy)



# Naive Bayes on Character Level TF IDF Vectors

accuracy = train_model(MultinomialNB(), xtrain_tfidf_ngram_char, y_train, xtest_tfidf_ngram_char)

print("NB, CharLevel Vectors: ", accuracy)
# Logistic on Count Vectors

accuracy = train_model(LogisticRegression(), X_train_countvect, y_train, X_test_countvect)

print("Count Vectors: ", accuracy)



# Logistic on Word Level TF IDF Vectors

accuracy = train_model(LogisticRegression(), xtrain_tfidf, y_train, xtest_tfidf)

print("WordLevel TF-IDF: ", accuracy)



# Logistic on Ngram Level TF IDF Vectors

accuracy = train_model(LogisticRegression(), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram)

print("N-Gram Vectors: ", accuracy)



# Logistic on Character Level TF IDF Vectors

accuracy = train_model(LogisticRegression(), xtrain_tfidf_ngram_char, y_train, xtest_tfidf_ngram_char)

print("CharLevel Vectors: ", accuracy)
# SVM on Ngram Level TF IDF Vectors

accuracy = train_model(SVC(), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram)

print("NB, N-Gram Vectors: ", accuracy)
accuracy = train_model(RandomForestClassifier(n_estimators=250, random_state=100), X_train_countvect, y_train, X_test_countvect)

print("Count Vectors: ", accuracy)



# Random Forest on Word Level TF IDF Vectors

accuracy = train_model(RandomForestClassifier(), xtrain_tfidf, y_train, xtest_tfidf)

print("WordLevel TF-IDF: ", accuracy)
accuracy = train_model(BaggingClassifier(), X_train_countvect, y_train, X_test_countvect)

print("Count Vectors: ", accuracy)



# Bagging on Word Level TF IDF Vectors

accuracy = train_model(BaggingClassifier(), xtrain_tfidf, y_train, xtest_tfidf)

print("WordLevel TF-IDF: ", accuracy)
import xgboost as xgb

accuracy = train_model(xgb.XGBClassifier(), X_train_countvect, y_train, X_test_countvect)

print("Count Vectors: ", accuracy)



# Boosting on Word Level TF IDF Vectors

accuracy = train_model(xgb.XGBClassifier(), xtrain_tfidf, y_train, xtest_tfidf)

print("WordLevel TF-IDF: ", accuracy)
import tensorflow as tf

from tensorflow import keras

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Embedding

from keras.layers import SpatialDropout1D

from keras.layers import LSTM

from keras.callbacks import EarlyStopping

from keras.layers import Bidirectional

from keras.layers import Dropout
#max words to be used

MAX_WORDS = 10000

#max length of the sequence

MAX_LEN = 50

#embedding dimension should be between 50 to 300

EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(X)

word_index = tokenizer.word_index

print('number of unique tokens are: ', len(word_index))
from keras.preprocessing.sequence import pad_sequences
X = tokenizer.texts_to_sequences(X)

X = pad_sequences(X, maxlen=MAX_LEN)

print('shape of data tensor is', X.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 100)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
model = Sequential()

model.add(Embedding(input_dim=MAX_WORDS, output_dim= EMBEDDING_DIM, input_length=MAX_LEN))

model.add(LSTM(300, recurrent_dropout=0.1))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history =model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=64)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)



plt.title('Training and validation accuracy')

plt.plot(epochs, acc, 'red', label='Training acc')

plt.plot(epochs, val_acc, 'blue', label='Validation acc')

plt.legend()



plt.figure()

plt.title('Training and validation loss')

plt.plot(epochs, loss, 'red', label='Training loss')

plt.plot(epochs, val_loss, 'blue', label='Validation loss')



plt.legend()



plt.show()