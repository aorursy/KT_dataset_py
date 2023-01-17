# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

import textblob, string

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble

from sklearn.model_selection import KFold

from sklearn.metrics import f1_score, roc_auc_score

from keras.preprocessing import text, sequence

from keras import layers, models, optimizers



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading datasets

true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
# ======= Data preprocessing =======

# concat two datasets into one

true['true'] = 1

fake['true'] = 0

df = pd.concat([true, fake])

df.head()
df.tail()
# KFold

X = np.array(df.iloc[:,1])

Y = np.array(df.iloc[:,4])



kf = KFold(n_splits=10, random_state=3, shuffle=True)

for train, test in kf.split(X):

    X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]

    

# label encode the target variable

encoder = preprocessing.LabelEncoder()

Y_train = encoder.fit_transform(Y_train)

Y_test = encoder.fit_transform(Y_test)
# ensure shape of dataset = [n,] instead of [n,1]

print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
# ===== Feature Engineering / Word Embeddings =====

# Need not prediction based. Since we do not need to predict any words.

# (Prediction based : FastText, CBOW, Skip-Gram)

# Frequency based 

# 1. Count vectors (this is not a good feature engieering method, but useful for later model building)

# create a count vectorizer object

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(df['text'])

# transform the training and test set 

xtrain_count = count_vect.transform(X_train)

xtest_count = count_vect.transform(X_test)
# 2, TF_IDF vectors

# main idea : reflect importance of words in doc set

# Word level : word level

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(df['text'])

xtrain_tfidf = tfidf_vect.transform(X_train)

xtest_tfidf = tfidf_vect.transform(X_test)

# n-gram level : n-words forming a phrase. phrase level

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram.fit(df['text'])

xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)

xtest_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)

# character level : character level

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram_chars.fit(df['text'])

xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_train) 

xtest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_test) 
# ===== Model Building =====

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):

    # fit the training dataset on the classifier

    classifier.fit(feature_vector_train, label)

    

    # predict the labels on validation dataset

    predictions = classifier.predict(feature_vector_valid)

    

    if is_neural_net:

        predictions = predictions.argmax(axis=-1)

    

    data = {'y_Actual' : Y_test,

            'y_Predicted' : predictions

           }

    #confusion matrix 

    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])

    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    #f1 score

    f1score = f1_score(Y_test, predictions, average='macro')

    #accuracy

    accuracy = metrics.accuracy_score(predictions, Y_test)

    #roc_auc

    roc_auc = roc_auc_score(Y_test, predictions)



    print(confusion_matrix)

    print("f1 score :",f1score)

    print("accuracy :", accuracy)

    print("roc auc:", roc_auc)

    print("\n")
# Naive Bayes on Count Vectors

print( "NB, Count Vectors: ")

train_model(naive_bayes.MultinomialNB(), xtrain_count, Y_train, xtest_count)



# Naive Bayes on Word Level TF IDF Vectors

print ("NB, WordLevel TF-IDF: ")

train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, Y_train, xtest_tfidf)



# Naive Bayes on Ngram Level TF IDF Vectors

print ("NB, N-Gram Vectors: ")

train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, Y_train, xtest_tfidf_ngram)



# Naive Bayes on Character Level TF IDF Vectors

print ("NB, CharLevel Vectors: ")

train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, Y_train, xtest_tfidf_ngram_chars)
df.head()
# SVM on Ngram Level TF IDF Vectors

print ("SVM, Count Vectors: ")

train_model(svm.SVC(), xtrain_count, Y_train, xtest_count)



# SVM on Ngram Level TF IDF Vectors

print ("SVM, WordLevel TF-IDF: ")

train_model(svm.SVC(), xtrain_tfidf, Y_train, xtest_tfidf)



# SVM on Ngram Level TF IDF Vectors

print ("SVM, N-Gram Vectors: ")

train_model(svm.SVC(), xtrain_tfidf_ngram, Y_train, xtest_tfidf_ngram)



# SVM on Ngram Level TF IDF Vectors

print ("SVM, CharLevel Vectors: ")

train_model(svm.SVC(), xtrain_tfidf_ngram_chars, Y_train, xtest_tfidf_ngram_chars)



print("done")