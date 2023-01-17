# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Read the data

import pandas as pd  

train = pd.read_csv("train.csv", header=-1)

test = pd.read_csv("test.csv", header=0)
#drop all NA

import numpy as np

train = train.replace(np.nan, '', regex=True)

test = test.replace(np.nan, '', regex=True)
#Vectorize word inputs

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import numpy as np
#extract info from each of 3 data columns

xtrain1 = np.transpose(train[[1]].values)[0]

xtrain2 = np.transpose(train[[2]].values)[0]

xtrain3 = np.transpose(train[[3]].values)[0]
xtest1 = test.title.values

xtest2 = test.content.values

xtest3 = test.best_answer.values
#Read the labels

y = np.transpose(train[[0]].values)[0]

from sklearn import preprocessing

#encode y into numbers

lbl_enc = preprocessing.LabelEncoder()

y_ = lbl_enc.fit_transform(train[[0]].values)

y_
#TF - IDF transformation. These parameters are key to achieve high score.

tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')
# Fitting TF-IDF to both training and test sets (semi-supervised learning)

tfv.fit(xtrain1)

xtrain1_tfv =  tfv.transform(xtrain1) 

xtest1_tfv =  tfv.transform(xtest1) 



tfv.fit(xtrain2)

xtrain2_tfv =  tfv.transform(xtrain2)

xtest2_tfv =  tfv.transform(xtest2) 



tfv.fit(xtrain3)

xtrain3_tfv =  tfv.transform(xtrain3) 

xtest3_tfv =  tfv.transform(xtest3)
#The vectorizing process takes a lot of time so we have to save the data, so that we can reuse it

import scipy

scipy.sparse.save_npz('xtrain1_tfv.npz', xtrain1_tfv)

scipy.sparse.save_npz('xtrain2_tfv.npz', xtrain2_tfv)

scipy.sparse.save_npz('xtrain3_tfv.npz', xtrain3_tfv)

scipy.sparse.save_npz('xtest1_tfv.npz', xtest1_tfv)

scipy.sparse.save_npz('xtest2_tfv.npz', xtest2_tfv)

scipy.sparse.save_npz('xtest3_tfv.npz', xtest3_tfv)
#load the data again

xtrain1_tfv = scipy.sparse.load_npz('xtrain1_tfv.npz')

xtrain2_tfv = scipy.sparse.load_npz('xtrain2_tfv.npz')

xtrain3_tfv = scipy.sparse.load_npz('xtrain3_tfv.npz')

xtest1_tfv = scipy.sparse.load_npz('xtest1_tfv.npz')

xtest2_tfv = scipy.sparse.load_npz('xtest2_tfv.npz')

xtest3_tfv = scipy.sparse.load_npz('xtest3_tfv.npz')
#stack them to create the whole train and test data set

train_tfv = scipy.sparse.hstack([xtrain1_tfv, xtrain2_tfv, xtrain3_tfv])

test_tfv = scipy.sparse.hstack([xtest1_tfv, xtest2_tfv, xtest3_tfv])
#I tried so many models: CNN, LSTM, FFN, SVM, xgboost, etc. But they are inefficient

#(runs so slow and the score produced is not much higher). They also need a lot of tuning

#Logistic regression works best in the sense that it is fast and produces high score



#Fit Logistic Regression to CTV data

from sklearn.linear_model import LogisticRegression

clf_tfv = LogisticRegression(C=1.0)

clf_tfv.fit(train_tfv, y_)
y_log_tfv = clf_tfv.predict_proba(test_tfv)
y_log_tfv_out = np.zeros((60000,))

for i in range(60000):

    y_log_tfv_out[i] = np.argmax(y_log_tfv[i]) + 1
id_ = np.zeros((60000,))

for i in range(60000):

    id_[i] = i + 1
log_tfv_output = np.hstack((np.reshape(id_,(60000,1)),np.reshape(y_log_tfv_out, (60000,1))))

np.savetxt("log_all_tfv_updated.csv", log_tfv_output, delimiter = ",")