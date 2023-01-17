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
train = pd.read_csv("../input/train_E6oV3lV.csv")
test = pd.read_csv("../input/test_tweets_anuFYb8.csv")
train.head()
test.head()
train['label'] = train['label'].astype('category')
train.info()
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re
train['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in train['tweet']]
test['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in test['tweet']]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train['text_lem'],train['label'])
vect = TfidfVectorizer(ngram_range = (1,4)).fit(X_train)
vect_transformed_X_train = vect.transform(X_train)
vect_transformed_X_test = vect.transform(X_test)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

modelSVC = SVC(C=100).fit(vect_transformed_X_train,y_train)
predictionsSVC = modelSVC.predict(vect_transformed_X_test)
sum(predictionsSVC==1),len(y_test),f1_score(y_test,predictionsSVC)
modelLR = LogisticRegression(C=100).fit(vect_transformed_X_train,y_train)
predictionsLR = modelLR.predict(vect_transformed_X_test)
sum(predictionsLR==1),len(y_test),f1_score(y_test,predictionsLR)
modelNB = MultinomialNB(alpha=1.7).fit(vect_transformed_X_train,y_train)
predictionsNB = modelNB.predict(vect_transformed_X_test)
sum(predictionsNB==1),len(y_test),f1_score(y_test,predictionsNB)
modelRF = RandomForestClassifier(n_estimators=20).fit(vect_transformed_X_train,y_train)
predictionsRF = modelRF.predict(vect_transformed_X_test)
sum(predictionsRF==1),len(y_test),f1_score(y_test,predictionsRF)
modelSGD = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3).fit(vect_transformed_X_train,y_train)
predictionsSGD = modelSGD.predict(vect_transformed_X_test)
sum(predictionsSGD==1),len(y_test),f1_score(y_test,predictionsSGD)
vect = TfidfVectorizer(ngram_range = (1,4)).fit(train['text_lem'])
vect_transformed_train = vect.transform(train['text_lem'])
vect_transformed_test = vect.transform(test['text_lem'])
FinalModel = LogisticRegression(C=100).fit(vect_transformed_train,train['label'])
predictions = FinalModel.predict(vect_transformed_test)
submission = pd.DataFrame({'id':test['id'],'label':predictions})
file_name = 'test_predictions.csv'
submission.to_csv(file_name,index=False)
