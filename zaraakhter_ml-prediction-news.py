#predicting if news is fake (0) or real (1) based on text using Machine Learning Algorithms

#Random Forest Classifier

#Gaussian Naive Bayes

#Bernoulli Naive Bayes

#Logistic Regression

#XGBoost Classifier
import numpy as np

import pandas as pd

import math 

import matplotlib.pyplot as plt 

import string

import re

import warnings

warnings.filterwarnings('ignore')
fake = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')

true = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
fake.describe()
true.describe()
fake.head(5)
true.head(5)
fake.subject.unique()
fake['subject'].value_counts().plot(kind='bar', figsize=(10,15))

fake.isnull().sum()
#some text is in date column, dropping the rows

fake = fake.drop(columns = ['date'])
true.subject.unique()
true['subject'].value_counts().plot(kind='bar', figsize=(10,15))
true.isnull().sum()
true = true.drop(columns = ['date'])
true.head(5)
fake.head(5)
#cleaning fake dataset title column

titles_fake = []

lowered_fake = []

for i in fake.title:

    

    titles_fake.append(re.sub(r'[^\w\s]','',i))



for i in titles_fake:

    lowered_fake.append(i.lower())
len(lowered_fake)
fake['title'] = lowered_fake
#cleaning fake dataset text column

texts_fake = []

lowered_text_fake = []

for i in fake.text:

    

    texts_fake.append(re.sub(r'[^\w\s]','',i))



for i in texts_fake:

    lowered_text_fake.append(i.lower())
fake['text'] = lowered_text_fake
fake.head()
#adding column for type, FAKE = 0

for i in range(len(fake)):

    fake['type'] = 0
fake.head(5)
#cleaning true dataset title column

titles_true = []

lowered_true = []

for i in true.title:

    

    titles_true.append(re.sub(r'[^\w\s]','',i))



for i in titles_true:

    lowered_true.append(i.lower())
true['title'] = lowered_true
#removing city names from text   

true['text_new'] = true['text'].str.split(')').str[1]
true = true.drop(columns = ['text'])

true = true.rename( columns = {'text_new': 'text'})
##cleaning true dataset title column

text_true = []

lowered_true = []

for i in true.text:

    

    text_true.append(re.sub(r'[^\w\s]','',str(i)))



for i in text_true:

    lowered_true.append(i.lower())
true['text'] = lowered_true
true.head(5)
#adding column for type TRUE = 1

for i in range(len(true)):

    true['type'] = 1
true.head(5)
len(true)
len(fake)
merged_data = pd.concat([fake, true], axis = 0)
merged_data.head(5)
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
#shuffling merged data to balance type of news

merged_data = merged_data.sample(frac=1)
merged_data.head(5)
x = merged_data

x = x.drop(columns = ['type'])

y = merged_data[['type']]
x.head(2)
y.head(2)
import nltk 

from nltk.corpus import stopwords

#using english library for stopwords and setting max_features to 1000 unique words

vectorizer = CountVectorizer(max_features=1000, stop_words=stopwords.words('english'))



#fitting and transforming on text column of dataset

X = vectorizer.fit_transform(x['text']).toarray()
#split train test models



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=1000, random_state=42)

rfc.fit(x_train, y_train.values.ravel()) 
y_pred_rfc = rfc.predict(x_test)
probs_rfc = rfc.predict_proba(x_test)[:,1] 
print('Confusin Matrix: ')

print(confusion_matrix(y_test,y_pred_rfc))

print('\n')

print('Classification Report: ')

print('-------------------------------------------------')

print(classification_report(y_test,y_pred_rfc))

print('-------------------------------------------------')

print('\n')

print('Accuracy is: ',accuracy_score(y_test, y_pred_rfc))
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred_gnb = gnb.fit(x_train, y_train.values.ravel()).predict(x_test)
probs_gnb = gnb.predict_proba(x_test)[:,1] 
print('Confusin Matrix: ')

print(confusion_matrix(y_test,y_pred_gnb))

print('\n')

print('Classification Report: ')

print('-------------------------------------------------')

print(classification_report(y_test,y_pred_gnb))

print('-------------------------------------------------')

print('\n')

print('Accuracy is: ',accuracy_score(y_test, y_pred_gnb))
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB()
y_pred_bnb = bnb.fit(x_train, y_train.values.ravel()).predict(x_test)
probs_bnb = bnb.predict_proba(x_test)[:,1] 
print('Confusin Matrix: ')

print(confusion_matrix(y_test,y_pred_bnb))

print('\n')

print('Classification Report: ')

print('-------------------------------------------------')

print(classification_report(y_test,y_pred_bnb))

print('-------------------------------------------------')

print('\n')

print('Accuracy is: ',accuracy_score(y_test, y_pred_bnb))
from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression(max_iter = 200)
y_pred_lgr = lgr.fit(x_train, y_train.values.ravel()).predict(x_test)
probs_lgr = lgr.predict_proba(x_test)[:,1] 
print('Confusin Matrix: ')

print(confusion_matrix(y_test,y_pred_lgr))

print('\n')

print('Classification Report: ')

print('-------------------------------------------------')

print(classification_report(y_test,y_pred_lgr))

print('-------------------------------------------------')

print('\n')

print('Accuracy is: ',accuracy_score(y_test, y_pred_lgr))
!pip install xgboost
from xgboost import XGBClassifier

xgboost = XGBClassifier()
y_pred_xgbc = xgboost.fit(x_train, y_train.values.ravel()).predict(x_test)
probs_xgbc = xgboost.predict_proba(x_test)[:,1] 
print('Confusin Matrix: ')

print(confusion_matrix(y_test,y_pred_xgbc))

print('\n')

print('Classification Report: ')

print('-------------------------------------------------')

print(classification_report(y_test,y_pred_xgbc))

print('-------------------------------------------------')

print('\n')

print('Accuracy is: ',accuracy_score(y_test, y_pred_xgbc))
from sklearn.metrics import roc_curve, roc_auc_score
#calculate AUC scores



lgr_auc = roc_auc_score(y_test, probs_lgr)

gnb_auc = roc_auc_score(y_test, probs_gnb)

bnb_auc = roc_auc_score(y_test, probs_bnb)

xgbc_auc = roc_auc_score(y_test, probs_xgbc)

rfc_auc = roc_auc_score(y_test, probs_rfc)





# summarize scores

print('Logistic: ROC AUC=%.3f' % (lgr_auc))

print('Guassian NB: ROC AUC=%.3f' % (gnb_auc))

print('Bernoulli NB: ROC AUC=%.3f' % (bnb_auc))

print('XGBoost: ROC AUC=%.3f' % (xgbc_auc))

print('Random Forest: ROC AUC=%.3f' % (rfc_auc))



# calculate ROC curves

lgr_fpr, lgr_tpr, _ = roc_curve(y_test, probs_lgr)

gnb_fpr, gnb_tpr, _ = roc_curve(y_test, probs_gnb)

bnb_fpr, bnb_tpr, _ = roc_curve(y_test, probs_bnb)

xgbc_fpr, xgbc_tpr, _ = roc_curve(y_test, probs_xgbc)

rfc_fpr, rfc_tpr, _ = roc_curve(y_test, probs_rfc)

lw =2

plt.figure(figsize = (15,10))

plt.plot(lgr_fpr, lgr_tpr, lw = lw, label='Logistic ROC curve (area = %0.2f)' % lgr_auc)

plt.plot(gnb_fpr, gnb_tpr, lw = lw, label='Gaussian NB ROC curve (area = %0.2f)' % gnb_auc)

plt.plot(bnb_fpr, bnb_tpr,  lw = lw, label='Bernoulli NB ROC curve (area = %0.2f)' % bnb_auc)

plt.plot(xgbc_fpr, xgbc_tpr, lw=lw,  label='XGBoost ROC curve (area = %0.2f)' % xgbc_auc)

plt.plot(rfc_fpr, rfc_tpr,  lw = lw, label='Random Forest ROC curve (area = %0.2f)' % rfc_auc)

plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()