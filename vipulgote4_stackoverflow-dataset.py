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
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
import lightgbm as lgb
import catboost as ct
import sklearn
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import KFold,RepeatedStratifiedKFold,RandomizedSearchCV,GridSearchCV,cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer,ENGLISH_STOP_WORDS
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, plot_confusion_matrix, plot_precision_recall_curve
dataset=pd.read_csv('/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv')
dataset.head()
draft_dataset=dataset
dataset.sort_values('CreationDate',inplace=True)
lb=LabelEncoder()
new_data=lb.fit_transform(dataset.CreationDate)
#dataset['DateCatCOl']=new_data
dataset.head()
dataset.drop(['Id','CreationDate'],axis=1,inplace=True)
dataset.head()
dataset.Y.value_counts().to_dict()
dataset['Y']=dataset.Y.map({'LQ_CLOSE': 0, 'LQ_EDIT': 1, 'HQ': 2})
dataset
import re
def clean_tags(T):
    T=T.lower()
    text=re.sub(r'<','',T)
    text=re.sub(r'>',' ',text)
    return text

dataset['Tags']=dataset['Tags'].map(clean_tags)
dataset.head()
dataset.Tags.value_counts()[:10]
count_v=CountVectorizer()
tags_vecorized=count_v.fit_transform(dataset.Tags)
dataset.drop('Tags',axis=1,inplace=True)
dataset.head()
import os
def clean_body(x):
    x=x.lower()
    x=re.sub(r'[^(a-zA-Z)\s]','', x)
    return x

dataset['Body']=dataset.Body.map(clean_body)
dataset.head()
dataset['CombineTextandBody']=dataset['Title']+' '+dataset['Body']
dataset.head()
dataset.drop(['Title','Body'],axis=1,inplace=True)
dataset.head()
label=dataset.pop('Y')
dataset.head()
train_x,test_x,train_y,test_y=train_test_split(dataset,label,test_size=0.15,random_state=42)
train_x.shape,test_x.shape
train_x.head()
tfidf=TfidfVectorizer()
transform_text_train=tfidf.fit_transform(train_x.CombineTextandBody)
transform_text_test=tfidf.transform(test_x.CombineTextandBody)
transform_text_train.shape
rskf=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
lr_classifier = LogisticRegression(C=1.)
#lr_classifier.fit(transform_text_train, train_y)
transform_text_test.shape,test_y.shape
#print(f"Validation Accuracy of Logsitic Regression Classifier is: {(lr_classifier.score(transform_text_test, test_y))*100:.2f}%")
score=cross_val_score(lr_classifier,transform_text_train, train_y,cv=3,n_jobs=-1)
score
xg_classifier = XGBClassifier(n_estimators=500,n_jobs=-1,random_state=42)
#xg_classifier.fit(transform_text_train, train_y)
#print(f"Validation Accuracy of XGBoost Clf. is: {(xg_classifier.score(transform_text_test, test_y))*100:.2f}%")
nb_classifier = MultinomialNB()
nb_classifier.fit(transform_text_train, train_y)
# Print the accuracy score of the classifier
print(f"Validation Accuracy of Naive Bayes Classifier is: {(nb_classifier.score(transform_text_test, test_y))*100:.2f}%")
lgb_model=lgb.LGBMClassifier()
lgb_model.fit(transform_text_train, train_y)
# Print the accuracy score of the classifier
print(f"Validation Accuracy of lgb_model Classifier is: {(lgb_model.score(transform_text_test, test_y))*100:.2f}%")

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
            'learning_rate':[0.1,0.01,0.05,0.001,0.005,0.03,0.003,0.006,0.08]}
lgb_model.get_params()
RS=RandomizedSearchCV(
    estimator=lgb_model, param_distributions=param_test,
    cv=3,
    refit=True,
    random_state=42,
    verbose=True)
RS.fit(transform_text_train, train_y)

RS.best_estimator_,RS.best_params_,RS.best_score_
parameter_list={
    'C':[0.10,0.6,0,3.0,4.0,5.,6.,9.,0.11,0.12,0.15,0.14,0.20],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

lr_classifier_2 = LogisticRegression(n_jobs=-1,random_state=42)

log_tune=RandomizedSearchCV(
    estimator=lr_classifier_2, param_distributions=parameter_list,
    cv=3,
    refit=True,
    random_state=42,
    verbose=True)

log_tune.fit(transform_text_train, train_y)
log_tune.best_estimator_,log_tune.best_params_,log_tune.best_score_
