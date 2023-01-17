

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input/jigsaw-toxic-comment-classification-challenge/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
print('Train: ',train.shape)

print('Test: ',test.shape)
train.head(10)
test.head()
train.describe()
train.isnull().sum()
test.isnull().sum()
temp=train.iloc[:,2:-1]

corr=temp.corr()

plt.figure(figsize=(8,6))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values, annot=True)
import re, string

tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): 

    return tok.sub(r' \1 ', s).split()
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_word = TfidfVectorizer(lowercase = True,tokenizer=tokenize,stop_words="english",max_features=100,analyzer='word')
features_train_word = vectorizer_word.fit_transform(train["comment_text"])

features_test_word = vectorizer_word.transform(test["comment_text"])
vectorizer_char = TfidfVectorizer(lowercase=True,tokenizer=tokenize,stop_words="english",max_features=100,analyzer='char')
features_train_char = vectorizer_char.fit_transform(train["comment_text"])

features_test_char = vectorizer_char.fit_transform(test["comment_text"])
from scipy import sparse

y = train[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']]

X = sparse.hstack([features_train_word,features_train_char])

x_test = sparse.hstack([features_test_word, features_test_char])
from sklearn.linear_model import LogisticRegression
target_label = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
predicted = np.zeros((x_test.shape[0],y.shape[1]))
for i,label in enumerate(target_label):

    lr = LogisticRegression(C=2,random_state = i,class_weight = 'balanced')

    print('Building {} model for column:{''}'.format(i,label)) 

    lr.fit(X,y[label])

    predicted[:,i] = lr.predict_proba(x_test)[:,1]
from sklearn.metrics import classification_report

label = 'insult'

y_pred = lr.predict(X)

classification_report(y[label],y_pred)
y_predicted_labels = lr.predict_proba(X)[:,1]

from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(y[label], y_predicted_labels)

metrics.auc(fpr, tpr)
test_predicted = pd.DataFrame(predicted,columns=y.columns)

submission = pd.concat([test['id'],test_predicted],axis=1)

submission.to_csv('submit.csv',index=False)

submission.head()