"""Import Builtin Python Modules"""

import string

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid', context='notebook', palette='viridis')

sns.despine(top=True, right=True)

import re



%matplotlib inline



"""Import Natural Language Toolkit Modules"""

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



"""Import SciKit Learn Modules"""

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.metrics import classification_report, accuracy_score, log_loss, confusion_matrix



"""Import IO modules for Google Colab"""

import io
nltk.download('stopwords')
"""Change to code when not using Google colab to import data"""

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('..input/test.csv')
# Concatenate test and train

df = pd.concat([train, test], axis = 0, sort=False)
sum_mbti = train[['type', 'posts']].groupby('type').count().sort_values(by='posts', ascending=False)
#Bar charts for missing data

#f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=sum_mbti.index, y=sum_mbti.posts,palette='plasma')



plt.xlabel('Type', fontsize=12)

plt.ylabel('Number Of posts', fontsize=12)
train['words_per_comment'] = train['posts'].apply(lambda x: len(x.split())/50)
plt.figure(figsize=(15,10))

sns.swarmplot("type", "words_per_comment", data=train)
df2 = train.copy()
def make_pie(data, label1, label2):

    labels = label1, label2

    dat = [((data==1).sum()/len(data)), ((data!=1).sum()/len(data))]

    colors = ['#ff9999','#66b3ff']



    # Plot

    plt.pie(x = dat, labels=labels, colors=colors,explode = (0, 0.1),pctdistance=0.8, autopct='%1.f%%', shadow=True, startangle=90)

    centre_circle = plt.Circle((0,0),0.50,fc='white')

    fig = plt.gcf()

    fig.gca().add_artist(centre_circle)
df2['mind'] = [1 if x[0] == 'E' else 0 for x in df2['type']]

df2['energy'] = [1 if x[1] == 'N' else 0 for x in df2['type']]

df2['nature'] = [1 if x[2] == 'T' else 0 for x in df2['type']]

df2['tactics'] = [1 if x[3] == 'J' else 0 for x in df2['type']]
#plt.subplot(221)

plt.subplots(figsize=(10,7))



plt.subplot(2,2,1)

make_pie(df2['mind'], 'Extrovert (E)', 'Introvert (I)')



plt.subplot(2,2,2)

make_pie(df2['energy'], "Intuition (N)","Sensing (S)")



plt.subplot(2,2,3)

make_pie(df2['nature'],"Thinking (T)","Feeling (F)")



plt.subplot(2,2,4)

make_pie(df2['tactics'], "Judging (J)" ,"Perceiving (P)")



plt.axis('equal')

plt.tight_layout()

plt.show()
df['posts'] = df['posts'].apply(lambda x: ' '.join(x.split('|||')))
pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

df['posts'] = df['posts'].replace(to_replace = pattern_url, value = '', regex = True)
stop_words = stopwords.words('english')
df['posts'] = df['posts'].apply(lambda x:' '.join(word.lower() for word in x.split()))
df['posts'] = df['posts'].str.replace('[^\w\s]', '')
df['posts'] = df['posts'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
df.head(20)
X= df['posts']

y = train['type']
X_trainn = X[:6506]

X_testt = X[6506:]
X_train, X_test, y_train, y_test = train_test_split(X_trainn, y, test_size = 0.2, random_state = 0)
"""

## CountVectorizer Model

X = df['posts']

y = train['type']

vect = CountVectorizer(lowercase = True, stop_words='english', max_features = 10, ngram_range = (1,3), max_df = 0.5, min_df =2)

X_vect = vect.fit_transform(X)

vectorizer = TfidfVectorizer(stop_words='english',max_df = 0.5, min_df =2)

X_vec = vectorizer.fit_transform(X)

X_trainn = X_vec[:6506,:]

X_testt = X_vec[6506:,:]

"""
"""

## TF-IDF Vectorizer Model

vectorizer = TfidfVectorizer(stop_words='english',max_df = 15, min_df =2)

X_vec = vectorizer.fit_transform(X)

X_trainn = X_vec[:6506,:]

X_testt = X_vec[6506:,:]

"""
rf = RandomForestClassifier(n_estimators = 100) 
sgd = Pipeline([('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', SGDClassifier(loss='hinge', penalty='l2',

                                      alpha=1e-3, random_state=42, 

                                      max_iter=5, tol=None))])
sgd.fit(X_train, y_train)
sgd_y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(sgd_y_pred, y_test))
predictions = sgd.predict(X_testt)
test['Predictions'] = sgd.predict(X_testt)
test.head()
#predictions = pd.DataFrame()



test['mind'] = test['Predictions'].apply(lambda x: 1 if x[0] == 'E' else 0)

test['energy'] = test['Predictions'].apply(lambda x: 1 if x[1] == 'N' else 0)

test['nature'] = test['Predictions'].apply(lambda x: 1 if x[2] == 'T' else 0)

test['tactics'] = test['Predictions'].apply(lambda x: 1 if x[3] == 'J' else 0)

test.head()
test = test.drop(['posts', 'Predictions'], axis=1)
"""Export the resulting submission"""

test.to_csv(r'../input/Final Submission.csv', index = None, header=True)