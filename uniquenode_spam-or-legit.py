import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

import re

from pylab import rcParams

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn import metrics 

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.datasets import make_classification

import textblob

import random





from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
spam_df = pd.read_csv('../input/spam.csv', encoding='latin-1')
spam_df = spam_df.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], axis=1)
spam_df = spam_df.rename(columns={'v1':'Type','v2':'Text'})
spam_df.Type.describe()
translator = str.maketrans({key: None for key in string.punctuation})

def cleanText(tweet):

    tweet = tweet.lower()

    tweet = tweet.translate(translator)

    tweet = re.sub(r'\b\w\b', '', tweet)

    return tweet
spam_df.groupby('Type').size()
spam_df['clean_text'] = spam_df.Text.map(cleanText)
spam_df.head()
ham_spam = pd.get_dummies(spam_df['Type'],drop_first=True)
spam_df = spam_df.drop('Type',axis = 1)
spam_df = pd.concat([spam_df,ham_spam], axis=1)
spam_df.head()
def tb_score(text):

    return textblob.TextBlob(text).sentiment.polarity

def tb_score_subjectivity(text):

    return textblob.TextBlob(text).sentiment.subjectivity
spam_df['tb_polarity'] = spam_df.clean_text.map(tb_score)

spam_df['tb_subjectivity'] = spam_df.clean_text.map(tb_score_subjectivity)
spam_df.head()
#train, test = sklearn.model_selection.train_test_split(spam_df) DO THIS LAST
def len_words(text):

    return(len(text.split()))
def avg_word_length(length_text):

    return int(round(length_text - sum(spam_df.length)/len(spam_df.length),1))
spam_df['length'] = spam_df.clean_text.map(len_words)
spam_df['length_compared_to_avg'] = spam_df.length.map(avg_word_length)
spam_df.head()
def tb_score_compared_to_avg(score):

    return round(score - sum(spam_df.tb_polarity)/len(spam_df.tb_polarity),2)

def tb_subjectivity_compared_to_avg(score):

    return round(score - sum(spam_df.tb_subjectivity)/len(spam_df.tb_subjectivity),2)

spam_df['tb_polarity_compared_to_avg'] = spam_df.tb_polarity.map(tb_score_compared_to_avg)

spam_df['tb_subjectivity_compared_to_avg'] = spam_df.tb_subjectivity.map(tb_subjectivity_compared_to_avg)
spam_df.head()
new_spam_df = spam_df.drop(['Text','clean_text'],axis=1)
new_spam_df.head()
new_spam_df.corr()
x = new_spam_df.drop('spam', axis=1)

y = new_spam_df['spam']
x['intercept'] = 1.0
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = LogisticRegression()

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print('Accuracy of logistic regression classifier on test set: {:.2f}.'.format(model.score(X_test, y_test)))