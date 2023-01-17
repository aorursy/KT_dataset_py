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
train=pd.read_csv('../input/nlp-getting-started/train.csv')

test= pd.read_csv('../input/nlp-getting-started/test.csv')

train.head()
train.shape
train.isnull().sum()
not_disaster = train[train['target']==0]

not_disaster.head()
disaster = train [train['target']==1]

disaster.head()
no_of_disaster = len(disaster)

no_of_disaster

no_of_not_disaster = len(not_disaster)

targets = train['target'].value_counts()
import seaborn as sns

sns.barplot(x=targets.index,y=targets)
train['text'] = train['text'].apply(lambda x: x.lower())

test['text'] = test['text'].apply(lambda x: x.lower())
import re

import string

def clean_text(text):

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
train['text'] = train['text'].apply(lambda x: clean_text(x))

test['text'] = test['text'].apply(lambda x: clean_text(x))
train.head()
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
stopwords = stopwords.words('english')
def remove_stopwords(text):

    text = [item for item in text.split() if item not in stopwords]

    return ' '.join(text)



train['updated_text'] = train['text'].apply(remove_stopwords)

test['updated_text'] = test['text'].apply(remove_stopwords)



train.head()
from nltk.stem.snowball import SnowballStemmer



stemmer = SnowballStemmer("english")



def stemming(text):

    text = [stemmer.stem(word) for word in text.split()]

    return ' '.join(text)



train['stemmed_text'] = train['updated_text'].apply(stemming)

test['stemmed_text'] = test['updated_text'].apply(stemming)

train.head()
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(analyzer='word',binary=True)

cv.fit(train['stemmed_text'])



trained = cv.fit_transform(train['stemmed_text'])

tested = cv.transform(test['stemmed_text'])
y=train['target']
from sklearn.svm import SVC

from sklearn import model_selection



model1 = SVC()

scores = model_selection.cross_val_score(model1,trained,y,cv=3,scoring='f1')
scores.mean()
from sklearn.naive_bayes import MultinomialNB

model2 = MultinomialNB(alpha=1)

scores = model_selection.cross_val_score(model2, trained, y, cv=3, scoring="f1")

scores.mean()
from sklearn.linear_model import LogisticRegression

model3 = LogisticRegression()

scores = model_selection.cross_val_score(model3,trained,y,cv=3,scoring='f1')

scores.mean()
model2.fit(trained,y)
submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

submission['target'] = model2.predict(tested)
submission.head()
submission.to_csv("submission.csv", index=False)