# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import re

import nltk

from nltk.corpus import stopwords

from nltk import regexp_tokenize

from nltk.stem import WordNetLemmatizer

import spacy





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv(r'/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv(r'/kaggle/input/nlp-getting-started/test.csv')

train.head()
import re

from nltk.tokenize import WordPunctTokenizer



tokenizer = WordPunctTokenizer()



def clean_text(text):

    text = text.lower()

    text = re.sub(r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*', ' ', text)

    text = ' '.join(tokenizer.tokenize(text))

    return text



train_prep = train.text.apply(lambda x: clean_text(x))

test_prep = test.text.apply(lambda x: clean_text(x))

train['text_prep'] = train_prep

test['text_prep'] = test_prep
nlp = spacy.load('en_core_web_lg')
with nlp.disable_pipes():

    train_vectors = np.array([nlp(text).vector for text in train.text_prep])

    test_vectors = np.array([nlp(text).vector for text in test.text_prep])



print(train_vectors.shape, test_vectors.shape)
X_train = train_vectors

y_train = train.target.to_numpy()



train_x, test_x, train_y, test_y = train_test_split(X_train, y_train, test_size=0.2, random_state=13)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

svc = SVC(kernel='rbf', C=65, probability=True, random_state=13)

rfc = RandomForestClassifier(n_estimators=110, random_state=13)

etc = ExtraTreesClassifier(n_estimators=110, random_state=13)

gbc = GradientBoostingClassifier(n_estimators=110, learning_rate=0.2, random_state=13)
vcf = VotingClassifier(estimators=[('etc', etc), ('svc', svc), ('rfc', rfc), ('gbc', gbc)], voting='soft')

vcf.fit(X_train, y_train)
preds = vcf.predict(test_vectors)

print(accuracy_score((vcf.predict(test_x)), test_y))

print(len(test['id']), len(preds))
submission = pd.DataFrame(columns=['id', 'target'])

submission['id'] = test['id']

submission['target'] = preds

submission.to_csv('submission.csv', index=False)