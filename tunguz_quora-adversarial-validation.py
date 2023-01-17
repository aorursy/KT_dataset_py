# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import f1_score, roc_auc_score

import gc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from scipy.sparse import hstack
from scipy.sparse import coo_matrix
from tqdm import tqdm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train['target'] = 0
test['target'] = 1
train_test = pd.concat([train, test], axis =0)
train_test.tail()
target = train_test['target'].values
target = train_test['target'].values

text = train_test['question_text']


del train, test, train_test
gc.collect()


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=8000)
word_vectorizer.fit(text)
word_features = word_vectorizer.transform(text)

del word_vectorizer
gc.collect()

kf = KFold(n_splits=5, shuffle=True, random_state=43)
oof_pred = np.zeros([target.shape[0],])

for i, (train_index, val_index) in tqdm(enumerate(kf.split(target))):
    x_train, x_val = word_features[list(train_index)], word_features[list(val_index)]
    y_train, y_val = target[train_index], target[val_index]
    classifier = LogisticRegression(C=5, solver='sag')
    classifier.fit(x_train, y_train)
    val_preds = classifier.predict_proba(x_val)[:,1]
    oof_pred[val_index] = val_preds
    print(f1_score(y_val, val_preds > 0.1))

score = 0
thresh = .5
for i in np.arange(0.1, 1.001, 0.01):
    temp_score = f1_score(target, (oof_pred > i))
    if(temp_score > score):
        score = temp_score
        thresh = i

print("CV: {}, Threshold: {}".format(score, thresh))
roc_auc_score(target, oof_pred)
