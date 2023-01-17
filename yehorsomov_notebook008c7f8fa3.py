import numpy as np

import pandas as pd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import string

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss

from sklearn.metrics import accuracy_score

from scipy.sparse import hstack
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv').fillna('unknown')
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']



X_train = train['comment_text']

y = train[class_names]



X_test = test['comment_text']
all_text =  pd.concat([X_train, X_test])
vectorizer = TfidfVectorizer(

                       min_df=3, max_df=0.8, 

                       ngram_range=(1, 2),

                       strip_accents='unicode',

                       sublinear_tf=True,)
vectorizer.fit(all_text)
train_word = vectorizer.transform(X_train)

test_word = vectorizer.transform(X_test)
char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    ngram_range=(1, 4),

    max_features=20000)
char_vectorizer.fit(all_text)
train_char = char_vectorizer.transform(X_train)

test_char = char_vectorizer.transform(X_test)
train_feat = hstack([train_char, train_word])

test_feat = hstack([test_char, test_word])
y_pred = pd.read_csv('../input/sample_submission.csv')



for c in class_names:

    clf = LogisticRegression(C=4)

    clf.fit(train_feat, y[c])

    y_pred[c] = clf.predict_proba(test_feat)[:,1]

    pred_train = clf.predict_proba(train_feat)[:,1]

    print('log loss ',c, ':', log_loss(y[c], pred_train))
y_pred.to_csv("mysubmission.csv", index=False)