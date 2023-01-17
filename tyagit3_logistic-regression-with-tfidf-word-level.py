import numpy as np

import pandas as pd

import os

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from scipy.sparse import hstack
!dir '../input'
INPUT_COLUMN = 'comment_text'

DATA_PATH = '../input'

train = pd.read_csv(os.path.join(DATA_PATH,'train.csv'))

test = pd.read_csv(os.path.join(DATA_PATH,'test.csv'))
word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 1),

    max_features=10000)
text = pd.concat([train[INPUT_COLUMN], test[INPUT_COLUMN]])
def preprocess(data):

    '''

    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution

    '''

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))

    return data
text = preprocess(text)
word_vectorizer.fit(text)
x_train = word_vectorizer.transform(train[INPUT_COLUMN])

x_test = word_vectorizer.transform(test[INPUT_COLUMN])
y = np.where(train['target'] >= 0.5, 1, 0)
classifier = LogisticRegression(C=0.1, solver='sag')



cv_score = np.mean(cross_val_score(classifier, x_train, y, cv=3, scoring='roc_auc'))
print('CV score is {}'.format(cv_score))



classifier.fit(x_train, y)

predictions = classifier.predict_proba(x_test)[:, 1]
submission = pd.DataFrame.from_dict({

    'id': test['id'],

    'prediction': predictions

})
submission.to_csv('submission.csv', index=False)