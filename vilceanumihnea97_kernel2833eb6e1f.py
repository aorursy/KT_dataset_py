import numpy as np

import pandas as pd

import os

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import TweetTokenizer

%matplotlib inline
tokenize = TweetTokenizer().tokenize
print(os.listdir())

train = pd.read_json('/kaggle/input/whats-cooking-kernels-only/train.json')

test = pd.read_json('/kaggle/input/whats-cooking-kernels-only/test.json')



print(train.head())



ytrain = train['cuisine']

print(ytrain.head(5))



Id = test['id']

print(Id.head(5))
train['cuisine'].value_counts().plot(kind='bar')
def arraytotext(records):

    return [" ".join(record).lower() for record in records]
tfidf = TfidfVectorizer(binary=True)
train2 = train

print((train2['ingredients'][0]))

print(arraytotext(train2['ingredients'][0]))
train_features = tfidf.fit_transform(arraytotext(train['ingredients']))

test_features = tfidf.transform(arraytotext(test['ingredients']))
classifier = SVC(C=200, kernel='rbf', degree=3,gamma=1, \

                 coef0=1, shrinking=True,tol=0.001, probability=False,\

                 cache_size=200,class_weight=None, verbose=False,\

                 max_iter=-1,decision_function_shape=None,\

                 random_state=None)
model = OneVsRestClassifier(classifier)

scores = cross_val_score(classifier,train_features, ytrain, cv=2)

print ("Accuracy: %0.2f (+/- %0.2f)" % \

       (scores.mean(), scores.std() * 2))
model.fit(train_features, ytrain)
predictions = model.predict(test_features)

print(predictions)
submission = pd.DataFrame()

submission['id'] = Id

submission['cuisine'] = predictions

submission.to_csv('submission.csv', index=False)
