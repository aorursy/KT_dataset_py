import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np
df = pd.read_csv('../input/reviews/Restaurant_Reviews.tsv', delimiter = '\t')  
print(df.shape)

df.head()
df.isnull().sum()
messages=df.copy()
messages.head(10)
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import re
ps = PorterStemmer()

corpus = []

for i in range(0, len(messages)):

    review = re.sub('[^a-zA-Z]', ' ', messages['Review'][i])

    review = review.lower()

    review = review.split()

    

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus.append(review)
corpus[5]
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=1500,ngram_range=(1,3))

X=tfidf_v.fit_transform(corpus).toarray()
tfidf_v.get_params()
X.shape
y=messages['Liked']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
count_df = pd.DataFrame(X_train, columns=tfidf_v.get_feature_names())
count_df.head()
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

import itertools
classifier=MultinomialNB(alpha=0.1)
previous_score=0

for alpha in np.arange(0,1,0.1):

    sub_classifier=MultinomialNB(alpha=alpha)

    sub_classifier.fit(X_train,y_train)

    y_pred=sub_classifier.predict(X_test)

    score = metrics.accuracy_score(y_test, y_pred)

    if score>previous_score:

        classifier=sub_classifier

    print("Alpha: {}, Score : {}".format(alpha,score))
classifier=MultinomialNB(alpha=0.2)

classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

score = metrics.accuracy_score(y_test, pred)

print("accuracy:   %0.3f" % score)
from sklearn.metrics import classification_report



model_score= (classification_report(y_test, pred))

print(model_score)
import matplotlib.pyplot as plt 

from sklearn.metrics import plot_confusion_matrix

 

plot_confusion_matrix(classifier , X_test, y_test) 

plt.show()
from sklearn.metrics import plot_roc_curve



disp=plot_roc_curve(classifier,X_test, y_test);