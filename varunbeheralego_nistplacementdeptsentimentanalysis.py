# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import nltk, re, time

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

import random
data = pd.read_csv("../input/placement_survey_data.csv")

data.head()
data = data[['Feedback regarding all placement related activities conducted till date.','How would you rate the placement department?']]

data.columns = ['review','liked']

data.head()
data['liked'] = (data['liked'] > 2).astype(int)

data.head()
corpus = []

for i in range(len(data.index)):

    review = re.sub('[^a-zA-Z]', ' ', data['review'][i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)
cv = CountVectorizer(max_features = 252) #choosing top 'n' most frequent tokens

X = cv.fit_transform(corpus).toarray()

y = data.iloc[:,-1].values

np.asarray(X.sum(axis=0)) #frequency of tokens
indices = range(len(data.index))

X_train, X_test,y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size = 0.3, random_state = random.randint(0, 1000))
sc = StandardScaler()

X_train = sc.fit_transform(X_train.astype(float))

X_test = sc.transform(X_test.astype(float))
clf = MLPClassifier(solver='sgd', activation = 'tanh', alpha=1e-5, hidden_layer_sizes=(64, 64, 64, 32, 16, 4, 4, 4), learning_rate_init = 0.15, learning_rate = 'adaptive', max_iter = 10000, random_state = random.randint(0, 1000), verbose = False, shuffle = True, early_stopping = True)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred)
d = {'test': y_test, 'pred': y_pred}

testpred = pd.DataFrame(data=d)

testpred.insert(0, column = 'review', value = data['review'][indices_test].values)

print(testpred.head())

testpred.to_csv("output.csv")