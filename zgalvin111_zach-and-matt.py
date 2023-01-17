# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Gather Testing text from csv
path = '../input/train.csv'

trainData = pd.read_csv(path, names=['ID', 'Text', 'Toxic','Severe_Toxic','Obscene', 'Threat', 'Insult', 'Identity_Hate'], low_memory = False, header=0)
X = trainData.Text
y = [trainData.Toxic, trainData.Severe_Toxic, trainData.Obscene, trainData.Threat, trainData.Insult, trainData.Identity_Hate]

n = int(0.75 * len(X))
X_train = X[:n]
X_test = X[n:]

y_train = [trainData.Toxic[:n], trainData.Severe_Toxic[:n], trainData.Obscene[:n], trainData.Threat[:n], trainData.Insult[:n], trainData.Identity_Hate[:n]]
y_test = [trainData.Toxic[n:], trainData.Severe_Toxic[n:], trainData.Obscene[n:], trainData.Threat[n:], trainData.Insult[n:], trainData.Identity_Hate[n:]]
# Turn text to numbers
# Transform X into word counts
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tf_transformer = TfidfTransformer()
X_train = tf_transformer.fit_transform(X_train_counts)
# Test model on each category
means = []
for i in range(len(y_train)):
    mlb = MultinomialNB().fit(X_train, y_train[i])
    # Transform X and then predict
    if i == 0:
        X_test = count_vect.transform(X_test)
        X_test = tf_transformer.transform(X_test)
    predicted = mlb.predict(X_test)
    means.append(np.mean(predicted == y_test[i]))

mean = 0
for i in range(len(means)):
    mean += means[i]

m = mean / len(means)
print(m)
