# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/emails.csv", encoding= "latin-1")
data.spam.value_counts()
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(data["text"],data["spam"], test_size=0.2, random_state=10)
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(stop_words="english")

vect.fit(train_X)

print(vect.get_feature_names()[0:20])

print(vect.get_feature_names()[-20:])
X_train_df = vect.transform(train_X)

X_test_df = vect.transform(test_X)

type(X_test_df)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = MultinomialNB(alpha=1.8)

model.fit(X_train_df,train_y)

pred = model.predict(X_test_df)

accuracy_score(test_y, pred)

print(classification_report(test_y, pred , target_names = ["Ham", "Spam"]))
confusion_matrix(test_y,pred)