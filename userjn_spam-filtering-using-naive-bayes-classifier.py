# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt

import seaborn as sb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
span_data_frame = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')

print(span_data_frame.head())

print(span_data_frame.describe())
span_data_frame.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

span_data_frame = span_data_frame.rename(columns={"v1": "class", "v2": "message"})

print(span_data_frame.head())

span_data_frame.info()
plt.xlabel("Labels")

plt.title('Number of ham and spam messages')



sb.countplot(span_data_frame['class'])



plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(span_data_frame['message'], span_data_frame['class'],

                                                    test_size=0.15)
vectorizer = TfidfVectorizer()



X_train_transformed = vectorizer.fit_transform(X_train).toarray()

X_test_transformed = vectorizer.transform(X_test).toarray()


try:



    gnb = GaussianNB()



    gnb.fit(X_train_transformed, Y_train)



    model_prediction = gnb.predict(X_test_transformed)

    model_accuracy_score = accuracy_score(Y_test, model_prediction)



   # print(model_prediction)



    print("Model accuracy score is %f" % model_accuracy_score)



except Exception as e:

    print(e)
