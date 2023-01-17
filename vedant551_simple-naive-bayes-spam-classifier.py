# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # data visualization

sns.set_context('talk')

import matplotlib.style as style 

style.use('seaborn-poster')

style.use('ggplot')

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='Latin1')

data.head()
#drop unnecessary columns and rename 

data.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

data.columns = ['label', 'msg']

data.head()
data.groupby('label').describe()
sns.countplot(data['label'], palette=('Accent'));
data['msg_len'] = data['msg'].apply(lambda x: len(x))
data.hist('msg_len', by='label');
# Split the dataset into training and testing set.

X_train, X_test, y_train, y_test = train_test_split(data['msg'], data['label'], random_state=1)
# Convert text to BoW (Bag of Words). 

count_vector = CountVectorizer()

training_data = count_vector.fit_transform(X_train)

testing_data = count_vector.transform(X_test)
# Apply the naive bayes algorithm.

naive_bayes = MultinomialNB()

naive_bayes.fit(training_data, y_train)

pred = naive_bayes.predict(testing_data)

print("{0:.2%}".format(accuracy_score(y_test,pred)))