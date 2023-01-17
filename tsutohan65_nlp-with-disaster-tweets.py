# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/nlp-with-disaster-tweets-cleaning-data/train_data_cleaning.csv')

dt = pd.read_csv('/kaggle/input/nlp-with-disaster-tweets-cleaning-data/test_data_cleaning.csv')
df
dt
x_train = df.iloc[:,3]

x_train
y_train = df.target

y_train
x_test = dt.iloc[:,3]

x_test
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics
# Initialize a CountVectorizer object: count_vectorizer

count_vectorizer = CountVectorizer(stop_words='english')
# Transform the training data using only the 'text' column values: count_train 

count_train = count_vectorizer.fit_transform(x_train.values)

count_train
# Transform the test data using only the 'text' column values: count_test 

count_test = count_vectorizer.transform(x_test.values)

count_test
nb_classifier = MultinomialNB()
### Fit the classifier to the training data

nb_classifier.fit(count_train, y_train)
### Create the predicted tags: pred

pred = nb_classifier.predict(count_test)

pred
len(pred), np.sum(pred)
submission = pd.DataFrame({

        "id": dt["id"],

        "target": pred

    })
submission.to_csv('submission.csv',index = False)