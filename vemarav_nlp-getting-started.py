# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, ensemble, tree



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
display(train_df.head(4))
# remove null values

columns=['keyword', 'location']

train_df.drop(columns=columns, inplace=True)

test_df.drop(columns=columns, inplace=True)
# vectorize text

count_vectorizer = feature_extraction.text.CountVectorizer()



vectorized_train_samples = count_vectorizer.fit_transform(train_df.text) 

vectorized_test_samples = count_vectorizer.transform(test_df.text)
model = tree.DecisionTreeClassifier()

model.fit(vectorized_train_samples, train_df.target)

model.score(vectorized_train_samples, train_df.target)
test_target = model.predict(vectorized_test_samples)

model.score(vectorized_test_samples, test_target)
test_df['target'] = test_target

test_df.drop(columns=['text']).to_csv('submission.csv', index=False)