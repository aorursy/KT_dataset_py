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
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train_df.head(10)
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test_df.head(10)
print('train shape: ', train_df.shape)

print('test shape: ', test_df.shape)
y_train = train_df['target']
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

train_vectors = vectorizer.fit_transform(train_df['text'])

test_vectors = vectorizer.transform(test_df['text'])

from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()

reg.fit(train_vectors,y_train)

y_pred = reg.predict(test_vectors)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission['target'] = y_pred

sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)
