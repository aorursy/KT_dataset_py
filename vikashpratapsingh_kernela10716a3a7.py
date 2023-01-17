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
import spacy

from nltk.tokenize import sent_tokenize

#nlp = spacy.load('en')

train_data=pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')

test_data=pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')

sample_data=pd.read_csv('/kaggle/input/quora-insincere-questions-classification/sample_submission.csv')
train_data.head(2)
test_data.head(2)
sample_data.head(2)
train_data[train_data.target==1].head(2)
X=train_data['question_text']

y=train_data['target']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=41)
from sklearn.feature_extraction.text import TfidfVectorizer



tf=TfidfVectorizer()

train_tf= tf.fit_transform(X_train)

test_tf= tf.transform(X_test)
from sklearn.svm import LinearSVC

# Create the LinearSVC model

model = LinearSVC(random_state=1, dual=False)

# Fit the model

model.fit(train_tf, y_train)
# Uncomment and run to see model accuracya

print(f'Model test accuracy: {model.score(test_tf, y_test)*100:.3f}%')
X_val = test_data['question_text']
val_tf= tf.transform(X_val)
pred = model.predict(val_tf)
sample_data['prediction'] = pred
sample_data.to_csv('submission.csv')
sample_data