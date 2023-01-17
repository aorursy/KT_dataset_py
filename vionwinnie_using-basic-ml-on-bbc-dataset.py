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
import pandas as pd

import io

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/bbc-fulltext-and-category/bbc-text.csv')

df.head()
from collections import Counter

Counter(df.category)

df['category'] = df.category.map({'sport':0,'business':1,'tech':2,'entertainment':3,'politics':4})
df
## Set random seed 

seed = 42

np.random.seed(seed) 



## Shuffle Data

def shuffle(df, n=3, axis=0):     

    df = df.copy()

    random_states = [2,42,4]

    for i in range(n):

        df = df.sample(frac=1,random_state=random_states[i])

    return df



new_df = shuffle(df)

new_df
split_idx = int(len(df)*0.8)

print(split_idx)

train_df = new_df.loc[:split_idx,:]

test_df = new_df.loc[split_idx:,:]

print(train_df.groupby(['category'])['text'].count())

print(test_df.groupby(['category'])['text'].count())
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(train_df.text).toarray()

labels = train_df.category

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import cross_val_score



models = [

    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),

    MultinomialNB(),

    LogisticRegression(random_state=0),

    KNeighborsClassifier(n_neighbors=3)

]
CV = 5  # Cross Validate with 5 different folds of 20% data ( 80-20 split with 5 folds )



#Create a data frame that will store the results for all 5 trials of the 3 different models

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = [] # Initially all entries are empty



#For each Algorithm 

for model in models:

    model_name = model.__class__.__name__

    # create 5 models with different 20% test sets, and store their accuracies

    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)

    # Append all 5 accuracies into the entries list ( after all 3 models are run, there will be 3x5 = 15 entries)

    for fold_idx, accuracy in enumerate(accuracies):

        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

cv_df.groupby('model_name').accuracy.mean()

X_test = tfidf.transform(test_df.text)

y_test = test_df.category
test_model = models[1]

test_model.fit(features,labels)

y_pred = test_model.predict(X_test)
from sklearn.metrics import confusion_matrix

import seaborn as sns



conf_mat = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_mat, annot=True, fmt='d',

            xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])

plt.ylabel('Actual')

plt.xlabel('Predicted')

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)