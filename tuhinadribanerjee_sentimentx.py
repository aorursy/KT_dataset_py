# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import urllib

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/sentiment_train.csv')
df
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(color_codes=True)
df.head()
df.shape
pos_sentiment_count = len(df[df.label == 1])

neg_sentiment_count = len(df[df.label == 0])

print(pos_sentiment_count, neg_sentiment_count,sep="\n")
plt.figure( figsize=(6,5))

ax = sns.countplot(x='label', data=df)
# Initialize the CountVectorizer

count_vectorizer = CountVectorizer()



# Create the dictionary from the corpus

feature_vector = count_vectorizer.fit(df.sentence )



# Get the feature names

features = feature_vector.get_feature_names()

print("Total number of features: ", len(features))



df_features = count_vectorizer.transform(df.sentence)

print(df_features.shape)
# Converting the matrix to a dataframe

train_ds_df = pd.DataFrame(df_features.todense())

print (train_ds_df)



# Setting the column names to the features i.e. words

train_ds_df.columns = features



print(df[4:12])

print(train_ds_df.iloc[4:12, 204:212])



print(train_ds_df[['brokeback', 'mountain', 'is', 'such', 'horrible', 'movie']][0:1])