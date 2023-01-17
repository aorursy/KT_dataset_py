# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import urllib
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/spam_or_not_spam (medit).csv",encoding='latin-1')
df
df.shape
df.drop_duplicates(inplace=True)
df.shape
pos_sentiment_count = len(df[df.label == 1])
neg_sentiment_count = len(df[df.label == 0])
print(pos_sentiment_count, neg_sentiment_count,sep="\n")
plt.figure( figsize=(6,5))
ax = sns.countplot(x='label', data=df)
nan_value = float("NaN")
df.replace("", nan_value, inplace=True)
df.dropna(subset = ["email"], inplace=True)
print(df)
count_vectorizer = CountVectorizer()
feature_vector = count_vectorizer.fit(df.email )

features = feature_vector.get_feature_names()
print("Total number of features: ", len(features))

df_features = count_vectorizer.transform(df.email)
print(df_features.shape)
df.shape