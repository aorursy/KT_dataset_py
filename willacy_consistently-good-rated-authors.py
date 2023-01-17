# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/books.csv',error_bad_lines=False)

df.head(3)
df[df.text_reviews_count>10][['title','average_rating','text_reviews_count']].sort_values(by=['average_rating','text_reviews_count'], ascending=[0,0]).head(20)
df[(df.text_reviews_count>10)&(df["title"].str.contains(r'Calvin and Hobbes'))][['title','authors','average_rating','text_reviews_count']].sort_values(by=['average_rating','text_reviews_count'], ascending=[0,0])
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
sns.distplot(df.average_rating)
print("Skewness: %f" % df['average_rating'].skew())

print("Kurtosis: %f" % df['average_rating'].kurt())
df = df[df.text_reviews_count>9]
sns.distplot(df.average_rating)
df.describe()
threshold = sum(df.average_rating)/len(df.average_rating)

df['Score'] = [1 if i > threshold else 0 for i in df.average_rating]

df.head(3)
author_df = df[['authors','Score']].groupby('authors').sum()

author_df['book_count']=df['authors'].groupby(df.authors).count()

author_df['ratio'] = author_df['Score']/author_df['book_count']

author_df[author_df.ratio == 1].sort_values('book_count', ascending=False).head(10)