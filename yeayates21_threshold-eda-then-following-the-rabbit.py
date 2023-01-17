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
# Visualizations

import matplotlib.pyplot as plt

%matplotlib inline

# Garbage Collector

import gc

import sys

# Stats

from scipy import stats

# Text analysis

from fuzzywuzzy import fuzz
# load training data

j_df = pd.read_csv("../input/train.csv")
j_df.head()
plt.hist(j_df['target'])

plt.show()
plt.hist(j_df[j_df['target']!=0]['target'])

plt.show()
j_df[j_df['target']!=0]['target'].describe()
plt.hist(j_df[j_df['target']!=0]['target'], 15)

plt.show()
plt.hist(j_df[j_df['target']!=0]['target'], 40)

plt.show()
plt.hist(j_df[j_df['target']!=0]['target'], 100)

plt.show()
print(j_df[(j_df['target']!=0) & (j_df['target']>=0.10) & (j_df['target']<=0.20)]['target'].describe())

print(stats.mode(j_df[(j_df['target']!=0) & (j_df['target']>=0.10) & (j_df['target']<=0.20)]['target']))
j_df[(j_df['target']!=0) & (j_df['target']==1/6)].head()
print("Number of records where toxicity_annotator_count is 1: {}".format(len(j_df[(j_df['toxicity_annotator_count']==1)])))

print("Most common # of annotators:")

print(j_df['toxicity_annotator_count'].value_counts().head())

print("Most annotators: ", max(j_df['toxicity_annotator_count']))
j_df[(j_df['target']!=0) & (j_df['target']==1.0)]['toxicity_annotator_count'].value_counts().head(1000)
j_df[(j_df['target']!=0) & (j_df['target']==1.0) & (j_df['toxicity_annotator_count']==4)].head()
pd.set_option('display.max_colwidth', -1)

j_df[(j_df['target']!=0) & (j_df['target']==1.0) & (j_df['toxicity_annotator_count']==4)]['comment_text'].head(40)
j_df[(j_df['target']!=0) & (j_df['target']==1.0) & (j_df['toxicity_annotator_count']==4)]['comment_text'].iloc[33]
print("Number of annotators:")

print(j_df[(j_df['target']!=0) & (j_df['target']==1.0)]['toxicity_annotator_count'].iloc[16])

print("Comment:")

print(j_df[(j_df['target']!=0) & (j_df['target']==1.0)]['comment_text'].iloc[16])
my_string = j_df[(j_df['target']!=0) & (j_df['target']==1.0)]['comment_text'].iloc[16]



def my_str_compare(x):

    return fuzz.token_sort_ratio(x,my_string)



jdf0 = j_df[j_df['target']==0]

jdf0['distance2mys1'] = jdf0['comment_text'].apply(my_str_compare)

jdf0.sort_values(by=['distance2mys1'], ascending=False).head()
my_string = j_df[(j_df['target']!=0) & (j_df['target']==1.0)]['comment_text'].iloc[0]

print("Toxic string we're going to compare: {}".format(my_string))



def my_str_compare(x):

    return fuzz.token_sort_ratio(x,my_string)



jdf0 = j_df[j_df['target']==0]

jdf0['distance2mys1'] = jdf0['comment_text'].apply(my_str_compare)

print("Showing top 5 similar non-toxic comments:")

jdf0.sort_values(by=['distance2mys1'], ascending=False).head()
my_string = j_df[(j_df['target']!=0) & (j_df['target']==1.0) & (j_df['toxicity_annotator_count']==53)]['comment_text'].iloc[0]

print("Toxic string we're going to compare: {}".format(my_string))



def my_str_compare(x):

    return fuzz.token_sort_ratio(x,my_string)



jdf0 = j_df[j_df['target']==0]

jdf0['distance2mys1'] = jdf0['comment_text'].apply(my_str_compare)

print("Showing top 5 similar non-toxic comments:")

jdf0.sort_values(by=['distance2mys1'], ascending=False).head()