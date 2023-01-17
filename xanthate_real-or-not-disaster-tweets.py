import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('dark')



import re

import os
# read dataset

file_path = '/kaggle/input/nlp-getting-started/train.csv'

tweet_df = pd.read_csv(file_path)

tweet_df.head(3)
# shape of dataset

data_instances, predictors = tweet_df.shape[0], tweet_df.shape[1]

print(f'There are {data_instances} data instances and {predictors} predictors')
# Null values in location and keyword columns

null_value_kw, null_value_location = tweet_df.location.isnull().sum(), tweet_df.keyword.isnull().sum()

print(f'Null value in "Keyword": {null_value_kw} \nNull value in "Location": {null_value_location}')
tweet_df.target.value_counts()
plt.figure(figsize=(6, 5))

x = tweet_df.target.value_counts()

ax = sns.barplot(x.index, x)

plt.xlabel('target')

plt.ylabel('count')

plt.show()
tweet_df.loc[:10, ['text']]
len(tweet_df[tweet_df.target==1].text.str.len() > 130)
tweet_df[tweet_df.text.str.len() > 130].target.value_counts()
plt.figure(figsize=(8, 6))

tweet_df[tweet_df.target==1].text.str.len()

sns.distplot(tweet_df[tweet_df.target==1].text.str.len(), kde=False)

plt.xlabel("Tweet Length")

plt.ylabel("No. of tweets")

plt.show()
plt.figure(figsize=(8, 6))

tweet_df[tweet_df.target==1].text.str.len()

sns.distplot(tweet_df[tweet_df.target==0].text.str.len(), kde=False)

plt.xlabel("Tweet Length")

plt.ylabel("No. of tweets")

plt.show()
for i in range(10):

    x = tweet_df.text[i]

    print(f"{x}")
tweet_df.loc[:,['text']]
tweet_df['text'][0]
# This is a messy way to do it. Better way is to use lambda function. Implemented below for another case.

N = len(tweet_df)

number_of_words = []

for i in range(N):

    #str(tweet_df['text'][i]).split()

    number_of_words.append(len(str(tweet_df.loc[i,['text']]).split()))



number_of_words = pd.DataFrame(number_of_words, columns=['number_of_words'], dtype=np.int64)

number_of_words.head()
tweet_df = pd.concat([tweet_df, number_of_words], axis=1)
len(tweet_df[tweet_df.target == 1].number_of_words)
ax = sns.distplot(tweet_df[tweet_df.target==1].number_of_words, kde=False, rug=True)

ax.set_yscale('log')

plt.show()
sns.set_style('darkgrid')

#fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.distplot(tweet_df[tweet_df.target==0].text.str.len(), kde=True, color='red')

#ax[0].xlabel('Number of chars in tweet')

sns.distplot(tweet_df[tweet_df.target==1].text.str.len(), kde=True, color='green')

#ax[1].xlabel('Number of chars in tweet')

plt.show()
# just learnt about lambda function implemented it here for the first time;

# would replace the messy code above with this in the next commit

tweet_df['mean_word_length'] = tweet_df.text.apply(lambda x: np.mean([len(n) for n in x.split()]))

tweet_df['mean_word_length'].head()
# plot the distribution comparision between word lengths of disaster and non-diaster tweets.

sns.distplot(tweet_df[tweet_df.target == 1].mean_word_length, kde=True, color='green')

sns.distplot(tweet_df[tweet_df.target == 0].mean_word_length, kde=True, color='red')

plt.show()
corpus = tweet_df[tweet_df.target == 1].text.str.lower().str.split()