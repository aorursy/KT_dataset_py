# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats



import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import time

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Auxiliry functions



def get_date_from_string(s):

    try:

        date = time.strptime(s, ' %b %d, %Y')

        return date

    except :

        return time.strptime(" Jan 1, 1900", ' %b %d, %Y')
df_orig = pd.read_csv('../input/employee_reviews.csv')
df_orig.tail()
df_orig.columns
df = df_orig[['company', 'overall-ratings', 'work-balance-stars', 'culture-values-stars', 'carrer-opportunities-stars', 'comp-benefit-stars', 'senior-mangemnet-stars']]



to_convert = ['work-balance-stars', 'culture-values-stars', 'carrer-opportunities-stars', 'comp-benefit-stars', 'senior-mangemnet-stars']



def convert_string_to_float(row, col):

    value = ro

for col in to_convert:

    df[col] = df.apply(lambda x: float(x[col]) if x[col].replace('.','',1).isdigit() else 3, axis=1)

    

df['date'] = df_orig['dates'].apply(get_date_from_string)
# sort the rows by Date

df = df.sort_values('date')
# Let's check how each company is doing with her employees.

df.groupby('company')['overall-ratings'].describe() # Per Company show the overall-ratings statistics.
mean = df.groupby('company')['overall-ratings'].mean()

std = df.groupby('company')['overall-ratings'].std()

labels = ['amazon', 'apple', 'facebook', 'google', 'microsoft', 'netflix']

colors = ['red', 'blue', 'green', 'black', 'gray', 'pink']



fig, axes = plt.subplots(3,3, figsize=(20,15))

plt.figure(figsize=(12,8))

for i, (m, s) in enumerate(zip(mean, std)):

    x = np.arange(m - 4 , m + 4, 0.01)

    y = stats.norm.pdf(x, m, s)

    axes[i // 3][i % 3].plot(x,y, label=labels[i], color=colors[i])

    axes[i // 3][i % 3].plot([m,m], [0,stats.norm.pdf(m,m,s)])

    axes[i // 3][i % 3].set_xlim(0, 10)

    axes[i // 3][i % 3].set_ylim(0, 0.5)

    axes[i // 3][i % 3].legend()
df.groupby('company').boxplot(figsize=(28, 12))
# Only keep rows of the netflix company

df_netflix = df[df['company'] == 'netflix']
df_netflix.describe()
bins = np.arange(1,7)

df_netflix.plot(kind='hist', subplots=True, layout=(2,3), figsize=(20, 12), bins=bins, width=0.5, align='mid', sharex=None)
# Extract carrer-opportunities-stars by position

groups = df_orig.iloc[df_netflix.index].groupby('location')

groups.filter(lambda x : len(x)>20).groupby('location')['overall-ratings'].mean().plot(kind='bar', subplots=True, figsize=(12,8))
groups = df_orig.iloc[df_netflix.index].groupby('job-title')

groups.filter(lambda x : len(x)>20).groupby('job-title')['overall-ratings'].mean().plot(kind='bar', subplots=True, figsize=(12,8))
# Only keep rows of the amazon company

df_amazon = df[df['company'] == 'amazon']
df_amazon.describe()
bins = np.arange(1,7)

df_amazon.plot(kind='hist', subplots=True, layout=(2,3), figsize=(20, 12), bins=bins, width=0.5, align='mid', sharex=None)
groups = df_orig.iloc[df_amazon.index].groupby('location')

groups.filter(lambda x : len(x)> 100).groupby('location')['overall-ratings'].mean().plot(kind='bar', subplots=True, figsize=(12,8))
df_facebook = df[df['company'] == 'facebook']
df_facebook.plot(kind='hist', subplots=True, layout=(2,3), figsize=(20, 12), bins=bins, width=0.5, align='mid', sharex=None)
low_index = df_facebook['overall-ratings'] == 1

df_facebook_low = df_facebook[low_index]



indices = df_facebook_low.index[:10]



# Take the summary column of the first five lowest reviews

summary = df_orig.iloc[indices]['summary']





for s in summary:

    print (s)
df_google = df[df['company'] == 'google']
df_google.plot(kind='hist', subplots=True, layout=(2,3), figsize=(20, 12), bins=bins, width=0.5, align='mid', sharex=None)
df_google = df[df['company'] == 'google']



df_google.groupby(df['date'].map(lambda x: x.tm_year)).mean().plot(kind='bar', subplots=True, layout=(2,3), figsize=(25, 12))

pass
df.groupby(['company', df['date'].map(lambda x: x.tm_year)])['overall-ratings'].mean().plot(kind='bar', subplots=True)
df['date'][0]
import datetime

datetime.datetime.strptime(df['date'][0])