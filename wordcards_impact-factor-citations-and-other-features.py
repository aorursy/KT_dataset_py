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
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

from textwrap import wrap
filename='../input/impact-factor-of-top-1000-journals/Impact-Factor-Ratings.xlsx'



data_df = pd.DataFrame()



for y in range(2013, 2020):

    tmp_df = pd.read_excel(open(filename,'rb'),sheet_name=str(y))

    start_y, end_y = y-3, y%100

    prefix = str(start_y)+'-'+str(end_y)+' '

    orig_citations, orig_documents = prefix+'Citations', prefix+'Documents'

    tmp_df.rename(columns={orig_citations:'Citations', orig_documents:'Documents'}, inplace=True)

    tmp_df['year'] = y

    data_df = pd.concat([data_df, tmp_df])

    

data_df.reset_index(drop=True, inplace=True)

data_df['HP'] = data_df['Highest percentile'].str.split('\n')

data_df['percentile'] = data_df['HP'].apply(lambda x:x[0])

data_df['rank'] = data_df['HP'].apply(lambda x:x[1])

data_df['genre'] = data_df['HP'].apply(lambda x:x[2])

data_df.drop(['HP', 'Highest percentile'], axis=1, inplace=True)
data_df.head(3)
top_journal = data_df.groupby('Source title').mean()['Citations'].sort_values(

    ascending=False)[:10].index.to_list()

labels = ['\n'.join(wrap(j,25)) for j in top_journal]

features = ['Citations', 'CiteScore', 'Documents', 'SNIP']



fig, ax = plt.subplots(1, 4, sharey=True, figsize=(16,12))

plt.tight_layout()

for i, f in enumerate(features):

    sns.barplot(x=f, y='Source title', data=data_df, order=top_journal, ax=ax[i])

    ax[i].set_yticklabels(labels, fontsize=16)

    ax[i].vlines([data_df[f].mean()], -0.5, 9.5, linestyles='dashed')
frequent_publisher= data_df['Publisher'].value_counts().sort_values(

    ascending=False)[:10].index.to_list()

labels = ['\n'.join(wrap(p,25)) for p in frequent_publisher]



fig, ax = plt.subplots(1, 4, sharey=True, figsize=(16,12))

plt.tight_layout()

for i, f in enumerate(features):

    sns.barplot(x=f, y='Publisher', data=data_df, order=frequent_publisher, ax=ax[i])

    ax[i].set_yticklabels(labels, fontsize=16)

    ax[i].vlines([data_df[f].mean()], -0.5, 9.5, linestyles='dashed')
frequent_genre= data_df['genre'].value_counts().sort_values(ascending=False)[:10].index.to_list()

labels = ['\n'.join(wrap(g,25)) for g in frequent_genre]



fig, ax = plt.subplots(1, 4, sharey=True, figsize=(16,12))

plt.tight_layout()

for i, f in enumerate(features):

    sns.barplot(x=f, y='genre', data=data_df, order=frequent_genre, ax=ax[i])

    ax[i].set_yticklabels(labels, fontsize=16)

    ax[i].vlines([data_df[f].mean()], -0.5, 9.5, linestyles='dashed')
fig, ax = plt.subplots(1, 2, figsize=(14,4))

sns.distplot(data_df['Citations'], ax=ax[0])

ax[0].set_title('Citations original value')

sns.distplot(np.log(data_df['Citations']), ax=ax[1])

ax[1].set_title('Log of Citations');
feature_list = ['CiteScore','Documents', '% Cited', 'SNIP', 'SJR']



fig, ax = plt.subplots(2, 3, figsize=(16,7))

plt.tight_layout()

for i, f in enumerate(feature_list):

    axy, axx = divmod(i,3)

    sns.distplot(data_df[f], kde=False, ax=ax[axy, axx])

plt.delaxes(ax=ax[1,2])
data_df[feature_list].min()
data_df['CiteScore_log'] = np.log(data_df['CiteScore'])

data_df['Citations_log'] = np.log(data_df['Citations'])

data_df['Documents_log'] = np.log(data_df['Documents'])

data_df['SNIP_log1p'] = np.log1p(data_df['SNIP'])

data_df['SJR_log'] = np.log(data_df['SJR'])



new_feature_list = ['CiteScore_log', 'Documents_log', '% Cited', 'SNIP_log1p', 'SJR_log']



fig, ax = plt.subplots(2, 3, sharey=True, figsize=(16, 7))

plt.tight_layout()

for i, f in enumerate(new_feature_list):

    axy, axx = divmod(i, 3)

    sns.scatterplot(x=f, y='Citations_log', data=data_df, ax=ax[axy, axx])

plt.delaxes(ax=ax[1,2])
def plot_scatters(y):

    axy = y-2013

    sns.scatterplot(x='CiteScore', y='Citations', data=data_df[data_df['year']==y], ax=ax[axy, 0])

    sns.scatterplot(x='Documents', y='Citations', data=data_df[data_df['year']==y], ax=ax[axy, 1])

    sns.scatterplot(x='SNIP', y='Citations', data=data_df[data_df['year']==y], ax=ax[axy, 2])

    sns.scatterplot(x='SJR', y='Citations', data=data_df[data_df['year']==y], ax=ax[axy, 3])

    for axx in range(4):

        ax[axy, axx].set_title(y)

        ax[axy, axx].set_xscale('log')

        ax[axy, axx].set_yscale('log')
fig, ax = plt.subplots(7, 4, sharex=True, sharey=True, figsize=(16,18))

plt.subplots_adjust(hspace=0.3)

for year in range(2013, 2020):

    plot_scatters(year)