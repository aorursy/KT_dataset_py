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
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')
test_id = test.id

train_target = train.target
train.head()
def description(df):

    print(f'Dataset Shape:{df.shape}')

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.iloc[0].values

    summary['Second Value'] = df.iloc[1].values

    summary['Third Value'] = df.iloc[2].values

    return summary

print('Variable Description of  train Data:')

description(train)
# Removing the unneeded "id" column

train = train.drop('id', axis=1)

def replace_nan(data):

    for column in data.columns:

        if data[column].isna().sum() > 0:

            data[column] = data[column].fillna(data[column].mode()[0])







replace_nan(train)

replace_nan(test)
description(train)
import matplotlib.ticker as ticker



plt.figure(figsize=(10,7))

ncount = len(train)



ax = sns.countplot(train.target, palette='coolwarm')

plt.title("TARGET DISTRIBUTION", fontsize = 20)

plt.xlabel('Number of Axles', fontsize = 20)



# Make twin axis

ax2=ax.twinx()



# Switch so count axis is on right, frequency on left

ax2.yaxis.tick_left()

ax.yaxis.tick_right()



# Also switch the labels over

ax.yaxis.set_label_position('right')

ax2.yaxis.set_label_position('left')



ax2.set_ylabel('Frequency [%]', fontsize = 15)

ax.set_ylabel('Count', fontsize = 15)



for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

            ha='center', va='bottom') # set the alignment of the text



# Use a LinearLocator to ensure the correct number of ticks

ax.yaxis.set_major_locator(ticker.LinearLocator(11))



# Fix the frequency range to 0-100

ax2.set_ylim(0,100)

ax.set_ylim(0,ncount)



# And use a MultipleLocator to ensure a tick spacing of 10

ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))



# Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars

ax2.grid(True)
import matplotlib.gridspec as gridspec # to do the grid of plots
bin_list = ['bin_0','bin_1', 'bin_2', 'bin_3', 'bin_4']





grid = gridspec.GridSpec(3, 2) # The grid of chart

plt.figure(figsize=(16,20))



for n, col in enumerate(train[bin_list]): 

    ax = plt.subplot(grid[n]) # feeding the figure of grid

    sns.countplot(x=col, data=train, hue = 'target',palette='Set2') 

    ax.set_ylabel('Count', fontsize=15) # y axis label

    ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label

    ax.set_xlabel(f'{col} values', fontsize=15)

    
num_list_one = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

num_list_two = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
def ploting_cat_fet(df, cols, vis_row, vis_col, palette = 'Set1'):

    grid = gridspec.GridSpec(vis_row, vis_col) # The grid of chart

    plt.figure(figsize=(15, 25)) # size of figure



    for i, col in enumerate(train[cols]):

        ax = plt.subplot(grid[i])

        sns.countplot(x = col, data = train, palette= palette)

        ax.set_ylabel('Count', fontsize=15) # y axis label

        ax.set_title(f'{col} Distribution by Target', fontsize=10) # title label

        ax.set_xlabel(f'{col} values', fontsize=10)
ploting_cat_fet(train, num_list_one, 3,2, palette = 'ch:2.5,-.2,dark=.3')
ploting_cat_fet(train, num_list_two, 5,1)
ord_list = ['ord_1', 'ord_2', 'ord_3','ord_4', 'ord_5']
ploting_cat_fet(train, ord_list,5,1,palette='muted')
test
train = train.drop(['target'], axis=1)

test = test.drop(['id'],axis=1)
data = pd.concat([train, test])
columns = [i for i in data.columns]

dummies = pd.get_dummies(data,columns=columns, drop_first=True,sparse=True)
train = dummies.iloc[:train.shape[0], :]

test = dummies.iloc[train.shape[0]:, :]
train = train.sparse.to_coo().tocsr()

test = test.sparse.to_coo().tocsr()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=0.1,solver='lbfgs')

lr.fit(train, train_target)
prediction = lr.predict_proba(test)[:,1]
submission = pd.DataFrame({'id':test_id,'target':prediction})

submission.to_csv('submission.csv', index=False)
submission.head()