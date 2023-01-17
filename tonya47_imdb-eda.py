# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = os.path.join(dirname, filename)
df = pd.read_csv(path)
df.head(3)
df.info()      # df.shape = (1000, 12)
# define a function to calulate the percentage of missing values and plot it out.

def chk_ms_val(data):
  # prepare for plot the missing value
  percent_missing = data.isnull().sum() / len(data)
  percent_missing_df = pd.DataFrame({'percent_missing': percent_missing})
  percent_missing_df = percent_missing_df.sort_values(['percent_missing'], ascending = False).reset_index(drop=False)

  # define the style of the plot
  sns.set(style="whitegrid")
  
  # Initialize the matplotlib figure
  f, ax = plt.subplots(figsize=(6, 12))
  # Plot the total crashes
  sns.set_color_codes("pastel")
  sns.barplot(x="percent_missing", y="index", data=percent_missing_df,
              label="Total missing", color="b")
    
chk_ms_val(df)
null_data = df[df.isnull().any(axis=1)]
null_data.head(3)                         # these are the subset with empty values.
train_data = df.dropna()
train_data.shape                          # the rest of the dataset become trainset.
temp_RM = train_data.loc[:,('Rating', 'Metascore')]
temp_RM.plot(x = 'Rating', y = 'Metascore', kind = 'scatter')
temp_RM.corr()
temp_year = train_data.loc[:, 'Year']
temp_year.plot(kind = 'hist', figsize=(10, 6))
temp_AbvEt = train_data[train_data["Rating"] >= 8]
temp_AbvEt.head(3)                                  # shape is (70, 12)
temp_AbvEt_DA = temp_AbvEt.loc[:, ('Title', 'Director', 'Actors')]

# Split the items into individules columns.
temp_T = temp_AbvEt_DA['Actors'].str.split(",", n=10, expand = True)      # This is the common way to deal with multiple items in one cell.
temp_AbvEt_DA = pd.concat([temp_AbvEt_DA, temp_T], axis=1).rename(columns={0: 'Actor1',1: 'Actor2',2: 'Actor3',3: 'Actor4'}).drop('Actors', axis=1)

temp_AbvEt_DA.head(3)
# define a function for profile of 'Director'
def showmethebest(data, column):
  pd.DataFrame(data[column].value_counts()).reset_index().plot(x='index', y=column, kind = 'bar', figsize=(12,8))

showmethebest(temp_AbvEt_DA, 'Director')
# use the same function on actor1 to actor4
for i in ['Actor1', 'Actor2', 'Actor3', 'Actor4']:
  showmethebest(temp_AbvEt_DA, i)                    
# Get the target subdata
temp_GD = train_data.loc[:,('Genre', 'Director')]
temp_T = temp_GD['Genre'].str.split(",", n=10, expand = True)
temp_GD = pd.concat([temp_GD, temp_T], axis=1).rename(columns={0: 'Genre1',1: 'Genre2',2: 'Genre3'}).drop('Genre', axis=1)

temp_GD.head()
# reshape the dataset 

# stack
yoyo = temp_GD.set_index('Director')
yoyo = yoyo.stack().reset_index().drop('level_1', axis=1)
yoyo.columns = ['Director', 'Genre']

# pivot
yoyo = yoyo.reset_index().groupby(["Director", "Genre"])["index"].count().reset_index(name='count')
New_temp_GD = yoyo.pivot(index='Director', columns='Genre',values='count')
New_temp_GD.fillna(0, inplace=True)

New_temp_GD.head(3)
# visualisation -- draw a series of plots. For each Genre of movie, who are the top 10 Directors.

New_temp_GD = New_temp_GD.reset_index()

# Loop out the series plot to show 
for col in New_temp_GD.columns[1:21]:
  X = New_temp_GD['Director'].tolist()
  Y = New_temp_GD[col].apply(int).tolist()
  Z = [x for _,x in sorted(zip(Y,X))]

  kkkk = pd.DataFrame({'Director': Z[-10:][::-1], col : sorted(Y)[-10:][::-1] })

  act_plot = sns.catplot(x = 'Director', y = col, data=kkkk, height=3, aspect=3, kind = "bar")
  act_plot.set_xticklabels(rotation=45, horizontalalignment='right')
train_data.plot(x='Rating', y='Revenue (Millions)',kind = 'scatter')
temp_RtRv = train_data.loc[:,('Rating', 'Revenue (Millions)')]
temp_RtRv.head(3)
# what is the correlation say?
temp_RtRv.corr()
# get the subset data
temp_tv = train_data.loc[:, ('Title', 'Votes')]


# plot, for the top 30 movies by votes
temp_tv.sort_values(by='Votes', ascending=False).nlargest(30, 'Votes', keep = 'all').plot(x = 'Title', y='Votes', kind='bar',figsize=(12,7))
# get the subset
temp_tr = train_data.loc[:, ('Title', 'Rating')]

temp_tr.sort_values(by='Rating', ascending=False).nlargest(30, 'Rating', keep = 'last').plot(x = 'Title', y='Rating', kind='bar',figsize=(12,7))
# temp_tr.sort_values(by='Rating', ascending=False).nlargest(30, 'Rating', keep = 'all').plot(x = 'Title', y='Rating', kind='bar',figsize=(12,7))
