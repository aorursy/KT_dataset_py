import pandas as pd

import datetime

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import ast

%matplotlib inline
df = pd.read_csv('../input/ted_main.csv')
df.head(5)
sns.jointplot(x=df['languages'], y=df['comments'], kind='reg')
# Well, looks like the correlation between number of comments and languages (pearsonr) is moderate.

# Let us try removing some outliers (comments > 2000) and see if we get better results.
temp_df = df[df['comments'] < 2000]

sns.jointplot(x=temp_df['languages'], y=temp_df['comments'], kind='reg')
# The correlation increased slightly, not considerably. 
sns.jointplot(x=temp_df['views'], y=temp_df['comments'], kind='reg')
# Based on the above figure, we can interpret that there is a strong corelation between comments and views.
sns.jointplot(x=df['languages'], y=df['views'], kind='reg')
# Looks like the correlation is moderate (0.38)
# Get Year from datetime input. 

def get_year(date):

    return int(datetime.datetime.utcfromtimestamp(date).strftime('%Y'))
# Get half of day from datetime input

def get_half_of_day(date):

    h = int(datetime.datetime.utcfromtimestamp(date).strftime('%H'))

    if (h > 12):

        return 1

    else:

        return 0
# Get month from datetime input

def get_month(date):

    return int(datetime.datetime.utcfromtimestamp(date).strftime('%m'))
df['year'] = df['film_date'].apply(get_year)

df['month'] = df['film_date'].apply(get_month)

df['half_of_day'] = df['film_date'].apply(get_half_of_day)
df['half_of_day'].value_counts()

# Looks like we only had 7 recordings done during the second half of the day!
# Since the year 2000, how have the number of comments changed year over year?

temp_df = df[df['year'] > 2000]
sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.barplot(x=temp_df['year'], y=temp_df['comments'],hue=temp_df['half_of_day'], estimator=np.sum)
# 2009 had the highest number of comments, followed by 2013. In general, the years 2009-2013 had a lot of comments.
sns.barplot(x=temp_df['month'], y=temp_df['comments'], hue=temp_df['half_of_day'], estimator=np.sum)
# Well, looks like Feb is the clear winner here. Jan and Aug are kinda bad for comments.
# What was the mean duration of these talks?

df['duration'].mean()
# sample rating string.

df['ratings'][0]
rating_names = set([])



# Method to read get a list of rating "names".

def split_ratings(ratings):

    val = ast.literal_eval(ratings)

    for rating in val:

        rating_names.add(rating['name'])
df['ratings'].apply(split_ratings)
# Let us take a look at the rating names.

rating_names
# Method to return count of each rating. Here, column name represents a rating name.

def get_count_from_ratings(ratings, col_name):

    val = ast.literal_eval(ratings)

    for rating in val:

        if rating['name'] == col_name:

            return rating['count']
for name in rating_names:

    df[name] = df['ratings'].apply(lambda rating : get_count_from_ratings(rating, col_name=name))
#  Let us take another look at the updated dataframe now.

df.head(5)
temp_df = df[df['year'] > 2000]
sns.barplot(x=temp_df['year'], y=temp_df['Fascinating'], estimator=np.sum)
sns.barplot(x=temp_df['year'], y=temp_df['Funny'], estimator=np.sum)
sns.barplot(x=temp_df['year'], y=temp_df['Obnoxious'], estimator=np.sum)
df.columns
pos_names = ['Beautiful', 'Fascinating', 'Jaw-dropping', 'Ingenious', 'Funny', 'Informative', 'Persuasive']

neg_names = ['Unconvincing', 'Obnoxious', 'Longwinded', 'Confusing']
def get_positive_count(row):

    pos_count=0

    for exp in pos_names:

        pos_count = pos_count + row[exp]

        

    return pos_count    
def get_negative_count(row):

    neg_count=0

    for exp in neg_names:

        neg_count = neg_count + row[exp]

        

    return neg_count    
df['negative_count'] = df.apply(get_negative_count, axis=1)

df['positive_count'] = df.apply(get_positive_count, axis=1)
temp_df = df[df['year'] > 2000]
sns.barplot(x=temp_df['year'], y=temp_df['positive_count'], estimator=np.sum)
# 2010 beats 2011 by a slight margin!
sns.barplot(x=temp_df['year'], y=temp_df['negative_count'], estimator=np.sum)
sns.barplot(x=temp_df['month'], y=temp_df['positive_count'], estimator=np.sum)
sns.barplot(x=temp_df['month'], y=temp_df['negative_count'], estimator=np.sum)