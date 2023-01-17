import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import re

import calendar

import seaborn as sns

from matplotlib import pyplot as plt
history = pd.read_csv('/kaggle/input/netflix-data/NetflixViewingHistory.csv')

netflix_data = pd.read_csv('/kaggle/input/netflix-data/netflix_titles.csv')
dataset=history

sns.set(style="whitegrid")
dataset.head()
def title_segmentation(df, col, col2):

    index_col = df.columns.get_loc(col)

    index_col2 = df.columns.get_loc(col2)

    title=""

    for rows in range(len(df)):

        title = df.iat[rows, index_col]

        new_title=""

        season=""

        count=0

        for i in range(len(title)):

            if title[i]!=":":

                if count==0:

                    new_title = new_title+title[i]

                elif count==1:

                    season = season+title[i]

                else:

                    break

            else:

                count=count+1

                

        df.iat[rows, index_col] = new_title

        if season=="":

            season="N/A"

        df.iat[rows, index_col2] = season



dataset["Season"]=""

title_segmentation(dataset, 'Title', 'Season')
dataset.head()
dataset['Views_total']=1

g=dataset.groupby(['Title'])

df=g.sum()
drop=['Season','Date','Views_total']

dummy = dataset.drop(drop,axis=1)

dummy = pd.merge(df,dummy, on='Title', how='inner')

dummy = dummy.drop_duplicates()
print("Top 10 shows watched:")

dummy=dummy.sort_values(["Views_total"],ascending=False)

df = dummy.head(10)

df
dummy=dummy.sort_values(["Views_total"],ascending=False)

df = dummy.head(10)

fig, ax = plt.subplots(figsize=(10, 8))

chart = sns.barplot(x=df['Title'],y=df['Views_total'])

chart.set_xticklabels(labels=chart.get_xticklabels(),rotation=90, horizontalalignment='left')

plt.show()
dataset.drop('Views_total',axis=1,inplace=True)

dataset['Views_season']=1

g=dataset.groupby(['Title','Season'])

df=g.sum()
drop=['Date','Views_season']

dummy = dataset.drop(drop,axis=1)

on=['Title','Season']

dummy = pd.merge(df,dummy, on=on, how='inner')

dummy = dummy.drop_duplicates()
dummy=dummy.sort_values(["Views_season"],ascending=False)

df=dummy.head(10)
print("Top 10 shows watched according to season:")

df
dummy=dummy.sort_values(["Views_season"],ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))

chart=sns.barplot(x=df['Title'],y=df['Views_season'])

chart.set_xticklabels(labels=chart.get_xticklabels(),rotation=90, horizontalalignment='left')

plt.show()
dataset.drop('Views_season', axis=1, inplace=True)

dataset['Views_total']=1

g=history.groupby(['Title'])

df=g.sum()

dataset.drop('Views_total', axis=1, inplace=True)

dataset=pd.merge(dataset,df,on='Title',how='inner')
def date_segmentation(df, col, col2, col3):

    index_col = df.columns.get_loc(col)

    index_col2 = df.columns.get_loc(col2)

    index_col3 = df.columns.get_loc(col3)

    for rows in range(len(df)):

        date = df.iat[rows, index_col]

        m=""

        d=""

        count=0

        for i in range(len(date)):

            if date[i]!='/':

                if count==0:

                    d=d+date[i]

                elif count==1:

                    m=m+date[i]

                else:

                    break

            else:

                count=count+1

                if count==1:

                    df.iat[rows, index_col2]=int(d)

                else:

                    df.iat[rows, index_col3]=int(m)

        

        

dataset['Day']=0

dataset['Month']=0

date_segmentation(dataset, 'Date', 'Day', 'Month')
dataset.head()
dataset['Date']=pd.to_datetime(dataset['Date'])

dataset['Year'] = pd.DatetimeIndex(dataset['Date']).year
# Function to convert month number to month names



def month_converter(df, col, col2):

    index_col = df.columns.get_loc(col)

    index_col2 = df.columns.get_loc(col2)

    for rows in range(len(df)):

        month = df.iat[rows, index_col]

        

        if df.iat[rows,index_col]==1:

            df.iat[rows, index_col2]="January"

        if df.iat[rows,index_col]==2:

            df.iat[rows, index_col2]="February"

        if df.iat[rows,index_col]==3:

            df.iat[rows, index_col2]="March"

        if df.iat[rows,index_col]==4:

            df.iat[rows, index_col2]="April"

        if df.iat[rows,index_col]==5:

            df.iat[rows, index_col2]="May"

        if df.iat[rows,index_col]==6:

            df.iat[rows, index_col2]="June"

        if df.iat[rows,index_col]==7:

            df.iat[rows, index_col2]="July"

        if df.iat[rows,index_col]==8:

            df.iat[rows, index_col2]="August"

        if df.iat[rows,index_col]==9:

            df.iat[rows, index_col2]="September"

        if df.iat[rows,index_col]==10:

            df.iat[rows, index_col2]="October"

        if df.iat[rows,index_col]==11:

            df.iat[rows, index_col2]="November"

        if df.iat[rows,index_col]==12:

            df.iat[rows, index_col2]="December"



dataset['Month_Name']=""

month_converter(dataset, 'Month', 'Month_Name')
dataset.head()
# function to convert weekdays to weeknames



def day_converter(df,new_col,old_col):

    index_old = df.columns.get_loc(old_col)

    index_new = df.columns.get_loc(new_col)

    for row in range(0,len(df)):

        df.iat[row,index_new]=calendar.day_name[df.iat[row, index_old].weekday()]

        

dataset['Day_Name']=""

dataset['Date']=pd.to_datetime(history['Date'])

day_converter(dataset,'Day_Name','Date')
dataset.head()
df = dataset

drop = ['Date', 'Season', 'Views_total', 'Day',

       'Month', 'Month_Name', 'Day_Name']

df=df.drop(drop,axis=1)

df['Yearly Views'] = 1

g=df.groupby(['Year'])

df2=g.sum()
on=['Year']

df.drop('Yearly Views',axis=1,inplace=True)

dummy=pd.merge(df,df2,on=on, how='inner')

chart = sns.barplot(x=dummy['Year'], y=dummy['Yearly Views'])
df = dataset

drop = ['Date', 'Season', 'Views_total', 'Day',

       'Month', 'Day_Name']

df=df.drop(drop,axis=1)

df['Monthly Views'] = 1

g=df.groupby(['Year','Month_Name'])

df2=g.sum()
on=['Year','Month_Name']

df.drop('Monthly Views',axis=1,inplace=True)

dummy=pd.merge(df,df2,on=on, how='inner')

fig, ax = plt.subplots(figsize=(10, 8))

chart = sns.barplot(x=dummy['Month_Name'], y=dummy['Monthly Views'], hue=dummy['Year'])

chart.set_xticklabels(labels=chart.get_xticklabels(),rotation=90, horizontalalignment='left')

plt.show()
netflix_data.head()

netflix_data['Title']=netflix_data['title']
dataset=pd.merge(dataset,netflix_data,on='Title', how='inner')
movies = dataset[dataset['type']=="Movie"]
drop=['cast','title','show_id','description','type','Season','director','country','duration']

movies=movies.drop(drop, axis=1)
movies.head()
movie_details = pd.read_csv('/kaggle/input/netflix-data/IMDb movies.csv')
movie_details['Title']=movie_details['title']
movies=pd.merge(movies,movie_details,on='Title', how='inner')
drop=['title','original_title','imdb_title_id','reviews_from_users','reviews_from_critics','worlwide_gross_income','writer','usa_gross_income','budget', 'production_company']

movies=movies.drop(drop, axis=1)
movies.head()
# we will be extracting primary genre for visualisation purposes



def genre_segmentation(df, col):

    index_col = df.columns.get_loc(col)

    

    for row in range(len(df)):

        genre = df.iat[row,index_col]

        genre_new=""

        for i in range(len(genre)):

            if genre[i]!=',':

                genre_new=genre_new+genre[i]

            else:

                break

        

        df.iat[row,index_col]=genre_new

        

genre_segmentation(movies, 'genre')
drop=['Date','Month','Day','Year','Month_Name','Day_Name','year','duration','date_published']

dummy=movies.drop(drop,axis=1)

dummy = dummy.drop_duplicates()
dummy['Genre Views']=1

drop = ['avg_vote','Views_total','votes', 'metascore','release_year','rating']

g=dummy.drop(drop, axis=1).groupby(['genre'])

df=g.sum()
dummy.drop('Genre Views', axis=1, inplace=True)

dummy=pd.merge(dummy,df,on='genre',how='inner')
drop = ['Title', 'Views_total', 'date_added', 'release_year', 'rating',

       'listed_in','country', 'language', 'director', 'actors',

       'description', 'avg_vote', 'votes', 'metascore']

dummy2 = dummy.drop(drop, axis=1)
dummy2=dummy2.sort_values(["Genre Views"],ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))

chart = sns.barplot(x=dummy2['genre'],y=dummy2['Genre Views'])

chart.set_xticklabels(labels=chart.get_xticklabels(),rotation=90, horizontalalignment='left')

plt.show()