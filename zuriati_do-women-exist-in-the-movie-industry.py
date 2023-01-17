import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import json

import datetime as dt

import calendar

from collections import defaultdict

from wordcloud import WordCloud

from PIL import Image

%matplotlib inline

sns.set()
## Functions



def load_file(filename, lst):

    return pd.read_csv(filename, converters={x:json.loads for x in lst})



def extract_crew(df):

    new_df = pd.DataFrame()

    for row in df.itertuples():

        movie_id = row.movie_id

        tmp = pd.DataFrame(row.crew[0:30])

        tmp['movie_id'] = movie_id

        new_df = pd.concat([new_df, tmp], sort=False)

        new_df['gender'] = new_df['gender'].astype('int')

    return new_df



def extract_cast(df):

    new_df = pd.DataFrame()

    for row in df.itertuples():

        movie_id = row.movie_id

        tmp = pd.DataFrame(row.cast[0:30])

        tmp['movie_id'] = movie_id

        new_df = pd.concat([new_df, tmp], sort=False)

        new_df['gender'] = new_df['gender'].astype('int')

    return new_df



def prepare_data(df1, df2):

    df_gender = pd.DataFrame()

    tmp1 = pd.DataFrame(df1.groupby(['gender']).count()['name'])

    tmp1['type'] = 'crew'

    tmp2 = pd.DataFrame(df1.groupby(['gender']).count()['name'])

    tmp2['type'] = 'cast'

    df_gender = pd.concat([tmp1, tmp2], axis=0)

    return df_gender.reset_index()



def plot_graph(df, title):

    df['color'] = df.index.get_level_values(level='gender')

    df['color'] = df.color.apply(lambda x:'deeppink' if x == 1 else 'dodgerblue')



    #setting position and width for the bars

    plt.figure(figsize=(12,18))

    y_ticks = np.arange(30)

    plt.barh(y_ticks,df['movie_id'],color=df['color'])

    plt.yticks(y_ticks,df.index.get_level_values(level='name'),fontsize=12)

    plt.title(title,fontsize=14)

    plt.grid(False)

    plt.gca().invert_yaxis()

    plt.show()

    

def extract_titles(df, title_lst):

    new_df = pd.DataFrame()

    for title in title_lst:

        tmp = df[df.job == title]

        new_df = pd.concat([new_df, tmp], axis=0)

    return new_df



def preprocess_director_data(df_jobs, df_movies):

    df_merged = pd.merge(df_jobs, df_movies, left_on='movie_id', right_on='id', how='left')

    wm_director = df_merged[(df_merged.job == 'Director') & (df_merged.gender == 1)]

    m_director = df_merged[(df_merged.job == 'Director') & (df_merged.gender == 2)]

    tmp = m_director.groupby('Yr').count()[['name']]

    tmp_w = wm_director.groupby('Yr').count()[['name']]

    df = pd.concat([tmp, tmp_w], axis=1, sort=True)

    df.reset_index(inplace=True)

    df.columns = ['Yr','men', 'women']

    return df.fillna(0)
# load credits datasets

df_credits = load_file('../input/tmdb-movie-metadata/tmdb_5000_credits.csv', lst=['cast', 'crew'])
df_cast = extract_cast(df_credits)

df_crew = extract_crew(df_credits)

df_gender = prepare_data(df_crew, df_cast)
plt.figure(figsize=(8,6))

palette = ['oldlace','deeppink','dodgerblue']

labels=['Missing', 'Female', 'Male']

g = sns.barplot(x='type', hue='gender', y='name', data=df_gender, palette=palette)

g.legend(labels)

plt.xlabel('')

plt.ylabel('')

plt.title('Gender Distribution of Members')

plt.show(g)
#let's look at top 30 crew who's worked on the most number of movies.

gb_crew = df_crew.groupby(['name','gender']).agg({'movie_id':'count','job':'count'}).sort_values(by='movie_id',ascending=False).head(30)

plot_graph(gb_crew, 'Crew worked on most movies')
gb_cast= df_cast.groupby(['name','gender']).agg({'movie_id':'count'}).sort_values(by='movie_id',ascending=False).head(30)

plot_graph(gb_cast, 'Number of movies made by actors/actressess')
#let's start by extracting the job titles as stipulated above.

titles = ['Producer','Executive Producer','Director','Director of Photography','Costume Design','Editor','Makeup Artist',

        'Original Music Composer','Art Direction','Supervising Sound Editor','Sound Re-Recording Mixer',

        'Visual Effects Supervisor','Visual Effects Producer','Screenplay','Writer']



df_jobs = extract_titles(df_crew, titles)
gb_jobs = df_jobs.groupby(['gender','job']).count()



#setting position and width for the bars

pos = np.arange(15)

width = 0.3



#plotting the bars

fix,ax = plt.subplots(figsize=(12,8))



#create bar with female values ie 1

plt.bar(pos,gb_jobs.xs((1),level='gender')['name'],width,color='deeppink')

plt.bar([p + width for p in pos],gb_jobs.xs((2),level='gender')['name'],width,color='dodgerblue')



ax.set_xticks([p + width for p in pos])

ax.set_xticklabels(titles,rotation=90)

ax.grid(False)

plt.legend(['Female','Male'],loc='upper left',fontsize=12)

plt.title('Job Distribution by Gender', fontsize=14)

plt.show()
df_movies = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
# preprocess the movies dataset

# 1) convert release_date to datetime format to show the evolution overtime.



df_movies['release_date'] = pd.to_datetime(df_movies['release_date'])

df_movies['Yr'] = df_movies.release_date.dt.strftime('%Y')
df_director = preprocess_director_data(df_jobs, df_movies)
plt.figure(figsize=(12,10))

plt.plot(df_director['Yr'], df_director['men'],color='deeppink',label='Women')

plt.plot(df_director['Yr'], df_director['women'],color='dodgerblue',label='Men')

plt.title('Men vs Women Directors over the years', fontsize=14)

plt.grid(False)

plt.xticks(rotation=90)

plt.legend(fontsize=12)

plt.show()
dates = [df_director.Yr.min(),df_director[df_director.women != 0.0].iloc[0]['Yr']]

texts = ['1st movie released','1st movie directed by woman']



fig, ax = plt.subplots(figsize=(12,1))

ax.plot((dates[0],dates[1]),(0,0),'k',alpha=0.3)



for i, (text,date) in enumerate(zip(texts,dates)):

    ax.scatter(date,0,marker='s', s=100,color='crimson')

    ax.text(date,0.01,text,rotation=45,va="bottom",fontsize=14)



ax.yaxis.set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['top'].set_visible(False)

ax.xaxis.set_ticks_position('bottom')

ax.grid(False)

ax.patch.set_facecolor('white')



ax.get_yaxis().set_ticklabels([])

plt.show()
df_final = pd.merge(df_jobs, df_movies, left_on='movie_id', right_on='id', how='left')

df_budget = df_final.groupby('gender').agg({'budget':'sum','revenue':'sum','job':'count'}).reset_index()
plt.figure(figsize=(8,6))

palette = ['oldlace','deeppink','dodgerblue']

labels=['Missing', 'Female', 'Male']

g = sns.barplot(x='gender',y='budget', data=df_budget, palette=palette)

g.legend(labels)

plt.xlabel('')

plt.ylabel('')

plt.title('Gender Distribution of Members')

plt.show(g)
df_budget
#setting position and width for the bars

pos = np.arange(3)

width = 0.4



#plotting the bars

fix,ax = plt.subplots(figsize=(8,6))



bar_b = ax.bar(pos,df_budget.budget,width,color='deeppink')

bar_r = ax.bar([p + width for p in pos],df_budget.revenue,width,color='dodgerblue')



ax.set_title('Budget vs Revenue by Gender',fontsize=14)

ax.set_xticks([p + 0.5*width for p in pos])



#ax.set_yticks([p + 0.5*width for p in pos])

#ax.set_yticklabels(df_budget,fontsize=12)

ax.set_xticklabels(['Unknown', 'Female', 'Male'])

ax.grid(False)    

ax.patch.set_facecolor('white')

plt.legend(['Budget','Revenue'],loc='best',fontsize=12)

plt.show()