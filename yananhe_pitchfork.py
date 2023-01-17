import sqlite3, datetime

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.ticker import MaxNLocator



# For Interactive control

import ipywidgets as widgets

from ipywidgets import interact, interact_manual

import cufflinks as cf



# For Regression

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures 
# Import all data

with sqlite3.connect('../input/pitchfork-data/database.sqlite') as conn:

    artists = pd.read_sql('SELECT * FROM artists', conn)

    content = pd.read_sql('SELECT * FROM content', conn)

    genres = pd.read_sql('SELECT * FROM genres', conn)

    labels = pd.read_sql('SELECT * FROM labels', conn)

    reviews = pd.read_sql('SELECT * FROM reviews', conn)

    years = pd.read_sql('SELECT * FROM years', conn)
# # For presentation, hide code

# from IPython.display import HTML



# HTML('''<script>

# code_show=true; 

# function code_toggle() {

#  if (code_show){

#  $('div.input').hide();

#  } else {

#  $('div.input').show();

#  }

#  code_show = !code_show

# } 

# $( document ).ready(code_toggle);

# </script>

# <form action="javascript:code_toggle()"><input type="submit" value="Hide/Show raw code."></form>''') 
df_list = [artists, content, genres, labels, reviews, years]
for df in df_list:

    display(df.head())

    display(df.info())
# Check if there are duplicates reviewid: Yes

for df in df_list:

    print(df['reviewid'].nunique())
# Create datetime column for YYYY-mm

reviews['date'] = pd.to_datetime(reviews['pub_date'])

reviews['year_month'] = reviews['date'].dt.strftime('%Y-%m')
# Number of reviews over year

df_year = reviews.groupby('pub_year')['reviewid'].nunique()

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

df_year.plot.bar()
# Histogram of scores

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

reviews['score'].hist(ax=ax, bins=50, edgecolor='white', grid=False)
reviews['score'].describe()
df2 = reviews.groupby(['artist','year_month'], as_index=False).agg({'score': np.mean, 'reviewid':'nunique'})
@interact

def ind_artist(artist = df2['artist'].unique()):

    

    df2_artist = df2[df2['artist']==artist]

    df2_artist = df2_artist.set_index('year_month')



    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))



    # Plot monthly average score

    df2_artist['score'].plot.bar(ax=ax1, rot=0, color='silver')

    ax1.set_xlabel('')

    ax1.set_ylabel('Average Score')

    ax1.set_title('Average Review Score of Individual Artist Across Time')



    # Add bar values

    for p in ax1.patches:

        ax1.annotate('{:,.1f}'.format(np.round(p.get_height(),decimals=4)), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')



    # Remove the frame

    for spine in ax1.spines.values():

        spine.set_visible(False)



    # Plot monthly review numbers

    ax2 = ax1.twinx()

    ax2.plot(df2_artist['reviewid'], color='darkred', linewidth=2, marker='o', markersize=7, markeredgecolor='w')

    ax2.set_ylabel('Number of reviews', color='darkred')



    # Remove the frame

    for spine in ax2.spines.values():

        spine.set_visible(False)



    # Force y-axis ticks integer

    max_review = df2_artist['reviewid'].max()

    ax2.set_ylim([0,max_review+1])

    ax2.yaxis.set_major_locator(MaxNLocator(integer=True)) 
df3 = reviews.groupby(['artist'], as_index=False).agg({'score': np.mean, 'reviewid':'nunique'})

df3 = df3.rename(columns={'reviewid':'reviews'})
df3.head()
# Remove outliers

df3 = df3[df3['reviews']<=df3['reviews'].quantile(0.99)]



fig, ax = plt.subplots(1, 1, figsize=(12, 6)) 

ax = sns.violinplot(x=df3['reviews'], y=df3['score'])
# Create a bucket of 0.5 points for average scores

df3['score_g'] = df3['score'].apply(lambda x: np.ceil(x/0.5)*0.5)

df3.head()
# Plot scatter plot average score for an artist and number of reviews

fig, ax = plt.subplots(1, 1, figsize=(8, 6)) 

ax.scatter(x=df3['reviews'], y=df3['score_g'], s=10, color = 'b', alpha=0.2)   

ax.set_xlabel('Number of reviews for an artist')

ax.set_ylabel('Average score')

ax.set_title('Correlation between average score and number of reviews for an artist')

print('Correlation between average score and number of reviews for an artist: ')

print(df3[['reviews','score']].corr())



# Fit square regression model

x = df3['reviews'].values.reshape(-1,1)

y = df3['score_g'].values.reshape(-1,1)

model = PolynomialFeatures(degree=1) 

x_model = model.fit_transform(x) 

model.fit(x_model, y) 

model1 = LinearRegression() 

model1.fit(x_model, y)



# Plotting

x_range = np.arange(df3['reviews'].min(),df3['reviews'].max()+1,1).reshape(-1,1)

ax.plot(x_range, model1.predict(model.fit_transform(x_range)), color = 'r', linewidth=1, zorder=1, label='ZFS') 