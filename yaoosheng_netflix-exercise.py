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
df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')



print(df.shape)
df.head()
for i in df.columns:

    null_rate = df[i].isna().sum() / len(df) * 100 

    if null_rate > 0 :

        print(f"{i}'s null rate : {null_rate}%")
df = df.fillna('NULL')

df['year_added'] = df['date_added'].apply(lambda x :  x.split(',')[-1])

df['year_added'] = df['year_added'].apply(lambda x : x if x != 'NULL' else '2020')

df['year_added'] = df['year_added'].apply(int)
import matplotlib.pyplot as plt

import seaborn as sns

import plotly 

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (13, 13)

stop_words = ["https", "co", "RT", "The"] + list(STOPWORDS) 



wordcloud = WordCloud(stopwords = stop_words,background_color = 'black', width = 1000,  height = 1000, max_words = 121).generate(' '.join(df['title']))



plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Words in Title',fontsize = 30)

plt.show()
df['type'].unique()
movie = df[df['type'] == 'Movie']

tv_show = df[df['type'] == 'TV Show']
col = "type"

grouped = df[col].value_counts().reset_index()

grouped = grouped.rename(columns = {col : "count", "index" : col})



fig = px.pie(grouped, values='count', names='type', color_discrete_sequence=px.colors.sequential.Bluered, title='Content types')

fig.show()
import matplotlib.patheffects as path_effects

year_data = df['year_added'].value_counts().sort_index().loc[:2019]

type_data = df.groupby('type')['year_added'].value_counts().sort_index().unstack().fillna(0).T.loc[:2019] 



fig, ax = plt.subplots(1,1, figsize=(28, 15))

ax.plot(year_data.index, year_data,  color="white", linewidth=5, label='Total', path_effects=[path_effects.SimpleLineShadow(),

                       path_effects.Normal()])

ax.plot(type_data.index, type_data['Movie'], color='skyblue', linewidth=5, label='Movie', path_effects=[path_effects.SimpleLineShadow(),

                       path_effects.Normal()])

ax.plot(type_data.index, type_data['TV Show'], color='salmon', linewidth=5, label='TV Show', path_effects=[path_effects.SimpleLineShadow(),

                       path_effects.Normal()])



ax.set_xlim(2006, 2020)

ax.set_ylim(-40, 2700)



t = [

    2008,

    2010.8,

    2012.1,

    2013.1,

    2015.7,

    2016.1,

    2016.9

]



events = [

    "Launch Streaming Video\n2007.1",

    "Expanding Streaming Service\nStarting with Candata | 2010.11",

    "Expanding to Europe\n2012.1",

    "First Original Content\n2013.2",

    "Expanding to Japan\n2015.9",

    "Original targeting Kids\n2016/1",

    "Offline Playback Features to all of Users\n2016/11"

]



up_down = [ 

    100,

    110,

    280,

    110,

    0,

    0,

    0

]



left_right = [

    -1,

    -0,

    -0,

    -0,

    -1,

    -1,

    -1.6,

    ]



for t_i, event_i, ud_i, lr_i in zip(t, events, up_down, left_right):

    ax.annotate(event_i,

                xy=(t_i + lr_i, year_data[int(t_i)] * (int(t_i+1)-t_i) + year_data[int(t_i)+1]  * (t_i-int(t_i)) + ud_i),

                xytext=(0,0), textcoords='offset points',

                va="center", ha="center",

                color="w", fontsize=16,

                bbox=dict(boxstyle='round4', pad=0.5, color='#303030', alpha=0.90))

    

    # A proportional expression to draw the middle of the year

    ax.scatter(t_i, year_data[int(t_i)] * (int(t_i+1)-t_i) + year_data[int(t_i)+1]  * (t_i-int(t_i)), color='#E50914', s=300)



ax.set_facecolor((0.4, 0.4, 0.4))

ax.set_title("Why Netflix's Conetents Count Soared?", position=(0.23, 1.0+0.03), fontsize=30, fontweight='bold')

ax.yaxis.set_tick_params(labelsize=20)

ax.xaxis.set_tick_params(labelsize=20)

plt.legend(loc='upper left', fontsize=20)



plt.show()
temp_df = df['rating'].value_counts().reset_index()





# create trace1

trace1 = go.Bar(

                x = temp_df['index'],

                y = temp_df['rating'],

                marker = dict(color = 'rgb(255,165,0)',

                              line=dict(color='rgb(0,0,0)',width=1.5)))

layout = go.Layout(template= "plotly_dark",title = 'MOST OF PROGRAMME ON NEYFLIX IS TV-14 & TV-MA RATED' , xaxis = dict(title = 'Rating'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()
fig = px.pie(temp_df, values='rating', names='index',

             title='RATING & CONTENT TYPE',)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
df1 = df[df["type"] == "TV Show"]

df2 = df[df["type"] == "Movie"]



temp_df1 = df1['rating'].value_counts().reset_index()

temp_df2 = df2['rating'].value_counts().reset_index()





# create trace1

trace1 = go.Bar(

                x = temp_df1['index'],

                y = temp_df1['rating'],

                name="TV Shows",

                marker = dict(color = 'rgb(249, 6, 6)',

                             line=dict(color='rgb(0,0,0)',width=1.5)))

# create trace2 

trace2 = go.Bar(

                x = temp_df2['index'],

                y = temp_df2['rating'],

                name = "Movies",

                marker = dict(color = 'rgb(26, 118, 255)',

                              line=dict(color='rgb(0,0,0)',width=1.5)))





layout = go.Layout(template= "plotly_dark",title = 'RATING BY CONTENT TYPE' , xaxis = dict(title = 'Rating'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1, trace2], layout = layout)

fig.show()
from collections import Counter

country_data = df['country']

country_counting = pd.Series(dict(Counter(','.join(country_data).replace(' ,',',').replace(', ',',').split(',')))).sort_values(ascending=False)

country_counting.drop(['NULL'], axis=0, inplace=True)
tot = sum(country_counting)

top20 = sum(country_counting[:20]) # 22 is real 20% but for simple processing



print(f'total : {tot}')

print(f'top 20 countries : {top20}')

print(f'percentage : {top20}/{tot} = {100 * top20/tot}')
top20_country = country_counting[:20]


fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'domain'}]])

fig.add_trace(go.Bar(x = top20_country.index, y = top20_country),row=1, col=1)

fig.add_trace(go.Pie(labels = top20_country.index, values = top20_country),row=1, col=2)



fig.update_layout(title='Top 20 producing countries', xaxis_title = "Countries",

    yaxis_title = "amount")

fig.show()
df['country'] = df['country'].dropna().apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(','))

lst_col = 'country'

data2 = pd.DataFrame({

      col :  np.repeat(df[col].values, df[lst_col].str.len())

      for col in df.columns.drop(lst_col)}

    ).assign(**{lst_col:np.concatenate(df[lst_col].values)})[df.columns.tolist()]

year_country2 = data2.groupby('year_added')['country'].value_counts().reset_index(name='counts')



fig = px.choropleth(year_country2, locations="country", color="counts", 

                    locationmode='country names',

                    animation_frame='year_added',

                    range_color=[0,80],

                    color_continuous_scale=px.colors.sequential.Plasma

                   )



fig.update_layout(title='Comparison by country')

fig.show()
from sklearn.preprocessing import MultiLabelBinarizer # Similar to One-Hot Encoding



def relation_heatmap(df, title):

    df['genre'] = df['listed_in'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 

    Types = []

    for i in df['genre']: Types += i

    Types = set(Types)

    print(f"There are {len(Types)} types in the Netflix {title} Dataset")    

    test = df['genre']

    mlb = MultiLabelBinarizer()

    res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_, index=test.index)

    corr = res.corr()

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(15, 14))

    pl = sns.heatmap(corr, mask=mask, cmap= "coolwarm", vmax=.5, vmin=-.5, center=0, square=True, linewidths=.7, cbar_kws={"shrink": 0.6})

    

    plt.show()
relation_heatmap(movie, 'Movie')
relation_heatmap(tv_show, 'TV Show')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (13, 13)

stop_words = ["https", "co", "RT", "The"] + list(STOPWORDS) 



wordcloud = WordCloud(stopwords = stop_words,background_color = 'black', width = 1000,  height = 1000, max_words = 121).generate(' '.join(df['description']))



plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Words in description',fontsize = 30)

plt.show()

us_series_data=tv_show[tv_show['country']=='United States']
oldest_us_series=us_series_data.sort_values(by='release_year')[0:20]
fig = go.Figure(data=[go.Table(header=dict(values=['Title', 'Release Year'],fill_color='paleturquoise'),

                 cells=dict(values=[oldest_us_series['title'],oldest_us_series['release_year']],fill_color='pink'))

                     ])

fig.show()
newest_us_series=us_series_data.sort_values(by='release_year', ascending=False)[0:50]
fig = go.Figure(data=[go.Table(header=dict(values=['Title', 'Release Year'],fill_color='yellow'),

                 cells=dict(values=[newest_us_series['title'],newest_us_series['release_year']],fill_color='lavender'))

                     ])

fig.show()