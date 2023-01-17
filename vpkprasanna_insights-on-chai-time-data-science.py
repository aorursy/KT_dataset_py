from IPython.display import Image

print("Welcome All to Chai time data science Show")

Image(url= "https://chaitimedatascience.com/content/images/2020/07/default-2.png")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import plotly.express as px

import plotly

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams

import warnings

warnings.filterwarnings('ignore')

import re

import string

import nltk

from nltk.corpus import stopwords

from collections import Counter

from wordcloud import WordCloud

from collections import Counter

import operator

import plotly.graph_objects as go

figg = go.Figure()

ctds_episodes = pd.read_csv('../input/chai-time-data-science/Episodes.csv',parse_dates=['recording_date','release_date'])

youtube_thumbnails = pd.read_csv('../input/chai-time-data-science/YouTube Thumbnail Types.csv')

anchor_thumbnails = pd.read_csv('../input/chai-time-data-science/Anchor Thumbnail Types.csv')

description = pd.read_csv('../input/chai-time-data-science/Description.csv')
youtube_thumbnails.head()
anchor_thumbnails.head()
length = len(ctds_episodes["episode_id"])

print(f"Total number of episodes Till Now {length} Episodes")
ctds_episodes.head()
birthday_episode = ctds_episodes[ctds_episodes["heroes"].isnull()]

birthday_episode
birthday = ctds_episodes[ctds_episodes["episode_id"]=="E69"]

birthday
ctds_episodes["heroes_gender"].value_counts()
gender = ctds_episodes["heroes_gender"].value_counts()

gender_df = pd.DataFrame({"gender":gender.index,"frequency":gender.values})

fig = px.bar(data_frame=gender_df,x="gender",y="frequency",title="Gender Distribution",color="gender",height=500,width=1000)

fig.show()
rcParams["figure.figsize"] = 20,10

sns.countplot(x = ctds_episodes["flavour_of_tea"],hue=ctds_episodes["flavour_of_tea"])
temp_data = Counter(ctds_episodes["flavour_of_tea"])

sorted_d = dict(sorted(temp_data.items(), key=operator.itemgetter(1),reverse=True))

fig = px.funnel_area(names = list(sorted_d.keys()),values = list(sorted_d.values()),title="Chai Consumed by Host duirng the interviews") 

fig.show()


#  <font size="+2" color="indigo"><b>Flavour of Chai VS ML heores</b></font><br><br>

# rcParams["figure.figsize"] = 20,10

# sns.countplot(x = ctds_episodes["flavour_of_tea"],hue=ctds_episodes["heroes_gender"])







#  <font size="+2" color="indigo"><b>falvour of Chai VS Category Vs ML Heores</b></font><br><br>

# fig = px.bar(data_frame=ctds_episodes,x ="flavour_of_tea",color="category",title="Favourite Chai based on there Category")

# fig.show()

## From the above image we can conclude that most of the ML Heores loves Masala chai and Ginger Chai very much 



# <font size="+2" color="indigo"><b>Flavour of chai VS Gender of ML Heroes</b></font><br><br>

# rcParams["figure.figsize"] = 25,10

# sns.countplot(x = ctds_episodes["heroes_gender"],hue=ctds_episodes["flavour_of_tea"])







#  <font size="+2" color="indigo"><b>Flavour of chai Baased on ML Heroes Nationality </b></font><br><br>

# fig = px.bar(data_frame=ctds_episodes,x ="heroes_nationality",color="flavour_of_tea",title = "Flavour of chai Based on ML Heroes Nationality")

# fig.show()

## People from USA used to like and have almost all the flavour of chai
time = ctds_episodes["recording_time"].value_counts()

time_df = pd.DataFrame({"time":time.index,"frequency":time.values})

fig = px.bar(data_frame=time_df,x="time",y="frequency",title="Recording Time Distribution",color="time")

fig.show()
fig = px.bar(data_frame=ctds_episodes,x ="flavour_of_tea",y="recording_time",color="recording_time",title = "Chai Consumed by Bhutani Based on Recording Time")

fig.show()
rcParams["figure.figsize"] = 15,10

sns.countplot(x = ctds_episodes["heroes_gender"],hue = ctds_episodes["category"])
subscriber = ctds_episodes.sort_values(by="youtube_subscribers",ascending=False)
rcParams["figure.figsize"] = 20,10

px.bar(x="heroes",y="youtube_subscribers",data_frame=subscriber[:10],title="Which ML Hero Helped more to gain high subscirber")
fig = px.line(data_frame=ctds_episodes,x ="episode_id",y="youtube_subscribers",title="Growth Rate of Subscriber based on each Episode")

fig.show()
labels = ctds_episodes['heroes_nationality'].value_counts()[:10].index

values = ctds_episodes['heroes_nationality'].value_counts()[:10].values

country = dict(ctds_episodes['heroes_nationality'].value_counts()[:10])

hover_values =list(country.keys()) 

fig = px.pie(values=values,labels=labels,title="Distribution of people from different locations",hover_name=hover_values)

fig.show()
fig = px.choropleth(ctds_episodes,locations="heroes_location",locationmode="country names",hover_name="heroes",title="ML heores Present Location")

fig.show()
fig = px.choropleth(ctds_episodes,locations="heroes_nationality",locationmode="country names",hover_name="heroes",title="ML Heroes Nationality")

fig.show()
sns.countplot(ctds_episodes["recording_time"],hue=ctds_episodes["category"])
fig = px.area(ctds_episodes,y = "episode_duration",x = "episode_id",title="Eposide Duration for each episode")

fig.show()
ctds_episodes["differenceDate"] = ctds_episodes["release_date"]-ctds_episodes["recording_date"] 
fig = px.line(ctds_episodes,y="differenceDate",title = "episode release data difference")

fig.show()
spotify_star = ctds_episodes.sort_values(by="spotify_starts",ascending=False)

px.bar(data_frame=spotify_star[:15],x="heroes",y="spotify_starts",title="ML heores Starts on spotify")
anchorplays = ctds_episodes.sort_values(by="anchor_plays",ascending=True)
px.bar(data_frame=anchorplays,x="anchor_plays",y="heroes",orientation="h",color="anchor_plays")
most_liked = ctds_episodes.sort_values(by="youtube_likes",ascending=True)
px.bar(data_frame=most_liked,x="youtube_likes",y="heroes",orientation="h",color="youtube_likes",title="MOST LIKED VIDEOS OF ML HEROS")
fig = px.histogram(data_frame=ctds_episodes,x="episode_duration",y="youtube_impression_views",histnorm="percent",title="Youtube Views VS Episode Duration")

fig.show()
fig = px.histogram(data_frame=ctds_episodes.sort_values(by="youtube_impressions",ascending=False),x="heroes",y="youtube_impressions",title="Youtube Impression VS ML Heores")

fig.show()
rcParams["figure.figsize"] = 20,7

ctds_episodes["youtube_dislikes"].value_counts().plot(kind="bar",title="Dislikes Count ")
fig = px.line(data_frame=ctds_episodes,x ="episode_id",y="youtube_ctr",line_group="category",color="category")

fig.show()
dislike = ctds_episodes[ctds_episodes["youtube_dislikes"]>1]

dislike["data_difference"] = dislike["release_date"]-dislike["recording_date"]
dislike["data_difference"]
ctr = ctds_episodes[["youtube_thumbnail_type","youtube_impressions","youtube_impression_views","youtube_ctr","youtube_nonimpression_views"]]
temp = ctr.groupby(["youtube_thumbnail_type"]).sum()
temp.head()
fig = go.Figure()

PLOT_BGCOLOR='#99ff66'

fig.add_trace(go.Indicator(

    title = 'YT Impression Thumbnail type 0',

    mode = "number",

    value = temp.youtube_impressions[0],

    domain = {'row': 0, 'column': 0}))



fig.add_trace(go.Indicator(

    title = "YT Impression Thumbnail type 1",

    mode = "number",

    value = temp.youtube_impressions[1],

    domain = {'row': 0, 'column': 1}))





fig.add_trace(go.Indicator(

    title = 'YT Impression Thumbnail type 2',

    mode = "number",

    value = temp.youtube_impressions[2],

    domain = {'row': 0, 'column': 2}))



fig.add_trace(go.Indicator(

    title = 'YT Impression Thumbnail type 3',

    mode = "number",

    value = temp.youtube_impressions[3],

    domain = {'row': 0, 'column': 3}))



# Row 2 Starts from here



fig.add_trace(go.Indicator(

    title = 'CTR Thumbnail type 0',

    mode = "number",

    value = temp.youtube_ctr[0],

    domain = {'row': 1, 'column': 0}))



fig.add_trace(go.Indicator(

    title = "CTR Thumbnail type 1",

    mode = "number",

    value = temp.youtube_ctr[1],

    domain = {'row': 1, 'column': 1}))





fig.add_trace(go.Indicator(

    title = 'CTR Thubmnail type 2',

    mode = "number",

    value = temp.youtube_ctr[2],

    domain = {'row': 1, 'column': 2}))



fig.add_trace(go.Indicator(

    title = 'CTR Thubnail type 3',

    mode = "number",

    value = temp.youtube_ctr[3],

    domain = {'row': 1, 'column': 3}))



fig.update_layout(title='<b>Click Through Rate  and Youtube Impression based on  Thumbnail Type</b>',

                  template='seaborn',

                  grid = {'rows': 2, 'columns': 4, 'pattern': "independent"},paper_bgcolor=PLOT_BGCOLOR)
tot_audio = ctds_episodes['anchor_plays'].sum()

spotify_total = ctds_episodes['spotify_starts'].sum()

apple_total = ctds_episodes['apple_listeners'].sum()
fig = px.pie(labels=['Spotify','Apple Podcast','Others'],

                             values=[spotify_total, apple_total,tot_audio-(spotify_total+apple_total)],hover_name=['Spotify','Apple Podcast','Others'],title = "Podcast Streaming Platforms")

fig.show()
fig = px.line(data_frame=ctds_episodes,x="release_date",y="anchor_plays")

fig.show()
fig = px.line(data_frame=ctds_episodes,x="release_date",y="spotify_streams")

fig.show()
fig = px.line(data_frame=ctds_episodes,x="release_date",y="spotify_listeners")

fig.show()
fig = px.line(data_frame=ctds_episodes,x="release_date",y="youtube_impressions",color="category")

fig.show()
fig = px.line(data_frame=ctds_episodes,x="episode_id",y="spotify_streams",line_group="category",color="category")

fig.show()
fig = px.line(data_frame=ctds_episodes,x="episode_id",y="spotify_listeners",line_group="category",color="category")

fig.show()
figg.add_trace(go.Scatter(x = ctds_episodes["episode_id"], y = ctds_episodes["spotify_starts"],name="spotify_starts"))

figg.add_trace(go.Scatter(x = ctds_episodes["episode_id"], y = ctds_episodes["spotify_streams"],name="spotify_streams"))

figg.add_trace(go.Scatter(x = ctds_episodes["episode_id"], y = ctds_episodes["spotify_listeners"], name="spotify_listeners"))

figg.update_layout(title='Overall Spotify Comparison for each Episode',

                   xaxis_title='Episode',

                   yaxis_title='Count')

figg.show()
fig = px.line(data_frame=ctds_episodes,x="episode_id",y="apple_listened_hours",line_group="category",color="category",title="Apple listened hours")

fig.show()


fig = px.line(data_frame=ctds_episodes,x="episode_id",y="apple_avg_listen_duration",line_group="category",color="category")

fig.show()
fig = px.line(data_frame=ctds_episodes,x="episode_id",y="apple_listeners",line_group="category",color="category")

fig.show()
description.head()
def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
description["description"] = description["description"].apply(clean_text)
def text_preprocessing(text):

    """

    Cleaning and parsing the text.



    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(remove_stopwords)

    return combined_text
description["description"] = description["description"].apply(text_preprocessing)
description.head()
description['temp_list'] = description['description'].apply(lambda x:str(x).split())

top = Counter([item for sublist in description['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')

fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Description', orientation='h', width=700, height=700,color='Common_words')

fig.show()
fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')

fig.show()
def generate_word_cloud(text):

    wordcloud = WordCloud(

        width = 3000,

        height = 2000,

        background_color = 'black').generate(str(text))

    fig = plt.figure(

        figsize = (40, 30),

        facecolor = 'k',

        edgecolor = 'k')

    plt.imshow(wordcloud, interpolation = 'bilinear')

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.show()
description_text = description.description[:100].values

generate_word_cloud(description_text)

most_viewed = ctds_episodes.sort_values(by="youtube_views",ascending=False)
E_27 = pd.read_csv("/kaggle/input/chai-time-data-science/Cleaned Subtitles/E27.csv")

E_27.head()
san = E_27[E_27["Speaker"]=="Sanyam Bhutani"]

jer = E_27[E_27["Speaker"]=="Jeremy Howard"]
jer["Text"] = jer["Text"].apply(text_preprocessing)

generate_word_cloud(jer.Text.values)