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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from nltk.sentiment import SentimentAnalyzer

from wordcloud import WordCloud,STOPWORDS
df = pd.read_csv("../input/covid19-tweets/covid19_tweets.csv")

df.head()
df.describe()
df.shape
df.info()
def missing_data(data):

    total = data.isnull().sum()

    percent =((data.isnull().sum()/data.isnull().count())*100)

    ms_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])

    d_type = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        d_type.append(dtype)

    ms_data['data type'] = d_type

    return (np.transpose(ms_data))
missing_data(df)
df['user_location'].unique()
df['user_location'].value_counts()
df['user_description'].unique()
df['user_location'] = df['user_location'].fillna(df['user_location'].mode()[0])
missing_data(df)
df['hashtags'].value_counts()
df['hashtags'] = df['hashtags'].fillna(df['hashtags'].mode()[0])
df.dropna(axis=0,inplace=True)
df.shape
missing_data(df)
df['datet'] = pd.to_datetime(df['date'])
df['year'] = df['datet'].dt.year

df['month'] = df['datet'].dt.month

df['day'] = df['datet'].dt.day

df['date_only'] = df['datet'].dt.date
df.head()
plt.rcParams['figure.figsize'] = (12,9)

sns.barplot(x='user_followers',y='user_name',data = df.sort_values('user_followers',ascending=False)[:200])

plt.show()
plt.rcParams['figure.figsize'] = (12,5)

sns.countplot(x=df['user_location'],order=df['user_location'].value_counts().index[:20],palette='Set3')

plt.xticks(rotation=90)

plt.title("Number of User Locations")

plt.show()
plt.rcParams['figure.figsize'] = (12,9)

sns.countplot(x=df['hashtags'],order = df['hashtags'].value_counts().index[:5],data = df)

plt.xticks(rotation=90)

plt.show()
plt.rcParams['figure.figsize'] = (12,5)

sns.countplot(x=df['hashtags'],order = df['hashtags'].value_counts().index[:5],hue=df['user_verified'])

plt.xticks(rotation=90,size=10)

plt.show()
plt.rcParams['figure.figsize'] = (12,9)

sns.countplot(x=df['user_location'],order=df['user_location'].value_counts().index[:10],hue=df['user_verified'],palette='Accent')

plt.xticks(rotation=90)

plt.show()
tweet_df = df.groupby(['date_only'])['text'].count().reset_index()

tweet_df.columns = ["date_only","count"]
def plot_time_variation(df, x='date_only', y='count', hue=None, size=1, title=""):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    g = sns.lineplot(x=x, y=y, hue=hue, data=df)

    plt.xticks(rotation=90)

    if hue:

        plt.title(f'{y} grouped by {hue} | {title}')

    else:

        plt.title(f'{y} | {title}')

    ax.grid(color='black', linestyle='dotted', linewidth=0.75)

    plt.show()
plot_time_variation(tweet_df,title="Number of tweets / day",size=3)
stopwords = set(STOPWORDS)

def display_wordcloud(data,title):

    wordcloud = WordCloud(

    background_color = "white",

    stopwords=stopwords,

    max_words = 50,

    max_font_size=40,

    scale = 5,

    random_state=1,

    ).generate(str(data))

    

    fig = plt.figure(1,figsize=(12,10))

    plt.axis("off")

    plt.suptitle(title,fontsize=20)

    plt.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)

    plt.show()
india_tweets = df.loc[df.user_location=="India"]

display_wordcloud(india_tweets['text'],"Prevalent words in tweets from India")
united_st_df = df.loc[df.user_location == "United States"]

display_wordcloud(united_st_df['text'],"Prevalent words in tweets from United States")