# pandas to read our csv file

import pandas as pd
# save the csv file into a dataframe 'df'

df = pd.read_csv('../input/elon-musk-tweets-2015-to-2020/elonmusk.csv',low_memory=False, parse_dates=[['date', 'time']])
# make a copy if you need so that the changes made in original df doesn't affect the copy

df_copy = df.copy(deep=True)
# check the whole df

display(df)



# check an overview of the df

display(df.info())



# gives out quick analysis, notice the max retweets_count and min retweets_count and so on

display(df.describe())
# I don't need these columns, so dropping them. You can keep them if you want.

drop_list = ['id','conversation_id','created_at','name','timezone','user_id','cashtags','place','quote_url','near','geo','source','user_rt_id','user_rt','retweet_id','retweet_date','translate','trans_src','trans_dest','video','retweet']

df = df.drop(columns=drop_list)
# have a look again.

display(df.info())
# just in case texthero cant remove URLs

df['tweet'] = df['tweet'].str.replace('http\S+|www.\S+', '',case=False)
df
# Check the above link for other installation instructions.



!pip install texthero
# import texthero



import texthero as hero
# let's do text preprocessing

from texthero import preprocessing



# creating a custom pipeline to preprocess the raw text we have

custom_pipeline = [preprocessing.fillna

                   , preprocessing.lowercase

                  #  , preprocessing.remove_digits # you can uncomment this if you want to remove digits as well.

                   , preprocessing.remove_punctuation

                   , preprocessing.remove_diacritics

                   , preprocessing.remove_stopwords

                   , preprocessing.remove_whitespace

                   , preprocessing.stem]



# simply call clean() method to clean the raw text in 'tweet' col and pass the custom_pipeline to pipeline argument

df['clean_tweet'] = hero.clean(df['tweet'], pipeline = custom_pipeline)
df
# Check the above link for other installation instructions



!pip3 install sweetviz
# importing sweetviz

import sweetviz as sv
# creating another dataframe df1 for further analysis.



df1 = df.drop(columns=['date_time'])
#to analyze the data and create a report, simply call analyze() method passing in the dataframe as argument



elonmusk_report = sv.analyze(df1)
#display the report as html



elonmusk_report.show_html('elonmusk.html')
!pip3 install pytz
from datetime import datetime

from pytz import timezone
# In place of 'UTC', replace it with whatever the current timezone is in your df.

# In place of 'Asia/Kolkata', replace it with whatever timezone you want to convert into.



df['conv_datetime'] = df['date_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
# I don't need the "+5.30" localize information in my df.



df['datetime'] = df['conv_datetime'].dt.tz_localize(None)
# dropping the extra columns and setting the datetime as index.



df = df.drop(columns=['date_time','conv_datetime'])
df = df.set_index('datetime')
df
df1 = df.drop(columns=['tweet','username','link'])
import matplotlib.pyplot as plt



# using top_words() method, get the top N words and make a bar plot.

hero.top_words(df1['clean_tweet']).head(10).plot.bar(figsize=(15,10))

plt.show()
# Want to add more stop words to your list? No problem. Follow the below steps.



from texthero import stopwords

default_stopwords = stopwords.DEFAULT

#add a list of stopwords to the stopwords

stop_w = ["twitter","pic","com","yes","like","year","need","ok","exact","come soon","yeah",

          "yup","would","much","use"]

custom_stopwords = default_stopwords.union(set(stop_w))

#Call remove_stopwords and pass the custom_stopwords list

df1['clean_tweet'] = hero.remove_stopwords(df1['clean_tweet'], custom_stopwords)
# Let's visualize again.



hero.top_words(df1['clean_tweet']).head(10).plot.bar(figsize=(15,10))

plt.show()
# just checking for any null values

df1.clean_tweet.isna().sum()
# WordCloud with single line of code.



hero.visualization.wordcloud(df1['clean_tweet'],width = 400, height= 400,background_color='White')
#Add pca value to dataframe to use as visualization coordinates

df1['pca'] = (

            df1['clean_tweet']

            .pipe(hero.tfidf)

            .pipe(hero.pca)

   )

#Add k-means cluster to dataframe 

df1['kmeans'] = (

            df1['clean_tweet']

            .pipe(hero.tfidf)

            .pipe(hero.kmeans, n_clusters=5)

   )

df1.head()
# Generate scatter plot for pca and kmeans. Cool isn't it?

hero.scatterplot(df1, 'pca', color = 'kmeans', hover_data=['clean_tweet'] )
!pip3 install chart-studio
import seaborn as sns # visualization library

import chart_studio.plotly as py # visualization library

from plotly.offline import init_notebook_mode, iplot # plotly offline mode

init_notebook_mode(connected=True) 

import plotly.graph_objs as go # plotly graphical object
df2 = df.drop(columns=['username','tweet','link'])
df2.head()
plt.figure(figsize=(17,10))

sns.lineplot(data=df2['retweets_count'], dashes=False)

plt.title("Retweets over time")

plt.show()
plt.figure(figsize=(17,10))

sns.lineplot(data=df2['replies_count'], dashes=False)

plt.title("Replies over time")

plt.show()
plt.figure(figsize=(17,10))

sns.lineplot(data=df2['likes_count'], dashes=False)

plt.title("Likes over time")

plt.show()