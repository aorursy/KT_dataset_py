# Importing dependencies
import pandas as pd
import bq_helper 
import matplotlib.pyplot as plt
%matplotlib inline
import wordcloud
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
#Import hackernews dataset using BigQueryHelper
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")
#Import bitcoin dataset
bitcoin= pd.read_csv('../input/bitcoin_price.csv', parse_dates=['date'])
# Main query
query = """SELECT title,score,time
            FROM `bigquery-public-data.hacker_news.full`
            WHERE REGEXP_CONTAINS(title, r"(b|B)itcoin") AND type = "story" 
            ORDER BY time
        """
# check how big this query will be
hacker_news.estimate_query_size(query) #Valor devuelto en GBytes
#Execute the query and show the first results
hn_query_result = hacker_news.query_to_pandas_safe(query)
hn_query_result.head()
#Transform date column into datetime object
bitcoin.date = pd.to_datetime(bitcoin.date,infer_datetime_format=True)
#Transform the same column into date object
bitcoin.date=bitcoin.date.dt.date
hacker_news_stories=hn_query_result
#Obtaining Pandas DateTime from epoch time
hacker_news_stories['datetime']=pd.to_datetime(hn_query_result.time, unit='s')
#Obtaining only date (without time) from datetime
hacker_news_stories['datetime']= hacker_news_stories['datetime'].dt.date
#Joining all the tittles
words = ' '.join(hn_query_result.title).lower()
cloud = wordcloud.WordCloud(background_color='black',
                            max_font_size=220,
                            width=1800,
                            height=1000,
                            max_words=200,
                            collocations=False,
                            relative_scaling=.5).generate(words)
plt.figure(figsize=(20,10))
plt.axis('off')
plt.imshow(cloud); 
# Graph Data from hacker news
hn_dates= hacker_news_stories['datetime'].value_counts().sort_index().index.values
hn_count=hacker_news_stories['datetime'].value_counts().sort_index().values
#Smooth averaging groups of 11 values
hn_count_mean_11=pd.Series(hn_count).rolling(11,center=True).mean()
# Graph Data from bitcoin price
btc_dates=bitcoin.date.values
btc_prices=bitcoin.price.values
plot_hacker_news = go.Scatter(
    x=hn_dates,
    y=hn_count_mean_11,
    name='Posts in HackerNews'
)
plot_bitcoin = go.Scatter(
    x=btc_dates,
    y=btc_prices,
    name='Bitcoin price',
    yaxis='y2'
)
data = [plot_hacker_news, plot_bitcoin]
layout = go.Layout(
    title='Post about bitcoin in hackernews VS Bitcoin Price',
    yaxis=dict(
        title='Number of posts'
    ),
    yaxis2=dict(
        title='Bitcoin price',
        titlefont=dict(
            color='rgb(0,0,0)'
        ),
        tickfont=dict(
            color='rgb(0, 103, 0)'
        ),
        overlaying='y',
        side='right'
    ),
     xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=3,
                     label='3m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                    label='1y',
                    step='year',
                    stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')

