import pandas as pd
import numpy as np

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
import plotly.express as px

import cufflinks as cf
cf.go_offline(connected=True)
init_notebook_mode(connected=True)
cf.set_config_file(theme='polar')
filename = "/kaggle/input/beer-reviews-cleaned/beer_reviews_clean.csv"
df = pd.read_csv(filename)
df.head()
print('Number of unique breweries:', df.brewery_name.nunique())
print('Number of unique beer styles:', df.beer_style.nunique())
print('Number of beers reviewed:', df.beer_name.nunique())
x0 = df.review_appearance
x1 = df.review_aroma
x2 = df.review_overall
x3 = df.review_palate
x4 = df.review_taste

fig = go.Figure()
fig.add_trace(go.Histogram(x=x0, name='Review Appearance'))
fig.add_trace(go.Histogram(x=x1,  name='Review Aroma'))
fig.add_trace(go.Histogram(x=x2, name='Review Overall'))
fig.add_trace(go.Histogram(x=x3, name='Review Palate'))
fig.add_trace(go.Histogram(x=x4, name="Review Taste"))

fig.update_traces(opacity=0.75)
fig.update_layout(title_text = 'Distribution of Reviews',
                 xaxis_title_text='Review Score',
                 yaxis_title_text='Count',
                 bargap=0.2)
fig.show()
dfhigh = df.loc[df['review_overall'] >= 3]
dflow= df.loc[df['review_overall'] <= 2.5]
x0 = dfhigh.review_appearance
x1 = dfhigh.review_aroma
x2 = dfhigh.review_overall
x3 = dfhigh.review_palate
x4 = dfhigh.review_taste

fig = go.Figure()
fig.add_trace(go.Histogram(x=x0, name='Review Appearance'))
fig.add_trace(go.Histogram(x=x1,  name='Review Aroma'))
fig.add_trace(go.Histogram(x=x2, name='Review Overall'))
fig.add_trace(go.Histogram(x=x3, name='Review Palate'))
fig.add_trace(go.Histogram(x=x4, name="Review Taste"))

# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.update_layout(title_text = 'Distribution of Reviews - Highly Rated',
                 xaxis_title_text='Review Score',
                 yaxis_title_text='Count',
                 bargap=0.2)
fig.show()
x0 = dflow.review_appearance
x1 = dflow.review_aroma
x2 = dflow.review_overall
x3 = dflow.review_palate
x4 = dflow.review_taste

fig = go.Figure()
fig.add_trace(go.Histogram(x=x0, name='Review Appearance'))
fig.add_trace(go.Histogram(x=x1,  name='Review Aroma'))
fig.add_trace(go.Histogram(x=x2, name='Review Overall'))
fig.add_trace(go.Histogram(x=x3, name='Review Palate'))
fig.add_trace(go.Histogram(x=x4, name="Review Taste"))

# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.update_layout(title_text = 'Distribution of Reviews - Low Ratings',
                 xaxis_title_text='Review Score',
                 yaxis_title_text='Count',
                 bargap=0.2)
fig.show()
x = df['beer_abv'].sort_values(ascending=False)

fig= go.Figure()
fig.add_trace(go.Histogram(x=x/100, nbinsx=100, histfunc="count", name='count'))

fig.update_layout(
    title_text='Beer ABV',
    bargap=0.2,
    xaxis=dict(title="Percent Alcohol by Volume",
            tickformat = "%",
            hoverformat = '.2%'),
    yaxis_title='Count')
fig.show()
corr = df[['review_overall', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'beer_abv']].corr()
fig = go.Figure()

fig.add_trace(go.Heatmap(
    z=corr.values,
    x=list(corr.columns),
    y=list(corr.index),
    colorscale='blues'
    ))

fig.show()
df.beer_style.value_counts().head(10).iplot(kind='barh', title='10 Most Reviewed Beer Styles')
df.beer_style.value_counts().tail(10).iplot(kind='barh', title='10 Least Reviewed Beer Styles')
y = df['beer_name'].value_counts().sort_values(ascending=False).head(10).iplot(kind='barh', title='10 Most Reviewed Beers')
df.brewery_name.value_counts().head(10).iplot(kind='barh', title='Top 10 Most Reviewed Breweries')
df.brewery_name.value_counts().tail(10).iplot(kind='barh', title='Least Reviewed Breweries')
top20abv = df[['beer_name', 'brewery_name', 'beer_abv', 'beer_style']].sort_values('beer_abv', ascending=False).drop_duplicates('beer_name').head(10)

p = [go.Bar(x = top20abv['beer_abv'] / 100,
            y = top20abv['beer_name'],
            hoverinfo = 'x',
            text=top20abv['brewery_name'],
            textposition = 'inside',
            orientation='h',
            opacity=0.7, 
            marker=dict(
                color='rgb(1, 87, 155)'
            ))]

layout = go.Layout(title='Top 10 Strongest Beers by ABV',
                   xaxis=dict(title="ABV",
                              tickformat = "%",
                              hoverformat = '.2%'),
                   margin = dict(l = 220))

fig = go.Figure(data=p, layout=layout)

py.offline.iplot(fig)
btm20abv = df[['beer_name', 'brewery_name', 'beer_abv', 'beer_style']].sort_values('beer_abv', ascending=False).drop_duplicates('beer_name').tail(10)

p = [go.Bar(x = btm20abv['beer_abv'] / 100,
            y = btm20abv['beer_name'],
            hoverinfo = 'x',
            text=btm20abv['brewery_name'],
            textposition = 'inside',
            orientation='h',
            opacity=0.7, 
            marker=dict(
                color='rgb(1, 87, 155)'
            ))]

layout = go.Layout(title='Top 10 Weakest Beers by ABV',
                   xaxis=dict(title="ABV",
                              tickformat = "%",
                              hoverformat = '.2%'),
                   margin = dict(l = 220))

fig = go.Figure(data=p, layout=layout)

py.offline.iplot(fig)
df['review_average'] = df.apply(lambda x: (x.review_overall + x.review_aroma + x.review_appearance + x.review_palate + x.review_taste) / 5, axis=1)
df['total_reviews'] = 0

beers_grouped = df.groupby(['beer_beerid']).agg(dict(beer_name='first', brewery_name='first', beer_style = 'first', total_reviews='count', review_appearance='mean', review_overall='median', review_taste='mean', review_aroma='mean', review_average='mean', review_palate='mean')).reset_index()
beers_grouped.head()
beers_grouped.describe()
top_reviews = beers_grouped.loc[beers_grouped['total_reviews'] >= 100]
top_reviews.head()
top_beers = top_reviews.sort_values('review_average',ascending=False).head(15)
btm_beers = top_reviews.sort_values('review_average',ascending=False).tail(15)
x = top_beers['review_average']
y = top_beers['beer_name']

p = [go.Bar(x = x,
            y = y,
            hoverinfo = 'x',
            text=top_beers['brewery_name'],
            textposition = 'inside',
            orientation='h',
            opacity=0.7, 
            marker=dict(
                color='rgb(1, 87, 155)'
            ))]

layout = go.Layout(title='Top 15 Beers by Review Average',
                   xaxis=dict(title="Review Average"),
                   margin = dict(l = 220))

fig = go.Figure(data=p, layout=layout)

py.offline.iplot(fig)
x = btm_beers['review_average']
y = btm_beers['beer_name']

p = [go.Bar(x = x,
            y = y,
            hoverinfo = 'x',
            text=btm_beers['brewery_name'],
            textposition = 'inside',
            orientation='h',
            opacity=0.7, 
            marker=dict(
                color='rgb(1, 87, 155)'
            ))]

layout = go.Layout(title='Bottom 15 Beers by Review Average',
                   xaxis=dict(title="Review Average"),
                   margin = dict(l = 220))

fig = go.Figure(data=p, layout=layout)

py.offline.iplot(fig)
top_breweries = top_reviews.groupby('brewery_name').agg(dict(brewery_name='first', review_average='mean')).sort_values('review_average', ascending=False)
top_brew = top_breweries.head(15)
btm_brew = top_breweries.tail(15)
x = top_brew['review_average']
y = top_brew['brewery_name']

p = [go.Bar(x = x,
            y = y,
            hoverinfo = 'x',
            text=top_brew['review_average'],
            textposition = 'inside',
            orientation='h',
            opacity=0.7, 
            marker=dict(
                color='rgb(1, 87, 155)'
            ))]

layout = go.Layout(title='Top 15 Breweries by Review Average',
                   xaxis=dict(title="Review Average"),
                   margin = dict(l = 220))

fig = go.Figure(data=p, layout=layout)

py.offline.iplot(fig)
x = btm_brew['review_average']
y = btm_brew['brewery_name']

p = [go.Bar(x = x,
            y = y,
            hoverinfo = 'x',
            text=btm_brew['review_average'],
            textposition = 'inside',
            orientation='h',
            opacity=0.7, 
            marker=dict(
                color='rgb(1, 87, 155)'
            ))]

layout = go.Layout(title='Bottom 15 Breweries by Review Average',
                   xaxis=dict(title="Review Average"),
                   margin = dict(l = 220))

fig = go.Figure(data=p, layout=layout)

py.offline.iplot(fig)
top_styles = top_reviews.groupby('beer_style').agg(dict(beer_style='first', review_average='mean')).sort_values('review_average', ascending=False)
top_style = top_styles.head(15)
btm_style = top_styles.tail(15)
x = top_style['review_average']
y = top_style['beer_style']

p = [go.Bar(x = x,
            y = y,
            hoverinfo = 'x',
            text=top_style['review_average'],
            textposition = 'inside',
            orientation='h',
            opacity=0.7, 
            marker=dict(
                color='rgb(1, 87, 155)'
            ))]

layout = go.Layout(title='Top 15 Styles by Review Average',
                   xaxis=dict(title="Review Average"),
                   margin = dict(l = 220))

fig = go.Figure(data=p, layout=layout)

py.offline.iplot(fig)
x = btm_style['review_average']
y = btm_style['beer_style']

p = [go.Bar(x = x,
            y = y,
            hoverinfo = 'x',
            text=btm_style['review_average'],
            textposition = 'inside',
            orientation='h',
            opacity=0.7, 
            marker=dict(
                color='rgb(1, 87, 155)'
            ))]

layout = go.Layout(title='Bottom 15 Breweries by Review Average',
                   xaxis=dict(title="Review Average"),
                   margin = dict(l = 220))

fig = go.Figure(data=p, layout=layout)

py.offline.iplot(fig)
df.review_profilename.value_counts().head(10)
fig = go.Figure()
fig.add_trace(go.Box(y=df.review_profilename.value_counts(), boxmean='sd'))
fig.update_layout(title='Distribution of Reviews per User')
fig.show()
df.review_time = pd.to_datetime(df['review_time'])
group_by_date = df[['review_time']].groupby(df['review_time'].dt.date).agg(['count'])
group_by_date.iplot(kind='line', title='Reviews Over Time')