# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
init_notebook_mode()
df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines=False)
df.head()
df.info()
df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
df.isna().sum()
df.query('publication_date=="nat"')
df = df.dropna()
df['year_publication'] = df['publication_date'].dt.year
df['month_publication'] = df['publication_date'].dt.to_period('M')
df = df.rename(columns={'  num_pages':'num_pages'})
print('the latest publication', df['publication_date'].max())
print('the earliest publication', df['publication_date'].min())
df.info()
df.describe()
books = df.groupby('year_publication').agg({'title':'count'})
books.head()
trace = go.Bar(x=books.index, y=books['title'])
layout = go.Layout(title={'text': 'Data Distribution of Goodread Book', 'x':0.5, 'y':0.9})
fig=go.Figure(data=trace, layout=layout)
iplot(fig)
most_authors= df.groupby('authors').agg({'title': 'count'}).sort_values('title', ascending=False).head(10)
most_authors
colors = ['black' if(x<max(most_authors)) else 'green' for x in most_authors]
traceAuthors = go.Bar(x=most_authors.index, y=most_authors['title'], marker=dict(color=colors))
layout = go.Layout(title={'text': 'Most Productive Authors', 'x':0.5, 'y':0.9})
fig=go.Figure(data=traceAuthors, layout=layout)
iplot(fig)
good_ratings = df.query('ratings_count>10000').sort_values('average_rating', ascending=False).head(10)
good_ratings
traceGoodRatings = go.Bar(x=good_ratings['average_rating'], 
                          y=good_ratings['title'], 
                          orientation='h', 
                          marker=dict(color=colors))

layout=go.Layout(title={'text': 'Book with Good Ratings', 
                        'x':0.5, 
                        'y':0.9})

fig=go.Figure(data=traceGoodRatings,
              layout=layout)

iplot(fig)
most_reviews=df.query('ratings_count>10000').sort_values('text_reviews_count', ascending=False).head(10)
most_reviews
traceMostReviews = go.Bar(x=most_reviews['text_reviews_count'], y=most_reviews['title'], orientation='h', marker=dict(color=colors))
layout=go.Layout(title={'text': 'Book with Most Reviews', 'x':0.5, 'y':0.9})
fig=go.Figure(data=traceMostReviews, layout=layout)
iplot(fig)