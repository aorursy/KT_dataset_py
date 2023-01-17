import pandas as pd

import plotly.express as px
community = pd.read_csv('/kaggle/input/covidqa/community.csv')

multilingual = pd.read_csv('/kaggle/input/covidqa/multilingual.csv')

news = pd.read_csv('/kaggle/input/covidqa/news.csv')
community.head()
community['site'] = community.url.str.replace(".stackexchange.com", "")
px.histogram(community, x='site', color='source')
multilingual.head()
px.histogram(multilingual, x='language', color='source')
news.head()
fig = px.histogram(news, x='source', color='url')

fig.update_layout(showlegend=False)

fig.show()