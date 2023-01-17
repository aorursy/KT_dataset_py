import pandas as pd
import numpy as np
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/metal-by-nation/metal_bands_2017.csv', encoding='latin-1')
data
data.drop('Unnamed: 0', axis=1, inplace=True)
data.dropna(inplace=True)
data.drop_duplicates(subset='band_name', keep=False, inplace=True)
origin_cnt_top10 = data.origin.value_counts()[:10]
fig = px.pie(origin_cnt_top10, values=origin_cnt_top10, names=origin_cnt_top10.index, 
             title='Top 10 countries',)
fig.update_traces(textposition='inside', textinfo='label')
fig.show()
dataf_2000 = data[data.formed >= '2000']
formed_cnt = dataf_2000.formed.value_counts()
fig = px.bar(formed_cnt, x=formed_cnt.index, y=formed_cnt, labels={'y':'Bands created', 'index':'Year'})
fig.update_traces(texttemplate='%{value}', textposition='outside')
fig.update_layout(xaxis = dict(tickmode = 'linear'))
fig.show()
datas_2000 = data[data.split >= '2000']
split_cnt = datas_2000.split.value_counts()
fig = px.bar(split_cnt, x=split_cnt.index, y=split_cnt, labels={'y':'Bands disbanded', 'index':'Year'})
fig.update_traces(texttemplate='%{value}', textposition='outside')
fig.update_layout(xaxis = dict(tickmode = 'linear'))
fig.show()
formed_cnt.sort_index(inplace=True)
split_cnt.sort_index(inplace=True)
fig = go.Figure(data=[
    go.Bar(name='Created', x=formed_cnt.index, y=formed_cnt),
    go.Bar(name='Disbanded', x=formed_cnt.index, y=split_cnt)])
fig.update_layout(barmode='group', xaxis = dict(tickmode = 'linear'))
fig.show()
data.split.unique()
data = data[data.split != '-']
data['years'] = data['split'].astype('int32') - data['formed'].astype('int32')
print('How long metal bands live (in years)')
data.years.value_counts().sort_index()
data[data.years == 41]
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
comment_words = ''
stopwords = set(STOPWORDS)
for val in data['style']:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 700, height = 700,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)

plt.figure(figsize = (7, 7), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()