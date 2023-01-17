import re, string, unicodedata, random

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot



from collections import Counter

from itertools import chain



from nltk import word_tokenize, sent_tokenize

from nltk.corpus import stopwords



from wordcloud import WordCloud
STOPWORDS = stopwords.words('portuguese')



blind = {

    "<empty>": "",

}
def color_func(word, font_size, position, orientation, random_state=None,

                    **kwargs):

    COLORS = ['#b58900', '#cb4b16', '#dc322f', 

          '#d33682', '#6c71c4', '#268bd2', '#2aa198', '#859900']

    return COLORS[random.randint(0, len(COLORS)-1)]



def convert(x):

  x = str(x)

  return f'{x[:4]}-{x[4:]}'



def re_sub(text, pattern, repl):

    return re.sub(pattern, repl, text)





def remove_non_ascii(text):

    new_tokens = []

    tokens = text.split()

    

    for token in tokens:

        token = unicodedata.normalize('NFKD', token).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        new_tokens.append(token)

    

    return ' '.join(new_tokens)





def remove_punctuation(text):

    tokens = [c for c in text if c not in string.punctuation]

                

    return ''.join(tokens)





def strip_text(text):

    return text.strip()





def remove_stopwords(text):

    tokens = text.split()

    tokens = [token for token in tokens if token not in STOPWORDS]

                

    return ' '.join(tokens)





def normalize_serie(text):

    text = text.lower()

    text = remove_stopwords(text)

    text = remove_non_ascii(text)

    

    text = re_sub(text, r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", blind["<empty>"])

    text = re_sub(text, r"\b(\w*rt\w*)\b", blind["<empty>"])

    text = re_sub(text, r"\b(\w*jairbolsonaro\w*)\b", blind["<empty>"])

    text = re_sub(text, r"\b(k+)\b", blind["<empty>"])

    text = re_sub(text, r"\b(\d+)\b", blind["<empty>"])

    

    text = strip_text(text)

    text = remove_punctuation(text)



    return text
f = '/kaggle/input/bolsonaros-200-days-as-president-on-twitter/jairbolsonaro.csv'
df = pd.read_csv(f, date_parser=['created_at'])
df.head()
df.info()
df['YearMonth'] = pd.to_datetime(df['created_at']).apply(lambda x: int(f'{x.year}{x.month}'))
res = df.groupby('YearMonth')['id'].count()

print(res)
X = tuple(map(convert, res.index))

Y = res.values



fig = go.Figure(data=[go.Scatter(x=X, y=Y, text=Y)])

fig.update_layout(title='Tweets per month - @jairbolsonaro', 

                  xaxis_title='Month', yaxis_title='Tweets')



fig.show()
idx_retweet_count = df.groupby('YearMonth')['retweet_count'].transform(max) == df['retweet_count']

x = df[idx_retweet_count]['YearMonth'].apply(convert)



retweet_count = df[idx_retweet_count]['retweet_count']

hovertext = df[idx_retweet_count]['text']



fig = go.Figure(data=[go.Bar(

    x=x, 

    y=retweet_count,

    text=retweet_count,

    textposition='auto',

    hovertext=hovertext,

  )

])



fig.update_layout(title='Tweets most retweeted per month - @jairbolsonaro')



fig.show()
idx_favorite_count = df.groupby('YearMonth')['favorite_count'].transform(max) == df['favorite_count']



favorite_count = df[idx_favorite_count]['favorite_count']

hovertext = df[idx_favorite_count]['text']



fig = go.Figure(data=[go.Bar(

    x=x, 

    y=favorite_count,

    text=favorite_count,

    textposition='auto',

    hovertext=hovertext,

  )

])



fig.update_layout(title='Tweets most favorited per month - @jairbolsonaro')



fig.show()
temp = df[['YearMonth', 'favorite_count', 'retweet_count']

          ].groupby(['YearMonth'], as_index=False).sum()



fig = go.Figure(

    data=[

      go.Bar(name='Retweet', x=x, y=temp['retweet_count'], 

             text=temp['retweet_count'], textposition='auto'),

      go.Bar(name='Favorite', x=x, y=temp['favorite_count'], 

             text=temp['favorite_count'], textposition='auto')

])



fig.update_layout(title='Retweets <i>vs</i> Favorite tweets - @jairbolsonaro', barmode='group')

fig.show()
regex_mention = r'(@\w+)'

df['mentions'] = df.text.apply(lambda x: ' '.join(re.findall(regex_mention, x)))
mentions = df[['mentions', 'YearMonth']].loc[df.mentions.str.contains('@')].groupby('YearMonth', as_index=False).count().sort_values(by='YearMonth')



fig = go.Figure(data=go.Bar(name='Mentions', x=x, y=mentions['mentions'], 

                            text=mentions['mentions'], textposition='auto'))

fig.update_layout(title='Tweets that mention some user - @jairbolsonaro')

fig.show()
rts = df.loc[df.text.str.contains('RT ')].groupby('YearMonth', as_index=False).count().sort_values(by='YearMonth')['id'].values

fig = go.Figure(data=go.Bar(name='Mentions', x=x, y=rts, 

                            text=rts, textposition='auto'))

fig.update_layout(title='Retweets per Month - @jairbolsonaro')

fig.show()
not_rts = Y - rts



fig = go.Figure(

    data=[

         go.Bar(name='Tweets', x=x, y=not_rts, text=not_rts, textposition='auto'),

         go.Bar(name='RT', x=x, y=rts, text=rts, textposition='auto'),

])



fig.update_layout(title='Tweets composition per month - @jairbolsonaro', 

                  barmode='stack')



fig.show()
df['Hour'] = pd.to_datetime(df['created_at']).apply(lambda x: int(x.hour))
hours = df[['Hour', 'id']].groupby('Hour', as_index=False).count().sort_values(by='Hour')



fig = go.Figure(

      data=[go.Bar(x=hours['Hour'], y=hours['id'], 

                   text=hours['id'], textposition='auto')

      ],

)



fig.update_layout(title='Tweet Frequency by hour - @jairbolsonaro')

fig.show()
df['WeekDay'] = pd.to_datetime(df['created_at']).apply(lambda x: x.strftime('%w'))
weekdays = df[['WeekDay', 'id']].groupby('WeekDay', as_index=False).count().sort_values(by='WeekDay')

days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',]



fig = go.Figure(data=[go.Bar(x=days, y=weekdays['id'], text=weekdays['id'], textposition='auto')])

fig.update_layout(title='Tweet Frequency by weekday - @jairbolsonaro')



fig.show()
months_week = df[['YearMonth', 'WeekDay', 'id']].groupby(['YearMonth', 'WeekDay'], as_index=False).count()

  

fig = go.Figure()



fig.add_scatter(

    x=months_week['YearMonth'].apply(convert), 

    y=months_week['WeekDay'].apply(lambda x: days[int(x)]), mode='markers+text', 

    marker_color=[

                  '#b58900', '#cb4b16', '#dc322f', 

                  '#d33682', '#6c71c4', '#268bd2', 

                  '#2aa198',

                ] * len(x),

    # text=months_week['id'],

    marker=dict(size=months_week['id'] * .7)

)



fig.update_layout(title='Tweets frenquency by weekday per Month - @jairbolsonaro')



fig.show()
months_week = df[['WeekDay', 'Hour', 'id']].groupby(['WeekDay', 'Hour'], as_index=False).count()

  

fig = go.Figure()



fig.add_scatter(

    x=months_week['Hour'], y=months_week['WeekDay'].apply(lambda x: days[int(x)]), 

    mode='markers+text', text=months_week['id'],

    marker=dict(size=months_week['id'])

)



fig.update_layout(title='Tweets frenquency by hour per Weekday - @jairbolsonaro')



fig.show()
all_mentions = []

for year in X:

  

  mentions = []

  mentions_ = df.loc[

                    (df.YearMonth == int(year.replace('-', ''))) 

                    & (df.mentions != '')

                    & (df.text.str.contains('RT ') == False)

                  ]['mentions'].values



  for m in mentions_:

    for mention in m.split():

      mentions.append(mention)

  

  all_mentions.append(mentions)
counter = []

for mentions in all_mentions:

  counter.append(Counter(mentions))
mentions = []

saved_mention = ['@jairbolsonaro']



for co in counter:

  for mention in list(co.most_common()):

    values = []

    name = mention[0]

    

    if name.lower() in saved_mention:

      continue

    

    for co in counter:

      if name in chain(*co.most_common()):

        for mention in list(co.most_common()):

          if mention[0] == name:

            values.append(mention[1])

      else:

        values.append(0)



    if sum(values) > 1 and name.lower() not in saved_mention:

      mentions.append((name, values, sum(values)))

      saved_mention.append(name.lower())

        

mentions = sorted(mentions)
fig = go.Figure()



for mention in mentions:

  fig.add_trace(go.Scatter(x=x, y=mention[1], name=mention[0], mode='lines'))



fig.update_layout(title='Mentions per month - @jairbolsonaro',)



fig.show()
all_mentions = []

for year in X:

  

  mentions = []

  mentions_ = df.loc[

                    (df.YearMonth == int(year.replace('-', ''))) 

                    & (df.mentions != '')

                    & (df.text.str.contains('RT '))

                  ]['mentions'].values



  for m in mentions_:

    for mention in m.split():

      mentions.append(mention)

  

  all_mentions.append(mentions)
counter = []

for mentions in all_mentions:

  counter.append(Counter(mentions))
mentions = []

saved_mention = ['@jairbolsonaro']



for co in counter:

  for mention in list(co.most_common()):

    values = []

    name = mention[0]

    

    if name.lower() in saved_mention:

      continue

    

    for co in counter:

      if name in chain(*co.most_common()):

        for mention in list(co.most_common()):

          if mention[0] == name:

            values.append(mention[1])

      else:

        values.append(0)



    if sum(values) > 1 and name.lower() not in saved_mention:

      mentions.append((name, values, sum(values)))

      saved_mention.append(name.lower())

        

mentions = sorted(mentions)
fig = go.Figure()



for mention in mentions:

  fig.add_trace(go.Scatter(x=x, y=mention[1], name=mention[0], mode='lines'))



fig.update_layout(title='Mentions per month - @jairbolsonaro',)



fig.show()
df['normalized'] = df['text'].apply(normalize_serie)
wordcloud = WordCloud(

    width=3000,

    height=2000,

    background_color='#073642',

    collocations=False,

    

).generate(' '.join(df['normalized'].values))
fig = plt.figure(

    figsize=(20, 15),

    facecolor='k',

    edgecolor='k'

)



plt.axis('off')

plt.tight_layout(pad=0)

plt.imshow(wordcloud.recolor(color_func=color_func, random_state=3),

           interpolation="bilinear")

plt.show()