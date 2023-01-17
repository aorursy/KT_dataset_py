!pip install pymorphy2[fast]

!pip install bigartm10

!pip install razdel
import re

from collections import Counter



import pandas as pd

import numpy as np



import artm

from sklearn.feature_extraction.text import CountVectorizer

import pymorphy2

import razdel



from wordcloud import WordCloud

import matplotlib.pyplot as plt
news_df = pd.read_csv('/kaggle/input/russian-news-2020/news.csv')
news_df.head()
news_df.shape
news_df.loc[news_df['source'] == 'ria.ru', 'publication_date'] = (news_df.loc[news_df['source'] == 'ria.ru', 'publication_date'].str

                                                              .extract(r'(?P<date>\d{2}\.\d{2}\.\d{4})', expand=False)

                                                              .apply(lambda x: '-'.join(reversed(x.split('.'))) if type(x) is str else x))
news_df.loc[news_df['source'] == 'lenta.ru', 'publication_date'] = news_df.loc[news_df['source'] == 'lenta.ru', 'publication_date'].str.split('T').str.get(0)
month_mapper = {

    'января': '01',

    'февраля': '02',

    'марта': '03',

    'апреля': '04',

    'мая': '05',

    'июня': '06',

    'июля': '07',

    'августа': '08',

    'сентября': '09',

    'октября': '10',

    'ноября': '11',

    'декабря': '12'

}

news_df.loc[news_df['source'] == 'meduza.io', 'publication_date'] = (news_df.loc[news_df['source'] == 'meduza.io', 'publication_date']

                                                                     .apply(lambda x: f'{x.split()[3]}-{month_mapper[x.split()[2]]}-{x.split()[1].zfill(2)}' if type(x) is str else x))
news_df.loc[news_df['source'] == 'tjournal.ru', 'publication_date'] = pd.to_datetime(news_df.loc[news_df['source'] == 'tjournal.ru', 'publication_date'], unit='s').dt.strftime('%Y-%m-%d')
news_df.loc[news_df['source'] == 'tjournal.ru', 'text'] = news_df.loc[news_df['source'] == 'tjournal.ru', 'text'].str.replace('\n', '').str.replace(r'\s+', ' ')
news_df.loc[news_df['source'] == 'tjournal.ru', 'tags'] = news_df.loc[news_df['source'] == 'tjournal.ru', 'text'].str.findall(r'#\w+').str.join(', ').str.replace('#', '')
news_df.loc[news_df['source'] == 'tjournal.ru', 'text'] = news_df.loc[news_df['source'] == 'tjournal.ru', 'text'].apply(lambda x: x[:x.find('#')])
documents = news_df.text.tolist()
morph = pymorphy2.MorphAnalyzer()



def lemmatize(token):

    return morph.parse(token)[0].normal_form



def tokenize(text):

    return [lemmatize(token.text) for token in razdel.tokenize(text) if len(token.text) > 2]
cv = CountVectorizer(max_features=25000, min_df=5, max_df=0.25, tokenizer=tokenize)

n_wd = np.array(cv.fit_transform(documents).todense()).T

vocabulary = cv.get_feature_names()
bv = artm.BatchVectorizer(data_format='bow_n_wd',

                          n_wd=n_wd,

                          vocabulary=vocabulary)
model = artm.ARTM(

    num_topics=100, dictionary=bv.dictionary,

    scores=[

        artm.PerplexityScore(name='PerplexityScore', dictionary=bv.dictionary),

        artm.TopTokensScore(name='Top10TokensScore', num_tokens=10),

        artm.TopTokensScore(name='Top100TokensScore', num_tokens=100),

        artm.SparsityPhiScore(name='SparsityPhiScore'),

        artm.SparsityPhiScore(name='SparsityThetaScore'),

    ],

    regularizers=[

        artm.SmoothSparseThetaRegularizer(name='SmoothSparseThetaRegularizer', tau=-1e-4),

        artm.SmoothSparsePhiRegularizer(name='SmoothSparsePhiRegularizer', tau=-1e-4),

    ]

)

model.fit_offline(bv, num_collection_passes=50)
model.get_score('PerplexityScore')
plt.plot(model.score_tracker['PerplexityScore'].value)
model.get_score('SparsityPhiScore')
plt.plot(model.score_tracker['SparsityPhiScore'].value)
model.get_score('SparsityThetaScore')
plt.plot(model.score_tracker['SparsityThetaScore'].value)
for topic_name in model.topic_names:

    print(topic_name + ': ',)

    print(model.score_tracker['Top10TokensScore'].last_tokens[topic_name])
news_df['topic'] = model.transform(bv).idxmax(axis=0).str.replace('topic_', '').sort_index().astype(int)
for i in range(news_df.topic.max()):

    print(f'Topic: {i}')

    counts = news_df[news_df.topic == i].rubric.value_counts()

    print(counts[counts > 2])

    print()
for i in range(news_df.topic.max()):

    print(f'Topic: {i}')

    counts = news_df[news_df.topic == i].subrubric.value_counts()

    print(counts[counts > 2])

    print()
for i in range(news_df.topic.max()):

    print(f'Topic: {i}')

    tags = []

    for i in news_df[news_df.topic == i].tags.dropna():

        tags += i.split(', ')

    counts = Counter(tags)

    print('\n'.join(map(str, counts.most_common()[:5])))

    print()
for topic_name in model.topic_names:

    print(topic_name)

    top_tokens = set(model.score_tracker['Top100TokensScore'].last_tokens[topic_name])

    frequencies = Counter()

    for text in news_df[news_df.topic == int(topic_name.replace('topic_', ''))].text:

        frequencies.update([token for token in tokenize(text) if token in top_tokens])

    

    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(frequencies)

    plt.figure(figsize=(10, 5))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()
f = plt.figure()

f, ax = plt.subplots(100, 1, figsize=(75, 900))



for i, topic_name in enumerate(model.topic_names):

    counts = news_df[news_df.topic == int(topic_name.replace('topic_', ''))]['publication_date'].dropna().value_counts().to_dict()

    ax[i].bar(news_df['publication_date'].dropna().drop_duplicates().sort_values(), news_df['publication_date'].dropna().drop_duplicates().sort_values().map(counts))

    ax[i].set_title(topic_name)

    ax[i].tick_params(labelrotation=90)