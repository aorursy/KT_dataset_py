!pip install pymorphy2[fast]
import re

from collections import Counter

import pandas as pd

import numpy as np

from gensim import corpora, models

import pymorphy2

from wordcloud import WordCloud

import matplotlib.pyplot as plt
morph = pymorphy2.MorphAnalyzer()
def lemmatize(token):

    return morph.parse(token)[0].normal_form
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
texts = [

    [lemmatize(word) for word in re.findall(r'\w+', document.lower()) if len(word) > 2]

    for document in documents

]
dictionary = corpora.Dictionary(texts)

dictionary.filter_extremes(no_below=5, no_above=0.25, keep_n=25000)

corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = models.ldamulticore.LdaMulticore(corpus, id2word=dictionary, num_topics=100, passes=50, alpha='symmetric', eta=None, decay=0.5)
perplexity = ldamodel.log_perplexity(corpus)

print(2**(-perplexity))
for t, top_words in ldamodel.print_topics(num_topics=-1, num_words=10):

    print("Topic", t, ":", top_words)

    print()
news_df['topic'] = [max(i, key=lambda x: x[1])[0] for i in ldamodel[corpus]]
for i in range(news_df.topic.max()):

    print(f'Topic: {i}')

    counts = news_df[news_df.topic == i].rubric.value_counts()

    print(counts[counts > 5])

    print()
for i in range(news_df.topic.max()):

    print(f'Topic: {i}')

    counts = news_df[news_df.topic == i].subrubric.value_counts()

    print(counts[counts > 5])

    print()
for i in range(news_df.topic.max()):

    print(f'Topic: {i}')

    tags = []

    for i in news_df[news_df.topic == i].tags.dropna():

        tags += i.split(', ')

    counts = Counter(tags)

    print('\n'.join(map(str, counts.most_common()[:5])))

    print()
for i in range(news_df.topic.max()):

    print(f'Topic: {i}')

    frequencies = dict(ldamodel.show_topic(i, topn=100))

    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(frequencies)

    plt.figure(figsize=(10, 5))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()
f = plt.figure()

f, ax = plt.subplots(100, 1, figsize=(75, 900))



for i, topic_name in enumerate(range(news_df.topic.max())):

    counts = news_df[news_df.topic == topic_name]['publication_date'].dropna().value_counts().to_dict()

    ax[i].bar(news_df['publication_date'].dropna().drop_duplicates().sort_values(), news_df['publication_date'].dropna().drop_duplicates().sort_values().map(counts))

    ax[i].set_title(topic_name)

    ax[i].tick_params(labelrotation=90)