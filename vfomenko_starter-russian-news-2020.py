from collections import Counter

from datetime import datetime

import re

import os



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
news = pd.read_csv('/kaggle/input/news.csv')
news.head()
plt.hist(news['source'])

news['source'].value_counts()
ria = news[news.source == 'ria.ru']

ria.head()
print(ria.iloc[0].title)

print(ria.iloc[0].text)
print(f'Mean length: {ria.text.str.len().mean()}')

plt.figure(figsize=(20, 5))

sns.distplot(ria.text.str.len())
tags = []

for i in ria.tags.dropna():

    tags += i.split(', ')

tag_counter = Counter(tags)
len(tag_counter)
tag_counter.most_common()[:20]
lenta = news[news.source == 'lenta.ru']

lenta.head()
print(lenta.iloc[1].title)

print(lenta.iloc[1].text)
print(f'Mean length: {lenta.text.str.len().mean()}')

plt.figure(figsize=(20, 5))

sns.distplot(lenta.text.str.len())
print(f'First date: {lenta.publication_date.min()}')

print(f'Last date: {lenta.publication_date.max()}')
plt.figure(figsize=(10, 10))

lenta['rubric'].value_counts().plot.barh().invert_yaxis()

lenta['rubric'].value_counts()
plt.figure(figsize=(10, 20))

lenta['subrubric'].value_counts().plot.barh().invert_yaxis()

lenta['subrubric'].value_counts()
meduza = news[news.source == 'meduza.io']

meduza.head()
print(meduza.iloc[1].title)

print(meduza.iloc[1].text)
print(f'Mean length: {meduza.text.str.len().mean()}')

plt.figure(figsize=(20, 5))

sns.distplot(meduza.text.str.len())
tjournal = news[news.source == 'tjournal.ru']

tjournal.head()
print(tjournal.iloc[1].title.strip())

print(tjournal.iloc[1].text.replace('\n', '').strip())
print(f'Mean length: {tjournal.text.str.len().mean()}')

plt.figure(figsize=(20, 5))

sns.distplot(tjournal.text.str.len())
def convert_from_unix_time(ts):

    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print(f'First date: {convert_from_unix_time(int(tjournal.publication_date.min()))}')

print(f'Last date: {convert_from_unix_time(int(tjournal.publication_date.max()))}')
tags = []

for i in tjournal.text:

    tags += re.findall(r'#\w+', i)

tag_counter = Counter(tags)
len(tag_counter)
tag_counter.most_common()[:20]