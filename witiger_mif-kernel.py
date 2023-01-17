import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from subprocess import os
#print(os.listdir("../input"))
data = pd.read_csv('../input/50k_random.csv')
data.columns = data.columns.str.replace('ym:pv:','')
data = data.rename(columns={'URL': 'url'})
data = data.rename(columns={'referer': 'referrer'})
data.info()
data2 = data[["url","link","referrer","title"]] 
data2.info()
title = data2.title.value_counts()
title = pd.DataFrame({'top':title.index, 'count':title.values})
title = title[title['count']>1000]
titletop = title.head(10)

plt.figure(figsize=(15,15))
sns.barplot(x=titletop['count'], y=titletop['top'])
plt.xticks(rotation= 45)
plt.xlabel('Hit counts')
plt.ylabel('Page Title')
plt.title('Топ 30 страниц')
referrer = data2.referrer.value_counts()
referrer = pd.DataFrame({'top':referrer.index, 'count':referrer.values})
referrer = referrer[referrer['count']>1000]
referrertop = referrer.head(10)

plt.figure(figsize=(10,15))
sns.barplot(x=referrertop['count'], y=referrertop['top'])
plt.xticks(rotation= 45)
plt.xlabel('Hit counts')
plt.ylabel('Page referrer')
plt.title('Топ 30 источников перехода')
url = data2.url.value_counts()
url = pd.DataFrame({'top':url.index, 'count':url.values})
url = url[url['count']>1000]
urltop = url.head(10)

plt.figure(figsize=(10,15))
sns.barplot(x=urltop['count'], y=urltop['top'])
plt.xticks(rotation= 45)
plt.xlabel('Hit counts')
plt.ylabel('Page URL')
plt.title('Топ 30 URL ')
alltop = pd.concat([urltop,titletop,referrertop], axis = 0, ignore_index = True)
alltop = alltop.sort_values('count', ascending=False)
alltop = alltop[alltop['count']>10000]
alltop = alltop.reset_index().drop('index', axis=1)
top5 = pd.DataFrame()
top5 = pd.concat([alltop.loc[[0]],alltop.loc[[4]],alltop.loc[[8]],alltop.loc[[11]],alltop.loc[[14]]], axis = 0, ignore_index = True)
plt.figure(figsize=(10,15))
sns.barplot(x=top5['count'], y=top5['top'], palette = sns.color_palette("husl", len(top5['count'])))
plt.xticks(rotation= 45)
plt.xlabel('Hit counts')
plt.ylabel('Триггер')
plt.title('Топ 5 триггеров ')