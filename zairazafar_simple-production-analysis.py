# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
df = pd.read_csv("../input/FAO.csv",encoding="ISO-8859-1")
df.head(5)

graph_by_items = df['Item'].value_counts()
fig, ax = plt.subplots()
ax.tick_params(axis = 'x', labelsize = 10)
ax.tick_params(axis = 'y', labelsize = 10)
ax.set_title('Top 10 Items of Production', fontsize = 15, fontweight = 'bold')
graph_by_items[:10].plot(ax=ax, kind='bar', color='blue')
graph_by_area = df['Area'].value_counts()
fig, ax = plt.subplots()
ax.tick_params(axis = 'x', labelsize = 10)
ax.tick_params(axis = 'y', labelsize = 10)
ax.set_title("Top 10 Areas Of Production", fontsize = 15, fontweight = 'bold')
graph_by_area[:10].plot(ax=ax, kind = 'bar', color = 'green')
grouped = df.groupby('Area')
grouped_Armenia = grouped.get_group('Armenia')

graph_by_area = grouped_Armenia['Item'].value_counts()
fig, ax = plt.subplots()
ax.tick_params(axis = 'x', labelsize = 10)
ax.tick_params(axis = 'y', labelsize = 10)
ax.set_title("Top 5 Items in Armenia", fontsize = 15, fontweight = 'bold')
graph_by_area[:10].plot(ax=ax, kind = 'bar', color = 'green')
graph_by_area = grouped_Armenia['Element'].value_counts()
fig, ax = plt.subplots()
ax.tick_params(axis = 'x', labelsize = 10)
ax.tick_params(axis = 'y', labelsize = 10)
ax.set_title("Element Most Produced in Armenia", fontsize = 15, fontweight = 'bold')
graph_by_area[:10].plot(ax=ax, kind = 'bar', color = 'green')
top_countries=df.groupby(["Area"])[["Y2004","Y2005","Y2005","Y2006","Y2007","Y2008","Y2009","Y2010","Y2011",
                                    "Y2012","Y2013"]].sum()
top=pd.DataFrame(top_countries.agg("mean",axis=1).sort_values(ascending=False),columns=["Tonnes"])[:10]
import seaborn as sns
plt.figure(figsize=(8,8))
plt.gca().set_title("Top producers throughout the years")
sns.barplot(x=top["Tonnes"],y=top.index,data=top)
plt.gcf().subplots_adjust(left=.3)
plt.show()
import re
def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False
df['Food'] = df['Element'].apply(lambda tweet: word_in_text('Food', tweet))
df['Feed'] = df['Element'].apply(lambda tweet: word_in_text('Feed', tweet))

prg_langs = ['Food', 'Feed']
tweets_by_prg_lang = [df['Food'].value_counts()[True], df['Feed'].value_counts()[True]]

x_pos = list(range(len(prg_langs)))
width = 0.8
fig, ax = plt.subplots()
plt.bar(x_pos, tweets_by_prg_lang, width, alpha=1, color='g')

# Setting axis labels and ticks
ax.set_ylabel('Number of tweets', fontsize=15)
ax.set_title('Ranking: Food vs. Feed  (Raw data)', fontsize=10, fontweight='bold')
ax.set_xticks([p + 0.4 * width for p in x_pos])
ax.set_xticklabels(prg_langs)
plt.grid()

area_2013 = df.groupby('Area')['Y2013'].agg('sum')
print(area_2013)

area_1961 = df.groupby('Area')['Y1961'].agg('sum')
print(area_1961)
