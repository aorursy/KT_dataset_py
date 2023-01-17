# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import *

import nltk, datetime

import seaborn as sns



import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/sales_train.csv')

test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/sample_submission.csv')

items = pd.read_csv('../input/items.csv')

item_categories = pd.read_csv('../input/item_categories.csv')

shops = pd.read_csv('../input/shops.csv')

print('train:', train.shape, 'test:', test.shape)
train.head()
submission.head()
train.item_cnt_day.plot()

plt.title("Number of products sold per day");
from wordcloud import WordCloud

import random



def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)



item = ' '.join(items.item_name).lower()

# wordcloud for display address

plt.figure(figsize=(12,6))

wc = WordCloud(background_color='black', max_font_size=200,

                            width=1600,

                            height=800,

                            max_words=400,

                            relative_scaling=.5).generate(item)

plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))

#plt.imshow(wc)

plt.title("Items", fontsize=20)

plt.savefig('items-wordcloud.png')

plt.axis("off");