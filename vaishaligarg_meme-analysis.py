# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



hillary = pd.read_csv("../input/Hillary.csv", encoding = "ISO-8859-1", parse_dates = ['timestamp'])

clinton = pd.read_csv("../input/Clinton.csv", encoding = "ISO-8859-1", parse_dates = ['timestamp'])

hillary = pd.concat([hillary, clinton], ignore_index=True, join="inner")

hillary = hillary[hillary['timestamp'] > datetime(2016, 1, 1)]

hillary.sort(['timestamp'], inplace = True)

hillary = hillary.drop_duplicates(['id'])



trump = pd.read_csv("../input/Trump.csv", encoding = "ISO-8859-1", parse_dates = ['timestamp'])

donald = pd.read_csv("../input/Donald.csv", encoding = "ISO-8859-1", parse_dates = ['timestamp'])

trump = pd.concat([trump, donald], ignore_index=True, join="inner")

trump = trump[trump['timestamp'] > datetime(2016, 1, 1)]

trump.sort(['timestamp'], inplace = True)

trump = trump.drop_duplicates(['id'])
hillary['year'] = pd.DatetimeIndex(hillary['timestamp']).year

hillary['month'] = pd.DatetimeIndex(hillary['timestamp']).month

hillary_agg = hillary.groupby([hillary['year'], hillary['month']])['likes'].sum()



trump['year'] = pd.DatetimeIndex(trump['timestamp']).year

trump['month'] = pd.DatetimeIndex(trump['timestamp']).month

trump_agg = trump.groupby([trump['year'], trump['month']])['likes'].sum()



data = pd.concat([hillary_agg, trump_agg], axis = 1)

data.columns = ['Hillary', 'Trump']
data.plot()
#Comparison of network usage

hillary_network = hillary['network'].value_counts()

trump_network = trump['network'].value_counts()



labels = 'facebook', 'imgur', 'instagram', 'twitter'

colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']



fig, (ax1, ax2) = plt.subplots(1, 2)



# plot each pie chart in a separate subplot

ax1.pie(hillary_network, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)

# Set aspect ratio to be equal so that pie is drawn as a circle.

ax2.pie(trump_network,  labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)

                               
#Overall there are more memes about Trump than Clinton, except in a couple of cases 

#(forwards from grandma and Conservative Country) where there is a spike in hillary value



hillary_author = hillary['author'].value_counts()[:20]

trump_author = trump['author'].value_counts()[:20]

author_data = pd.concat([hillary_author, trump_author], axis = 1, join = 'inner')

author_data.columns = ['Hillary', 'Trump']

author_data['pct_hillary'] = np.round(100 * author_data['Hillary']/author_data['Hillary'].sum(), 2)

author_data['pct_trump'] = np.round(100 * author_data['Trump']/author_data['Trump'].sum(), 2)

author_data_filtered = author_data[['pct_hillary', 'pct_trump']]

author_data_filtered.plot(kind='bar')