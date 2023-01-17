# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from textblob import TextBlob





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/abcnews-date-text.csv",parse_dates=[0], infer_datetime_format=True)

data.columns = ['date','text']

data.head(5)
reindexed_data = data['text']

reindexed_data.index = data['date']

reindexed_data.head()
Xi = [0]*reindexed_data.shape[0]

Trump = [0]*reindexed_data.shape[0]

Trade = [0]*reindexed_data.shape[0]

War = [0]*reindexed_data.shape[0]



for i in range(reindexed_data.shape[0]):

    words = TextBlob(reindexed_data[i]).words

    for word in words:

        if word == "xi" or word == "jinping": Xi[i]=1

        if word == "trump": Trump[i]=1

        if word == "trade": Trade[i]=1

        if word == "war": War[i]=1   # as all headlines are in lowercase

keywords = pd.DataFrame({'text':reindexed_data,

                        'Xi':Xi,

                        'Trump':Trump,

                        'Trade':Trade,

                        'War':War},

                        index=reindexed_data.index)

keywords.head()
monthly = keywords.resample('M').sum()

#yearly = keywords.resample('A').sum()

print(monthly)
fig, ax = plt.subplots(figsize=(18,8))



ax.plot(monthly['Xi'], label='Xi');

ax.plot(monthly['Trump'], label='Trump');

ax.plot(monthly['Trade'], label='Trade');

ax.plot(monthly['War'], label='War');

ax.set_title('Keywords Frequency over Time');

ax.legend(loc='upper left');