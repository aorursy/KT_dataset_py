# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')

test=pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')
train.head()
labels=train['label']

train=train.drop('label',axis=1)

labels
from nltk.sentiment.vader import SentimentIntensityAnalyzer
model=SentimentIntensityAnalyzer()
sentiment=[]

for i in range(len(test)):

    prob=model.polarity_scores(train['tweet'][i])

    if(prob['pos']>prob['neu'] and prob['pos']>prob['neg']):

        sentiment.append('Positive')

    elif(prob['neg']>prob['pos'] and prob['neg']>prob['neu']):

        sentiment.append('Negative')

    else:

        sentiment.append('Neutral')

print(sentiment)