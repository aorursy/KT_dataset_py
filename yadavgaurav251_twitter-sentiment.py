# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt 

tweet_df = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv');
tweet_df
tweet_df.info()
tweet_df.describe()
del tweet_df['id']
tweet_df
sns.heatmap( tweet_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
tweet_df.hist(bins=30 , figsize=(13,5), color='c' )
sns.countplot(tweet_df['label'],label='Count')
tweet_df['length'] = tweet_df['tweet'].apply(len)
tweet_df['length'].plot(bins=100 , kind='hist' )
tweet_df.describe()
postive= tweet_df[tweet_df['label']==0]
negative= tweet_df[tweet_df['label']==1]
postive
negative
