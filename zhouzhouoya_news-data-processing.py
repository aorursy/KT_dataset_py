#Firstly,import.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
stop = set(stopwords.words('english'))


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn import model_selection
from sklearn.metrics import accuracy_score
#Secondly, load the data.
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('Done!')
#get training data
(market_train_df, news_train_df) = env.get_training_data()
news_train_df.head()
#Data Description:
#The news data contains information about news articles/alerts published about assets, such as article details, sentiment, and other commentary. Both the news article level and asset level (in other words, the table is intentionally not normalized).
print(f'{news_train_df.shape[0]} samples and {news_train_df.shape[1]} features in the training news dataset.')
#The file is too huge to work with text directly, so let's see a wordcloud of the last 100000 headlines.
text = ' '.join(news_train_df['headline'].str.lower().values[-1000000:])
wordcloud = WordCloud(max_font_size=None, stopwords=stop, background_color='white',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words in headline')
plt.axis("off")
plt.show()
#Thirdly, deal with possible data errors.
# sorted by time
market_train_orig = market_train_df.sort_values('time')
news_train_orig = news_train_df.sort_values('time')
market_train_df = market_train_orig.copy()
news_train_df = news_train_orig.copy()
del market_train_orig
del news_train_orig
#The statistics show that most data are stable after 2009 (the increment of volume and price, etc.). However, before 2009, the data were performed differently because of the housing bubble burst in 2008 Financial Crisis. Thus, we choose the data after 2009.
market_train_df = market_train_df.loc[market_train_df['time'].dt.date>=datetime.date(2009,1,1)]
news_train_df = news_train_df.loc[news_train_df['time'].dt.date>=datetime.date(2009,1,1)]
#Fourthly, remove outliers.
# Function to remove outliers
def remove_outliers(data_frame, column_list, low=0.02, high=0.98):
    for column in column_list:
        this_column = data_frame[column]
        quant_df = this_column.quantile([low,high])
        low_limit = quant_df[low]
        high_limit = quant_df[high]
        data_frame[column] = data_frame[column].clip(lower=low_limit, upper=high_limit)
    return data_frame
# Remove outlier
columns_outlier = ['takeSequence', 'bodySize', 'sentenceCount', 'wordCount', 'sentimentWordCount', 'firstMentionSentence','noveltyCount12H',\
                  'noveltyCount24H', 'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H', 'volumeCounts24H',\
                  'volumeCounts3D','volumeCounts5D','volumeCounts7D']
print('Clipping news outliers ...')
news_train_df = remove_outliers(news_train_df, columns_outlier)
#Finally, return the results.
print(f'{news_train_df.shape[0]} samples and {news_train_df.shape[1]} features in the training news dataset.')
news_train_df.describe()