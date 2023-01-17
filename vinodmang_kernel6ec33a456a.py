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
from kaggle.competitions import twosigmanews
import pandas as pd
import numpy as np
import plotly
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Figure, Layout

init_notebook_mode(connected=True)
env = twosigmanews.make_env()
market_data_df,news_data_df = env.get_training_data()
news_data_df.head()
market_data_df.head()
cutoff_year = '2016-09-09'
market_data_df = market_data_df[market_data_df['time'] > cutoff_year]
news_data_df = news_data_df[news_data_df['time'] > cutoff_year]
market_data_df['date'] = market_data_df.apply(lambda row: str(row['time']).split()[0],1)
news_data_df['date'] = news_data_df.apply(lambda row: str(row['time']).split()[0],1)
news_data_df.head()
news_cols_of_interest = ['sentimentNegative','sentimentPositive','sentimentNeutral']
string_cols_of_interest = 'headline'
def join_news_to_market(M,N,nrows=1000):
    row_cnt=0
    all_join=[]
    headline_data=[' ']
    for row in M.itertuples(index=False):
        joined_df = []
        joined_df.append(list(row))
        dt = row.date
        asset = row.assetName
        all_news_df = N[(N['date'] == dt) & (N['assetName']==asset)]
        row_cnt=row_cnt+1
        
        sentiment_data = np.array(all_news_df[news_cols_of_interest].mean(skipna=True))
        
        #if(all_news_df[string_cols_of_interest].count() ==0):
        #    continue;
        #else:
        s = " ".join(list(all_news_df[string_cols_of_interest]))
        
        j = np.concatenate((np.squeeze(np.transpose(np.array(joined_df))),np.array(sentiment_data),np.array([s])),0)
        all_join.append(j)
        if row_cnt > nrows:
            break
    return all_join
joined = join_news_to_market(market_data_df,news_data_df)
colnames=list(market_data_df.columns)
colnames.extend(news_cols_of_interest)
colnames.extend([string_cols_of_interest])
dat = np.squeeze(np.array(joined))
print(dat.shape)
print(colnames)
joined_df = pd.DataFrame(data=dat,columns=colnames)
joined_df.loc[:2,'headline']
import matplotlib.pyplot as plt
cols =['returnsOpenNextMktres10'].append(news_cols_of_interest)

a = joined_df.loc[:,['returnsOpenNextMktres10','sentimentNegative','sentimentPositive','sentimentNeutral']]
b=a.dropna()
c=b.infer_objects()
print(c.dtypes)
print(len(c))
c.corr()