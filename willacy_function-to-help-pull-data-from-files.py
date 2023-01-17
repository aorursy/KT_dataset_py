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
df = pd.read_csv('../input/2019_data.csv')
df.Statistic.value_counts()
def convert_df(df, statistic):

    df = df[df.Statistic == statistic]



    df = pd.pivot_table(df, index=['Player Name','Date','Statistic'], 

                   columns = 'Variable', 

                   values='Value', 

                   aggfunc=[lambda x: ''.join(str(v) for v in x)]).reset_index()



    df.columns = df.columns.get_level_values(0)[:3].append(df.columns.get_level_values(1)[3:].map(lambda x:x.split('(')[-1]).str.replace(')','').str.title()).str.replace(' ','_')

    df.drop('Statistic', axis=1, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])

    return df

    
df_WR = convert_df(df, 'Official World Golf Ranking')

df_WR.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df_WR.Avg_Points = pd.to_numeric(df_WR.Avg_Points)
df_WR.replace('nan',np.nan).head()
Top10 = df_WR[df_WR['Date']==df_WR.Date.max()].sort_values(by='Avg_Points', ascending=False)['Player_Name'].head(10).values
fig, ax = plt.subplots(figsize=(10,5))



columns = ['Player_Name','Date','Avg_Points']



ax = sns.lineplot(x="Date", y="Avg_Points", hue="Player_Name", 

                  data=df_WR[columns][df_WR.Player_Name.isin(Top10)])



plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title("Average Points for Current Top 10 Players")

plt.show()