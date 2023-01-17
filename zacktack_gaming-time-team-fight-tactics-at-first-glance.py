import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
master_df = pd.read_csv("/kaggle/input/tft-match-data/TFT_Master_MatchData.csv")

master_df.head(8)
time_last = master_df[['gameId','gameDuration']].drop_duplicates().gameDuration.agg(['min','mean','max']).to_frame()

time_last.gameDuration = time_last.gameDuration.apply(lambda x: round(x / 60))

time_last.rename(columns = {'gameDuration' : 'gameDuration (min)'},inplace=True)

time_last
px.bar(time_last,x=time_last.index,y=time_last.values,labels={'x':'Aggregate Function','y':'Time (Minutes)'},title='Game Duration Time')
from collections import Counter

import re



class_and_origin = master_df.combination.apply(lambda x: re.findall(r'[a-zA-Z]+[0-9]?_?[a-zA-Z]+',x)).to_frame()

result = Counter()

for data in class_and_origin.combination:

    result += Counter(data)



result = pd.DataFrame.from_dict(result,orient='index',columns=['Count']).sort_values('Count')

result
fig = px.bar(result,x='Count',y=result.index,color=result.index,labels={'y':'Combination'},title='Synergies Combination Usage',orientation='h',height=1000)

fig.layout.update(showlegend=False)

fig.show()
class_and_origin_top = master_df[master_df.Ranked <= 3].combination.apply(lambda x: re.findall(r'[a-zA-Z]+[0-9]?_?[a-zA-Z]+',x)).to_frame()

result = Counter()

for data in class_and_origin_top.combination:

    result += Counter(data)



result = pd.DataFrame.from_dict(result,orient='index',columns=['Count']).sort_values('Count')

result
fig = px.bar(result,x='Count',y=result.index,color=result.index,labels={'y':'Combination'},title='Synergies Combination Usage Among Top 3',orientation='h',height=1000)

fig.layout.update(showlegend=False)

fig.show()
popular_comp = master_df.combination.value_counts().to_frame()



popular_comp.head(5)
top_comp = master_df[master_df.Ranked <= 3].combination.value_counts().to_frame()

top_comp.head()