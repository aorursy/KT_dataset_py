# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed

df = pd.read_csv('../input/zomato-restaurants-data/zomato.csv',
                 encoding = "ISO-8859-1")

df1 = pd.read_excel('../input/zomato-restaurants-data/Country-Code.xlsx')
print(df.head())
df1.head()
df3 = pd.merge(df, df1, on=('Country Code'))
df3['Country'].value_counts().head(5)
df.groupby('Cuisines')['Aggregate rating'].mean().plot(kind='box') 
Par1=['Price range','Average Cost for two','Aggregate rating']
t1=df[Par1].corr(method='spearman')
sns.heatmap(t1);
print(df[df['Has Table booking'] == 'No']['Aggregate rating'].mean())
print(df[df['Has Table booking'] == 'Yes']['Aggregate rating'].mean())
print('Есть, причем сильные')
BBox = ((df.Longitude.min(),   df.Longitude.max(),      
         df.Latitude.min(), df.Latitude.max()))
ruh_m = plt.imread('../input/random/1.bmp')
fig, ax = plt.subplots(figsize = (20,30))
dfy=df[df['Rating color'] == "Yellow"]
dfg=df[df['Rating color'] == "Green"]
dfdg=df[df['Rating color'] == "Dark Green"]
dfo=df[df['Rating color'] == "Orange"]
dfr=df[df['Rating color'] == "Red"]
dfw=df[df['Rating color'] == "White"]
ax.scatter(dfy['Longitude'], dfy['Latitude'],color='Yellow')
ax.scatter(dfg['Longitude'], dfg['Latitude'],color='Green')
ax.scatter(dfdg['Longitude'], dfdg['Latitude'],c='DarkGreen')
ax.scatter(dfo['Longitude'], dfo['Latitude'],color='Orange')
ax.scatter(dfr['Longitude'], dfr['Latitude'],color='Red')
ax.scatter(dfw['Longitude'], dfw['Latitude'],color='grey')

ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(ruh_m, zorder=0, extent = BBox)

numeric = ['Longitude','Latitude','Price range','Aggregate rating','Votes']
sns.pairplot(df[numeric]);
t2=df[numeric].corr(method='spearman')
t2
sns.heatmap(t2);