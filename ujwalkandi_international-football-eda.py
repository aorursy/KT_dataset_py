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
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/international-football-results/results.csv')
df.head()
df.sample(5)
df.corr()
for i,col in enumerate(df.columns):
    print((i+1),'-',col)
away_team=pd.DataFrame(df.groupby('away_team')['home_score'].count().index)
away_score=pd.DataFrame(df.groupby('away_team')['home_score'].count().values,columns=['Score'])
away_score_team=pd.concat([away_team,away_score],axis=1)
plt.figure(figsize=(12,8))
away_score_team=away_score_team.sort_values(by='Score',ascending=False)
sns.barplot(x=away_score_team.away_team[:50],y=away_score_team.Score[:50])
plt.xticks(rotation=90)
plt.show()
df['tournament'].unique()
df['tournament'].value_counts()
plt.figure(figsize=(12,8))
sns.barplot(x=df['tournament'].value_counts().index[:20],y=df['tournament'].value_counts().values[:20])
plt.xticks(rotation=90)
plt.ylabel("Count")
plt.xlabel("Tournament")
plt.title("Total number of Tournaments played")
plt.show()
tournament=df['tournament'].value_counts()
names=tournament.index
values=tournament.values
plt.figure(figsize=(10,10))
sns.barplot(x=names[:10],y=values[:10])
plt.xticks(rotation=90)
plt.ylabel('Values')
plt.xlabel('Tournament')
plt.title('Tournament vs Values How Play in the World')
plt.show()
matches = df.astype({'date':'datetime64[ns]'})
tournament = matches['tournament'].value_counts()
tournament = tournament[:15]

plt.figure(figsize = (15,10))
ax = sns.barplot(y=tournament.index, x=tournament.values, orient='h')
ax.set_ylabel('Tournament', size=16)
ax.set_xlabel('Number of tournament', size=16)
ax.set_title("TOP 15 TYPE OF MATCH TOURNAMENTS", fontsize=18)
plt.show()
home_name_index=df.home_team.value_counts()
home_name_index=home_name_index.head(10)

plt.figure(figsize=(10,8))
ax=sns.barplot(x=home_name_index.index,y=home_name_index.values,palette=sns.cubehelix_palette(len(home_name_index.index)))
plt.xlabel('Home Team Name')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.title('Most common 100 Home Team Name')
plt.show()
veri=df[df['tournament']=='FIFA World Cup'].tail(56)

len(veri['home_team'].unique())
allteam=veri['home_team'].unique()
allteam
away_scores_allteam=[]
home_scores_allteam=[]
for team in allteam:
    toplam=sum(veri[veri['home_team']==team].away_score)
    away_scores_allteam.append(toplam)
    home_scores_allteam.append(sum(veri[veri['home_team']==team].home_score))
    toplam=0


all_team=pd.DataFrame([allteam,home_scores_allteam,away_scores_allteam])
   
plt.figure(figsize=(12,8))
sns.barplot(x=allteam,y=home_scores_allteam)
plt.title('FIFA CUP 2018 Home Scores',color='b',fontsize=15)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(12,8))
sns.barplot(x=allteam,y=away_scores_allteam)
plt.title('FIFA CUP 2018 Away Scores',color='b',fontsize=15)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(veri.city)
plt.xlabel('City')
plt.xticks(rotation=90)
plt.title('City List (Russia)',color='blue',fontsize=15)
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(veri.city,hue=veri.neutral)
plt.xticks(rotation=90)
plt.title('City for Hue Neutral')
plt.show()