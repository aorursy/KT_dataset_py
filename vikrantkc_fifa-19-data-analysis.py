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
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
df = pd.read_csv("/kaggle/input/fifa19/data.csv")
df
#to remove unnamed:0

df.columns.str.match('Unnamed')

df1 = df.loc[:,~df.columns.str.match('Unnamed')]
df1
df1.shape
df1.columns
df1.info()
df1.isnull().sum()
#using missingno for checking pattern of nan values and hoe nan distributed
#missingno offers very nice way to visulazie the distribustion of nan values
import missingno as msno
msno.matrix(df1)
#bar chart gives idea about how may missing values are there in each columns
msno.bar(df1)
#check for nan values
#club and position column have object dtype
df1[df1['Club'].isnull()]
df1[df['Position'].isnull()]
#replace nan value of club and positon
df1['Club'].fillna('No Club',inplace=True)
df1['Position'].fillna('ST',inplace=True)
#check if there is nan value left after fillna 
df1[df1['Club'].isnull()]
df1[df1['Position'].isnull()]
#using mean values for the float dtype
impute_mean = df1.loc[:,['Crossing', 'Finishing', 'HeadingAccuracy',
                        
                                 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',
                        
                                 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',
                        
                                 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',
                        
                                 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',
                        
                                 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',
                        
                                 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                        
                                 'GKKicking', 'GKPositioning', 'GKReflexes']]
for i in impute_mean.columns:
    df1[i].fillna(df1[i].mean(),inplace=True)
#checking after imputing 
df1[df1['Crossing'].isnull()]
#now checking for categorical values and imputed by mode 
#note that using mode() will return series(0,1 or more) but using mode()[0] will no return series
impute_mode  = df1.loc[:,['Body Type','International Reputation','Height',
                         'Weight','Preferred Foot','Jersey Number']]

for i in impute_mode.columns:
    df1[i].fillna(df1[i].mode()[0],inplace=True)
#now checking
df1[df1['Body Type'].isnull()]

#fillna for numerical or continuous numerical
impute_median = df1.loc[:,['Weak Foot','Skill Moves']]
for i in impute_median.columns:
    df1[i].fillna(df1[i].median(),inplace=True)
#checking
df1[df1['Weak Foot'].isnull()]
df1.columns[df1.isna().any()]
impute_mode1 = df.loc[:,['Work Rate', 'Real Face', 'Joined', 'Loaned From',
       'Contract Valid Until', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Release Clause']]
for i in impute_mode1.columns:
    df1[i].fillna(df1[i].mode()[0], inplace=True)
df1.columns[df1.isna().any()]
# correlation between different feature
f,ax = plt.subplots(figsize = (25,15))
ax = sns.heatmap(df1.corr(),annot=True,linewidths=0.5,linecolor='red',fmt='.1f',ax=ax)
plt.show()
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(background_color = 'white',
                     width = 1900,height=1000).generate(" ".join(df1.Nationality))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
plt.style.use('fivethirtyeight')
fig,ax = plt.subplots(figsize = (8,7))
graph = sns.countplot(ax=ax,x=df1['Preferred Foot'],data=df1,palette='bone')
graph.set_title('preferred Foot of the Player',fontsize=20)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
plt.show()
#different positions acquired by the players
plt.figure(figsize=(18,8))
plt.style.use('fivethirtyeight')
ax = sns.countplot('Position',data=df1,palette='bone')
ax.set_xlabel(xlabel = 'Different positon in football',fontsize=16)
ax.set_ylabel(ylabel='Count of Players',fontsize=18)
ax.set_title(label = 'Comparison of Position and Players',fontsize=20)
plt.show()
plt.figure(figsize=(13,9))
ax = sns.countplot(x='Height',data=df1,palette='dark')
ax.set_title(label = 'Count of player on basis of height',fontsize = 20)
ax.set_xlabel(xlabel='Height in foot per inch',fontsize=17)
ax.set_ylabel(ylabel='Count',fontsize=17)
plt.show()
#show diffrent work rate of player participating in fifa 2019
plt.figure(figsize=(15,7))
plt.style.use('fivethirtyeight')
sns.countplot(x='Work Rate',data = df1,palette='hls')
plt.title('diffrent work rates of the players participating in fifa 2019',fontsize=20)
plt.xlabel('work rates associated with the players')
plt.ylabel('count of players')
plt.show()
#shows players age
sns.set(style='dark',palette='colorblind',color_codes=True)
x=df1.Age
plt.figure(figsize=(15,8))
plt.style.use('ggplot')
ax = sns.distplot(x,bins=58,kde=False,color='g')
ax.set_xlabel(xlabel='Player\'s age',fontsize=17)
ax.set_ylabel(ylabel='number of players',fontsize=20)
ax.set_title(label='histogram of player age',fontsize=20)
plt.show()
fig,ax = plt.subplots(figsize=(12,8))
plt.style.use('fivethirtyeight')
graph = sns.countplot(ax=ax,x = df1['Skill Moves'],data=df1,hue='Preferred Foot',palette='PuBuGn_d')
graph.set_title('skill moves of players segregate by preferred foot',fontsize=24)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2.,height+0.1,height,ha='center')
top_country = df1['Nationality'].value_counts().head(15)
top_country
fig,ax = plt.subplots(figsize=(12,8))
x = top_country.values
y = top_country.index
ax.barh(y,x,align='center',color='dimgray')
ax.invert_yaxis()
ax.set_xlabel('number of players')
ax.set_ylabel('name of countries',rotation=0)
ax.set_title('top 10 countries with most number of player')
plt.show()
df1_best_players = pd.DataFrame.copy(df1.sort_values(by='Overall',ascending=False).head(10))
plt.figure(1,figsize=(12,6))
sns.barplot(x='Name',y='Overall',data=df1_best_players,palette='Reds')
plt.ylim(85,95)
plt.show()
### top 10 eldest players 
df1.sort_values(by='Age',ascending=False)[['Name','Club','Nationality','Overall','Age']].head()
### TOP 10 YOUNGEST PLAYERS
df1.sort_values(by='Age',ascending=True)[['Name','Club','Nationality','Overall','Age']].head()
### best finisher
df1.sort_values(by='Finishing',ascending=False)[['Name','Club','Nationality','Age','Finishing']].head()
### fastest players
df1.sort_values(by='SprintSpeed',ascending=False)[['Name','Club','Nationality','Overall','Age','SprintSpeed']].head()
###best dribbler
df1.sort_values(by='Dribbling',ascending=False)[['Name','Club','Nationality','Overall','Age','Dribbling']].head()
player = str(df1.loc[df1['Potential'].idxmax()][1])
print('Maximum Overall Perforamnce : '+str(df1.loc[df1['Overall'].idxmax()][1]))
pr_cols=['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
i=0
while i < len(pr_cols):
    print('Best {0} : {1}'.format(pr_cols[i],df1.loc[df1[pr_cols[i]].idxmax()][1]))
    i += 1
i=0
best=[]
while i<len(pr_cols):
    best.append(df1.loc[df1[pr_cols[i]].idxmax()][1])
    i += 1
plt.subplots(figsize=(25,15))
wordcloud =WordCloud(background_color='White',width=1920,height=900).generate(" ".join(best))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()