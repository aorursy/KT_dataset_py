# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Import basic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data using pandas packahe
data= pd.read_csv('../input/matches.csv')
# Information & Structure
print('The datatset information is provided below with different features and their type. \n')
print(data.info())
print('There are %d rows and %d columns in the imported IPL matches dataset.'%(data.shape[0], data.shape[1]))
# Dataset Summary
print('The numerical summary of the non-object features is provided below: \n')
data.describe()
# Glimpse of the dataset
print('First 5 rows of the dataset for getting a view of the dataset. \n')
data.head(5)
# Drop ID column which is not relevant for analysis.
data.drop('id',axis= 1, inplace= True)
# No of seasons
a=np.sort(data['season'].unique())
print(a)
print('%d seasons have been played in IPL.' %(len(a)))
# Name of Teams participated since IPL's inception
print('The teams which have participated since IPL first edition. \n')
b=np.sort(data['team1'].unique())
print(b)
# Rising Pune SuperGiant has been misspelled 
data.replace(['Rising Pune Supergiants'], ['Rising Pune Supergiant'], inplace= True)
b_updated=np.sort(data['team1'].unique())
print(b_updated)
print('%d teams have played since the incpetion of the league.'%(len(b_updated)))
# Cities where IPL matches have been played.
print('The cities where IPL matches have been played. \n')
c= pd.DataFrame(data.groupby('city')['city'].count())
c.columns=['Count']
c=c.reset_index()
max_match=c['Count'].max()
city= np.array(c[c['Count']==max_match]['city'])
print('%d matches have been played at %s which is highest for any city. \n'%(max_match,city))
print(c)

# Total Matches played by each team since inception of IPL.
print('Total number of matches played by each team since first edition. \n')
d= pd.DataFrame(data.groupby('team1')['team2'].count())
e= pd.DataFrame(data.groupby('team2')['team2'].count())
f=d+e
f=f.reset_index()
f.columns=['Team','Count']
print(f)
# Toss won by each team
print('The team which has won toss in each game. \n')
g= pd.DataFrame(data.groupby('toss_winner')['toss_winner'].count())
g.columns=['Count']
g=g.reset_index()
print(g)
# Decision taken by each team after winning toss
print('The toss decision taken after winning toss for each match. \n')
h=data['toss_decision'].unique()
print(h)
data.groupby(['toss_winner','toss_decision'])['toss_decision'].count()
#Result Statistics. Normal means the match was completed with result. Tied match means the team scored
#equal run after super over also. No result implies those matches where game was abandoned.
print('The different result after completion of the game. \n')
i= pd.DataFrame(data.groupby('result')['result'].count())
i.columns= ['Count']
i= i.reset_index()
print(i)
# Matches Won by Team
print('The match winner team detail below: \n')
j= pd.DataFrame(data.groupby('winner')['winner'].count())
j.columns= ['Count']
j= j.reset_index()
j_max= j['Count'].max()
j_winner= np.array(j[j['Count']==j_max]['winner'])
print(j)
print('\n%s has won %d matches which is highest for any team.' %(j_winner, j_max))
#Matches won by teams in each season.
print('The number of matches won by each team in each season. \n')
k= pd.DataFrame(data.groupby(['season','winner'])['winner'].count())
k.columns= ['Count']
k= k.reset_index()
print(k)

# Matches won by Runs and Macthes won by Maximum and Minimum margin
l= data['win_by_runs']
l= l.iloc[l.nonzero()[0]]
m= l.count()
n= l.max()
o= l.min()
p= data[data['win_by_runs']==n]
q= data[data['win_by_runs']==o]
print('There were total %d matches won by run with maximun margin of %d runs and minimum margin of %d run(s). \n'%(m,n,o))
print('\n The detail of match won by maximum margin is given below:\n', p)
print('\n The detail of match won by minimum margin is given below:\n', q)
#Macthes won by wicket and matches won by maximum and minimum margin
r= data['win_by_wickets']
r= r.iloc[r.nonzero()[0]]
s= r.count()
t= r.max()
u= r.min()
v= data[data['win_by_wickets']==t]
w= data[data['win_by_runs']==u]
print('There were total %d matches won by wickets with maximun margin of %d wicket and minimum margin of %d wicket(s). \n'%(s,t,u))
print('\n The detail of match won by maximum margin is given below:\n', v)
print('\n The detail of match won by minimum margin is given below:\n', w)
# Total players count who have won Man of Match Award and player who has won for maximum times
x= pd.DataFrame(data.groupby('player_of_match')['player_of_match'].count())
x.columns=['MOM']
x= x.reset_index()
y=x.count()
z=x['MOM'].max()
mom_player= x[x['MOM']==z]['player_of_match']
print('%d players have been awarded man of the match award since inception of the league.'%(y['MOM']))
print('%s has been awarded MOM award %d times which is highest for any player.  \n'% (np.array(mom_player),z))

# Number of matches played in each session
sns.countplot(x='season', data= data, palette= 'rainbow')
plt.title('Number of Match -  Season')
plt.xlabel('Season')
plt.ylabel('Number of Match')
# Number of Matches played by team
plt.figure(figsize = (5,5))
sns.barplot(y='Team', x= 'Count', data= f, palette= 'copper')
plt.title('Number of Match - Team')
plt.ylabel('Team')
plt.xlabel('Count')
# Number of Toss won by team
plt.figure(figsize = (5,5))
sns.barplot(y='toss_winner', x= 'Count', data= g, palette= 'summer')
plt.title('Toss Won - Team')
plt.ylabel('Team')
plt.xlabel('Count')
# Toss Decision - Seasonwise
plt.figure(figsize = (10,5))
sns.countplot(hue='toss_decision', y='season', data= data, palette= 'inferno')
plt.title('Toss Decision - Season')
plt.ylabel('Toss Decision')
plt.xlabel('Count')
# Toss Decision Team Wise
plt.figure(figsize = (5,5))
sns.countplot(hue='toss_decision', y='toss_winner', data= data, palette= 'autumn')
plt.title('Toss Decision - Toss_Winner')
plt.xlabel('Count')
plt.ylabel('Team Name')
# Result 
plt.figure(figsize = (5,5))
sns.countplot(x='result', data= data, palette= 'viridis')
plt.title('Match- Result')
plt.xlabel('Result Type')
plt.ylabel('Count')
# Matches Result - Team
plt.figure(figsize= (10,10))
sns.countplot(y='winner', hue='result', data= data, palette= 'magma')
plt.title('Team - Result Type')
plt.xlabel('Count')
plt.ylabel('Team')
# Matches won by Team
plt.figure(figsize= (5,5))
sns.countplot(y='winner', data= data, palette= 'magma')
plt.title('Match Won - Team')
plt.xlabel('Count')
plt.ylabel('Team')
# Matches Won by Team Seasonwise
plt.figure(figsize= (10,10))
sns.countplot(y='winner',hue='season', data= data, palette= 'rainbow')
plt.title('Match Won - Team')
plt.xlabel('Count')
plt.ylabel('Team')
# Player of the Match Awardee - Top 20 players 
plt.figure(figsize= (10,5))
xx=x.sort_values('MOM', ascending=False)[:20]
sns.barplot(y='player_of_match', x= 'MOM', data= xx, palette= 'inferno')
plt.title('Player of the Match Awardee - Top 20')
plt.xlabel('Number of Player of the Match Award')
plt.ylabel('Player Name')

