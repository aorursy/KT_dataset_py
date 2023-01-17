import numpy as np 
import pandas as pd
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from plotly.offline import init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import plotly.plotly as py
import plotly.graph_objs as go


train = pd.read_csv('../input/train_V2.csv')
train.sample(5)
train.info()
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,center=0)
plt.show()
corr_with_target = train.drop(['Id', 'matchId', 'groupId', 'winPlacePerc'], 1).corrwith(train['winPlacePerc'])
corr_with_target

combat_features=['assists',
'damageDealt',
'DBNOs' ,
'headshotKills',
'kills',
'killStreaks',
'vehicleDestroys',
'weaponsAcquired',
'longestKill']
combat_with_target = train[combat_features].corrwith(train['winPlacePerc'])
trace_combat = go.Scatter(
    x = combat_with_target.index,
    y = combat_with_target.values,
    mode = 'markers+lines',
    marker = dict(
        size = 10,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 2
        )))
data_combat = [trace_combat]
layout = {'title': 'Correlation between winPlacePerc and combat attributes',
          'yaxis' : {'title' : 'winPlacePerc'},
          'xaxis' : {'tickangle' : 45}}


fig = go.Figure(data=data_combat, layout=layout)
iplot(fig, show_link=False)



print("**** Weapons Acquired Stats ****")
print("Average person acquires {:f} weapons per match".format(train['weaponsAcquired'].mean()))
print("Maximum weapons ever recorded in a match is  :{:f}".format(train['weaponsAcquired'].max()))
plt.figure(figsize=(15,10))
plt.title("weaponsAcquired",fontsize=15)
sns.distplot(train['weaponsAcquired'])
data_fist_attack=train.copy()
plt.figure(figsize=(15,10))
data_fist_attack=data_fist_attack[data_fist_attack.weaponsAcquired==0]
data_fist_attack_grouped=data_fist_attack.groupby('kills')['damageDealt'].mean()
plt.title("Damage and kills by players without weapons")
data_fist_attack_grouped.plot()
plt.show()
print("**** Kill Streaks Stats ****")

print("Average Kill streaks per match is {:f}.".format(train['killStreaks'].mean()))
print("highest kill streaks ever recorded in a match is  {:f}".format(train['killStreaks'].max()))
sns.distplot(train['killStreaks'])
sns.jointplot(x="winPlacePerc", y="killStreaks",  data=train,  ratio=3, color="blue")
plt.show()
plt.figure(figsize=(15,10))
plt.title("longestKill",fontsize=15)
sns.distplot(train['longestKill'])
print("**** Longest Kill Stats ****")
print("longest kill ever recorded was :{:f} meters".format(train['longestKill'].max()))
print("Average shot is about {:f} meters".format(train['longestKill'].mean()))
train_headshotkills_temp=train.copy()
train_headshotkills_temp.loc[train_headshotkills_temp['headshotKills']>4]='4+'
sns.countplot(x=train_headshotkills_temp['headshotKills'].astype('str').sort_values())

print("****HeadShot Kills Stats****")
print("Most recorded headshots ever in a match is :{:f}".format(train['headshotKills'].max()))
print("83% of people have {} headshots".format(train['headshotKills'].quantile(0.83)))
print("Avergare Headshots:{:f}".format(train['headshotKills'].mean()))
data_headshot0 = train.copy()
data_headshot0 = data_headshot0[data_headshot0['headshotKills']==0]
sns.jointplot(x="winPlacePerc", y="kills",  data=data_headshot0,  ratio=3, color="blue")
plt.show()
train_temp=train.copy()
train_temp.loc[train_temp['kills']>7]='7+'
train_temp.groupby(['kills']).groups.keys()
plt.figure(figsize=(15,10))
sns.countplot(y=train_temp['kills'].astype('str').sort_values())
print("****Kills Stats*****")

print("Most num of kills  ever recorded in a match is :{:f}".format(train['kills'].max()))
print("90% of people have {} kills".format(train['kills'].quantile(0.99)))
print("Strangly last 10% of the data have a drastic increase in kill count, which is  {}".format(train['kills'].quantile(0.99)))
print("Avergare kills:{:f}".format(train['kills'].mean()))
kills = train.copy()

kills['killsCategories'] = pd.cut(kills['kills'], [-1, 0, 2, 5, 10, 60], labels=['0_kills','1-2_kills', '3-5_kills', '6-10_kills', '10+_kills'])
sns.barplot(x="winPlacePerc", y="killsCategories",
              data=kills)

train_temp2=train.copy()
var=train_temp2[train_temp2.kills<2]

sns.jointplot(x=var['winPlacePerc'], y=var['kills'],  data=var,  ratio=3, color="blue")
plt.show()

train_temp3=train.copy()
var3=train_temp3[train_temp3.kills<2]
var4 = var3.corr()['winPlacePerc'].nlargest(5)
var4=var4.drop(labels=['winPlacePerc' ])

trace3 = go.Scatter(
    x = var4.index,
    y = var4.values,
    mode = 'markers+lines',
    marker = dict(
        size = 10,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 2
        )))

data2 = [trace3]

layout = {'title': '0 - 1 kills Correlation with Target(top 5)',
          'yaxis' : {'title' : 'winPlacePerc'},
          'xaxis' : {'tickangle' : 45}}

fig = go.Figure(data=data2, layout=layout)
iplot(fig, show_link=False)
        
damage = train.copy()
damage_bins=[-1,0,200,400,600,1000,3000,6616]
damage_groups=["0","1-200","200-400","400-600","600-1000","1000-3000","3000-6616"]
damage['damageCategories'] = pd.cut(damage['damageDealt'],damage_bins,labels=damage_groups)
plt.figure(figsize=(15,10))
kill_bins=[-1, 0, 2, 5, 10, 60]
kill_groups=['0_kills','1-2_kills', '3-5_kills', '6-10_kills', '10+_kills']
damage['killsCategories'] = pd.cut(damage['kills'],kill_bins, labels=kill_groups)      
sns.barplot(x="damageCategories", y="winPlacePerc",
              data=damage,hue='killsCategories')
print("Max damage dealt ever :{:f}".format(train['damageDealt'].max()))
print("50% of people have dealt: {} ".format(train['damageDealt'].quantile(0.50)))
print("min damage dealt :  {}".format(train['kills'].min()))
print("Avergare damage:{:f}".format(train['damageDealt'].mean()))
train_temp4=train.copy()
kills_wins = ['kills',
              'DBNOs',              
'headshotKills',
'heals',     
'killPlace',       
'killPoints',        
'kills',        
'killStreaks',
'longestKill']
kills_wins_with_damage = train[kills_wins].corrwith(train['damageDealt'])

trace6 = go.Bar(
    x=kills_wins_with_damage.index,
    y=kills_wins_with_damage.values,
     marker=dict(
        color='rgb(49,130,189)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),
    opacity=0.6
)

data = [trace6]

layout = {'title': '0 - 1 kills Correlation with Target',
          'yaxis' : {'title' : 'Combat co-relation with Damage'},
          'xaxis' : {'tickangle' : 45}}

fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
train_temp4=train.copy()
kills_wins = ['kills','damageDealt']
kills_wins_with_damage = train[kills_wins].corrwith(train['winPlacePerc'])

trace6 = go.Bar(
    x=kills_wins_with_damage.index,
    y=kills_wins_with_damage.values,
     marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),
    opacity=0.6
)

data = [trace6]

layout = {'title': 'kills vs Damage Correlation with Target',
          'yaxis' : {'title' : 'Damage Dealth Co-relations'},
          'xaxis' : {'tickangle' : 45}}

fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)

survival_features=[
'heals',
'boosts',]

survival_with_target = train[survival_features].corrwith(train['winPlacePerc'])
    
trace = go.Scatter(
    x = survival_with_target.index,
    y = survival_with_target.values,
    mode = 'markers+lines',
    marker = dict(
        size = 20,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 2
        )))
data = [trace]
layout = {'title': 'Correlation between winPlacePerc and survival attributes',
          'yaxis' : {'title' : 'winPlacePerc'},
          'xaxis' : {'tickangle' : 45}}

# Display it
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
print("**** Boost and Heal Stats ****")
print("The average person uses {:.1f} heal items".format(train['heals'].mean()))
print("99% of people use {} or less,".format(train['heals'].quantile(0.99)) )
print("Maximum heal items ever used in a match {:.1f} heal items".format(train['heals'].max()))

print("The average person uses {:.1f} boost items".format(train['boosts'].mean()))
print("99% of people use {} or less,".format(train['boosts'].quantile(0.99)) )
print("Maximum heal items ever used in a match {:.1f} heal items".format(train['boosts'].max()))

print("**** WalkDistance Stats ****")
print("The average person runs/walks {:.1f} meters".format(train['walkDistance'].mean()))
print("Maximum distance covered on foot is {:.1f} meters".format(train['walkDistance'].max()))

print("**** Swimming Stats ****")
print("The average person swims {:.1f} meters".format(train['swimDistance'].mean()))
print("Maximum swim distance covered in a match is  {:.1f} meters ".format(train['swimDistance'].max()))

print("**** Ride Distance Stats ****")
print("An average person covers : {:.2f} meters on a vehicle".format(train['rideDistance'].mean()))
print("Max distance covered on vehicle in a game is  : {} meters".format(train['rideDistance'].max()))

distance_covered_features=['rideDistance','swimDistance','walkDistance']

distance_covered_with_target = train[distance_covered_features].corrwith(train['winPlacePerc'])
    
trace = go.Scatter(
    x = distance_covered_with_target.index,
    y = distance_covered_with_target.values,
    mode = 'markers+lines',
    marker = dict(
        size = 20,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 2
        )))
data = [trace]
layout = {'title': 'Correlation between winPlacePerc and distance covered attributes',
          'yaxis' : {'title' : 'winPlacePerc'},
          'xaxis' : {'tickangle' : 45}}

# Display it
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
plt.figure(figsize=(15,10))
df1= pd.pivot_table(train, values=['walkDistance','swimDistance','rideDistance'],index='winPlacePerc')
g = sns.pairplot(df1)
data_teams=train.copy()
solos=data_teams[data_teams['numGroups']>50]
duos = data_teams[(data_teams['numGroups']>25) & (data_teams['numGroups']<=50)]
squads = data_teams[data_teams['numGroups']<=25]
print("Out of total {} games,there are {} ({:.2f}%) solo games, {} ({:.2f}%) duo games and {} ({:.2f}%) squad games.".format(len(data_teams),len(solos), 100*len(solos)/len(data_teams), len(duos), 100*len(duos)/len(data_teams), len(squads), 100*len(squads)/len(data_teams),))
sns.lmplot(x='kills', y='winPlacePerc', data=solos,fit_reg=False)
ax = plt.gca()
ax.set_title("Solo Games ")

sns.lmplot(x='kills', y='winPlacePerc', data=duos,fit_reg=False)
ax = plt.gca()
ax.set_title("Duo Games")

sns.lmplot(x='kills', y='winPlacePerc', data=squads,fit_reg=False)
ax = plt.gca()
ax.set_title("Squad Games ")