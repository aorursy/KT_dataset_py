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
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
xl = pd.read_excel('/kaggle/input/ipl-data-set/Players.xlsx')
print(xl)
teams = pd.read_csv("/kaggle/input/ipl-data-set/teams.csv")
deliveries = pd.read_csv("/kaggle/input/ipl-data-set/deliveries.csv")
matches = pd.read_csv("/kaggle/input/ipl-data-set/matches.csv",parse_dates=['date'])
teamwise_home_and_away = pd.read_csv("/kaggle/input/ipl-data-set/teamwise_home_and_away.csv")
most_runs_average_strikerate = pd.read_csv("/kaggle/input/ipl-data-set/most_runs_average_strikerate.csv")
print(teams.info())
print("No. of teams: ",teams['team1'].nunique())
teams['team1'].unique()
print(deliveries.info())
deliveries.tail()
print(matches.info())
matches.head()
matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)

deliveries.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)

print("No. of Umpires 1: ",matches['umpire1'].nunique())
print("No. of Umpires 2: ",matches['umpire2'].nunique())
print("No. of Umpires 3: ",matches['umpire3'].nunique())

ump_set1 = set(matches['umpire1'].unique())               
ump_set2 = set(matches['umpire2'].unique())
ump_set3 = set(matches['umpire3'].unique())
all_set = ump_set1.intersection(ump_set2)
all_set = all_set.intersection(ump_set3)
print("Umpires who umpired as 1st,2nd and 3rd umpires: ",all_set, len(all_set))
plt.subplots(figsize=(14,6))
ax=matches['umpire1'].value_counts().plot.bar(width=0.9,color=sns.color_palette('bright',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.xlabel("Umpires", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.title("Umpires-1 who have umpired most (from highest to lowest)", fontsize=20)
plt.show()
plt.subplots(figsize=(14,6))
ax=matches['umpire2'].value_counts().plot.bar(width=0.9,color=sns.color_palette('pastel',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.xlabel("Umpires", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.title("Umpires-2 who have umpired most (from highest to lowest)", fontsize=20)
plt.show()
plt.subplots(figsize=(14,6))
ax=matches['umpire3'].value_counts().plot.bar(width=0.9,color=sns.color_palette('Blues'))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.xlabel("Umpires", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.title("Umpires-3 who have umpired most (from highest to lowest)", fontsize=20)
plt.show()
plt.subplots(figsize=(10,6))
ax=matches['toss_winner'].value_counts().plot.bar(width=0.9,color=sns.color_palette('RdYlGn',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.title("Teams that won the toss (from highest to lowest)", fontsize=20)
plt.xlabel("Teams", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()
plt.subplots(figsize=(10,6))
sns.countplot(x='Season',hue='toss_decision',data=matches ,palette=sns.color_palette('bright'))
plt.title("Decision to field or bat across seasons")
plt.show()
plt.subplots(figsize=(10,6))
sns.countplot(x='Season',data=matches,palette=sns.color_palette('colorblind'))  #countplot automatically counts the frequency of an item
plt.title("Number of matches played across Seasons")
plt.show()
pm = matches.groupby(['player_of_match'])['id'].count().reset_index('player_of_match').rename(columns={'player_of_match':'player','id':'count'})#.sort_values(ascending=False)
pm = pm.sort_values(by="count",ascending=False)
top_pm=pm[:10]

fig = go.Figure(data=[go.Scatter(
    x=top_pm['player'], y=top_pm['count'],
    mode='markers',
    marker=dict(
        color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',
               'rgb(44, 160, 101)', 'rgb(255, 65, 54)','rgb(92, 65, 54)','rgb(150, 65, 54)','rgb(30, 165, 54)',
              'rgb(100, 180, 120)', 'rgb(200, 90, 89)', 'rgb(225, 78, 124)'],
        opacity=[1, 0.9, 0.8,0.7, 0.6,0.5,0.45,0.4,0.35,0.3],
        size=[100, 90, 80, 70,60,50,40,30,20,10],
    )
)])
fig.update_layout(
    title="Players who recieved 'Player of Match' Award most",
    xaxis=dict(
        title='Players',        
    ),
    yaxis=dict(
        title='Number',       
    ))
fig.show()
print("Total number of Cities played: ",matches['city'].nunique())
print("Total number of Venues played: ",matches['venue'].nunique())
plt.subplots(figsize=(10,15))
ax = matches['venue'].value_counts().sort_values(ascending=True).plot.barh(width=.9,color=sns.color_palette('inferno',40))
ax.set_xlabel('Grounds')
ax.set_ylabel('count')
plt.title("Venues played (from most to least)")
plt.show()
cities = matches.groupby(['Season','city'])['id'].agg('count').reset_index()
cities.rename(columns={'id':'count'}, inplace=True)

fig = px.bar(cities, x="city", y="count", color='Season')
fig.show()

print(matches.columns)

not_same = matches[matches['toss_winner'] != matches['winner']]
same = matches[matches['toss_winner'] == matches['winner']]
print("Percentage of matches where toss winner is not same as winner: ",round(not_same.shape[0]/matches.shape[0],2) *100)
print("Percentage of matches where toss winner is same as winner: ", round(same.shape[0]/matches.shape[0],2) * 100)
toss_winner = pd.DataFrame({'result':['Yes','No'],'per':[same.shape[0], not_same.shape[0]] })
print(" = " * 40)
field = matches[matches['toss_decision'] == 'field']
bat = matches[matches['toss_decision'] == 'bat']
print("Percentage of matches where toss decision is 'field': ",round(field.shape[0]/matches.shape[0],2) *100)
print("Percentage of matches where toss decision is 'bat': ",round(bat.shape[0]/matches.shape[0],2) *100)
print(" = " * 40)
normal = matches[matches['result'] == 'normal']
tie = matches[matches['result'] == 'tie']
no_result = matches[matches['result'] == 'no result']
print("Percentage of matches where result is 'normal': ",round(normal.shape[0]/matches.shape[0],2) *100)
print("Percentage of matches where result is 'tie': ",round(tie.shape[0]/matches.shape[0],2) *100)
print("Percentage of matches where result is 'no result': ",round(no_result.shape[0]/matches.shape[0],2) *100)
result = pd.DataFrame({'Result':['Normal','Tie','No Result'],'per':[normal.shape[0], tie.shape[0], no_result.shape[0]] })
print(" = " * 40)
dl_applied_no = matches[matches['dl_applied'] == 0]
dl_applied_yes = matches[matches['dl_applied'] == 1]
dl = pd.DataFrame({'dl_applied':['yes','no'],'per':[dl_applied_yes.shape[0], dl_applied_no.shape[0]] })
print("Percentage of matches where DL is applied : ",round(dl_applied_yes.shape[0]/matches.shape[0],2) *100)
print("Percentage of matches where DL is not applied : ",round(dl_applied_no.shape[0]/matches.shape[0],2) *100)

# Pie Chart for Whether toss winner is same as match winner
fig = px.pie(toss_winner, values='per', names='result', color='result', title='Is Match winner same as toss winner?'
             ,color_discrete_map={'Yes':'lightcyan',
                                 'No':'royalblue' })
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

# Pie Chart for how many matches DL is applied
fig = px.pie(dl, values='per', names='dl_applied', title='Percentage of matches where DL is applied', color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

labels = result['Result']
values = result['per']

# Pie Chart for results of the matches played
fig = go.Figure(data=[go.Pie(labels=labels,title='Result of matches', values=values, pull=[0, 0.2, 0])])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
matches['date'].min(), matches['date'].max()
teamwise_home_and_away.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)
print(teamwise_home_and_away.info())
teamwise_home_and_away.head()
fig = go.Figure(data=[
    go.Bar(name='Home Win', x=teamwise_home_and_away['team'], y=teamwise_home_and_away['home_win_percentage']),
    go.Bar(name='Away Win', x=teamwise_home_and_away['team'], y=teamwise_home_and_away['away_win_percentage'])
])

fig.update_layout(barmode='group',title="Team wise - Home/Away wins")
fig.show()
most_runs_average_strikerate.info()
most_runs_average_strikerate.head(10)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

top15 = most_runs_average_strikerate[:15]

fig = go.Figure()
fig.add_trace(go.Bar(
   y=top15['batsman'],
    x=top15['out'],
    name='No. of Matches',
    orientation='h',
    marker=dict(
        color='rgba(80, 100, 67, 0.6)',
        line=dict(color='rgba(8, 1, 212, 1.0)', width=3)
    )
))
fig.add_trace(go.Bar(
   y=top15['batsman'],
    x=top15['strikerate'],
    name='Strike Rate',
    orientation='h',
    marker=dict(
        color='rgba(8, 1, 212, 0.6)',
        line=dict(color='rgba(8, 1, 212, 1.0)', width=3)
    )
))
fig.add_trace(go.Bar(
   y=top15['batsman'],
    x=top15['average'],
    name='Average Runs',
    orientation='h',
    marker=dict(
        color='rgba(158, 5, 19, 0.6)',
        line=dict(color='rgba(158, 5, 19, 1.0)', width=3)
    )
))

fig.update_layout(barmode='stack',title="Players - No. of matches, Strike Rate, Average Runs")
fig.show()
plt.subplots(figsize=(8,6))
b = deliveries.groupby(['batsman'])['batsman_runs'].sum()#.sort_values('batsman_runs')
b = b.sort_values(ascending=False)
b[100:200]

ax=b.sort_values(ascending=False)[:10].plot.bar(width=0.8,color=sns.color_palette('husl',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.1, p.get_height()+50),fontsize=15)
plt.show()
top_batsman = deliveries.groupby(['batsman','batsman_runs'])['total_runs'].count().reset_index()
top_batsman = top_batsman.pivot('batsman','batsman_runs','total_runs')

fig,ax=plt.subplots(3,2,figsize=(18,12))
top_batsman[1].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[0,0],color='#45ff45',width=0.8)
ax[0,0].set_title("Most 1's")
ax[0,0].set_ylabel('')
top_batsman[2].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[0,1],color='#df6dfd',width=0.8)
ax[0,1].set_title("Most 2's")
ax[0,1].set_ylabel('')
top_batsman[4].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[1,0],color='#fbca5f',width=0.8)
ax[1,0].set_title("Most 4's")
ax[1,0].set_ylabel('')
top_batsman[6].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[1,1],color='#ffff00',width=0.8)
ax[1,1].set_title("Most 6's")
ax[1,1].set_ylabel('')
top_batsman[0].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[2,0],color='#abcd00',width=0.8)
ax[2,0].set_title("Most 0's")
ax[2,0].set_ylabel('')
top_batsman[7].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[2,1],color='#f0debb',width=0.8)
ax[2,1].set_title("Most 7's")
ax[2,1].set_ylabel('')
plt.show()
top_scorers = deliveries.groupby(["match_id", "batsman","batting_team"])["batsman_runs"].sum().reset_index()
top_scorers.sort_values('batsman_runs', ascending=0).head(10)
top_scorers.nlargest(10,'batsman_runs')
batsmen = matches[['id','Season']].merge(deliveries, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)
#merging the matches and delivery dataframe by referencing the id and match_id columns respectively
season=batsmen.groupby(['Season'])['total_runs'].sum().reset_index()
season.set_index('Season').plot(marker='*')
plt.gcf().set_size_inches(10,8)
plt.title('Total Runs Across the Seasons')
plt.show()
men = batsmen.groupby(['Season','batsman'])['batsman_runs'].sum().reset_index()
men = men.groupby(['Season','batsman'])['batsman_runs'].sum().unstack().T
men['Total'] = men.sum(axis=1)
men = men.sort_values(by='Total',ascending=False)[:5]
men.drop('Total',axis=1,inplace=True)
men.T.plot(color=['red','skyblue','#772272','brown','limegreen'],marker='*')
fig=plt.gcf()
fig.set_size_inches(16,8)
plt.show()
Season_boundaries=batsmen.groupby("Season")["batsman_runs"].agg(lambda x: (x==6).sum()).reset_index()
a=batsmen.groupby("Season")["batsman_runs"].agg(lambda x: (x==4).sum()).reset_index()
Season_boundaries=Season_boundaries.merge(a,left_on='Season',right_on='Season',how='left')
Season_boundaries=Season_boundaries.rename(columns={'batsman_runs_x':'6"s','batsman_runs_y':'4"s'})
Season_boundaries.set_index('Season')[['6"s','4"s']].plot(marker='o')
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()
matches_played_byteams=pd.concat([matches['team1'],matches['team2']])
matches_played_byteams=matches_played_byteams.value_counts().reset_index()
matches_played_byteams.columns=['Team','Total Matches']

matches_played_byteams['wins']=matches['winner'].value_counts().reset_index()['winner']

matches_played_byteams.set_index('Team',inplace=True)

runs_per_over = deliveries.pivot_table(index=['over'],columns='batting_team',values='total_runs',aggfunc=sum)
runs_per_over[(matches_played_byteams[matches_played_byteams['Total Matches']>50].index)].plot(color=["b", "r", "#Ffb6b2", "g",'brown','y','#6666ff','black','#FFA500']) #plotting graphs for teams that have played more than 100 matches
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.xticks(x)
plt.ylabel('total runs scored')
fig=plt.gcf()
fig.set_size_inches(16,10)
plt.show()
high_scores=deliveries.groupby(['match_id', 'inning','batting_team','bowling_team'])['total_runs'].sum().reset_index() 
#reset_index() converts the obtained series into a dataframe
high_scores=high_scores[high_scores['total_runs']>=200]
#nlargest is used to sort the given column
high_scores.nlargest(10,'total_runs')