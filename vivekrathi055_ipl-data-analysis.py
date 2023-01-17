import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

sns.set()

import warnings

warnings.filterwarnings("ignore")
ipl_filepath_deliveries = '../input/ipl/deliveries.csv'

ipl_filepath_matches = '../input/ipl/matches.csv'



deliveries_data = pd.read_csv(ipl_filepath_deliveries)

deliveries_data.head()

matches_data = pd.read_csv(ipl_filepath_matches)

matches_data.head()
matches_data.isnull().sum()
matches_data = matches_data.drop('umpire3', axis=1)

matches_data.isnull().sum()
deliveries_data.isnull().sum()
deliveries_data = deliveries_data.fillna('0')

deliveries_data.isnull().sum()
matches_data.columns
matches_data.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings', 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab', 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant'] ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)



deliveries_data.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings', 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab', 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant'] ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)
print('Number of matches played so far : ', matches_data.shape[0])
print('Number of Seasons so far : ', len(matches_data['season'].unique()))
plt.figure(figsize=(10,6))

sns.countplot(x=matches_data['season'], data=matches_data)

plt.ylabel('Number of Matches Played')
temp = matches_data.sort_values('venue', ascending=False)

temp['venue'].value_counts().head(10)



matches_data['venue'].value_counts()
plt.figure(figsize=(15,6))

sns.countplot(x=temp['venue'], data=temp)

plt.xticks(rotation = 'vertical')

plt.ylabel('Number of Matches Played')
temp = matches_data['winner'].value_counts()

print('Most number of Player of Match Awards: ')

print(temp)
plt.figure(figsize=(15,6))

sns.countplot(x=matches_data['winner'], data=matches_data)

plt.xticks(rotation = 'vertical')
matches_data['toss_winner'].value_counts()[:10]
plt.figure(figsize=(15,6))

sns.countplot(x=matches_data['toss_winner'], data=matches_data)

plt.xticks(rotation = 'vertical')

matches_data['toss_decision'].value_counts()
plt.figure(figsize=(10,6))

sns.countplot(x=matches_data['toss_decision'], data=matches_data)

plt.xticks(rotation = 'vertical')

temp = matches_data['city'].value_counts()

print('Matches played at each city:')

print(temp.head(10))
plt.figure(figsize=(15,7))

sns.countplot(x=matches_data['city'], data=matches_data)

plt.xticks(rotation = 'vertical')
matches_data['city'].value_counts()[:10].plot(kind='bar', color='skyblue')
temp = matches_data['umpire1'].value_counts()

print('Number of matches played by different Umpires : ')

print(temp.head(10))
plt.figure(figsize=(15,6))

sns.countplot(x=matches_data['umpire1'], data=matches_data)

plt.xticks(rotation='vertical')
matches_data['umpire1'].value_counts()[:10].plot(kind='bar', color='skyblue')
temp = matches_data['player_of_match'].value_counts().head(10)

print('Most number of Player of Match Awards: ')

print(temp)
matches_data['player_of_match'].value_counts()[:10].plot(kind='bar',color='skyblue')

plt.title('Player of Match')
temp = matches_data.sort_values(['win_by_wickets','date'], ascending=False)

temp = temp[['season','date','team1','team2','winner','win_by_wickets','player_of_match']].head(10)

temp
temp = matches_data.sort_values('win_by_runs', ascending=False)

temp = temp[['season','date','team1','team2','winner','win_by_runs','player_of_match']].head(10)

temp
temp.plot(x='winner', y='win_by_runs', marker='o')
temp.columns
temp = matches_data.sort_values('season', ascending=False)

temp = matches_data.drop_duplicates(subset=['season'], keep='last')[['season', 'winner']].reset_index(drop=True)

temp
print("Toss Decisions : \n",((matches_data['toss_decision']).value_counts()/matches_data.shape[0]*100))
plt.subplots(figsize=(10,6))

sns.countplot(x='season',hue='toss_decision',data=matches_data)

plt.show()
plt.figure(figsize=(12,7))

temp = matches_data.toss_decision.value_counts()

sizes = (np.array((temp / temp.sum())*100))

plt.pie(sizes, labels=(np.array(temp.index)),colors=['lightgreen', 'lightblue'],

        autopct='%1.1f%%',shadow=True, startangle=90,explode=(0,0.03))

plt.title("Toss decision percentage")

plt.show()

          
plt.figure(figsize=(12,7))

temp = matches_data[matches_data['toss_winner']==matches_data['winner']]

slice = [len(temp),(matches_data.shape[0]-len(temp))]

labels = ['Toss Winner wins match', 'Toss Winner losses match']

plt.pie(slice, labels=labels,autopct='%1.2f%%',startangle=90,shadow=True,explode=(0,0.03))

plt.show()
matches_played_byteams=pd.concat([matches_data['team1'],matches_data['team2']])

matches_played_byteams=matches_played_byteams.value_counts().reset_index()

matches_played_byteams.columns=['Team','Total Matches']

matches_played_byteams['wins']=matches_data['winner'].value_counts().reset_index()['winner']

matches_played_byteams.set_index('Team',inplace=True)



trace1 = go.Bar(x=matches_played_byteams.index,

                y=matches_played_byteams['Total Matches'],

                name='Total Matches')



trace2 = go.Bar(x=matches_played_byteams.index,

                y=matches_played_byteams['wins'],

                name='Matches Won')



data = [trace1, trace2]

layout = go.Layout(barmode='stack')



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')

temp1 = matches_data

temp1['Toss_Winner_is_Match_Winner'] = 'no'

temp1['Toss_Winner_is_Match_Winner'].ix[matches_data['toss_winner']==matches_data['winner']] = 'yes'

plt.figure(figsize=(15,7))

sns.countplot(x='toss_winner', hue='Toss_Winner_is_Match_Winner', data=temp1)

plt.xticks(rotation='vertical')

plt.show()
deliveries_data.head()
bowlers = matches_data[['id','season']].merge(deliveries_data, left_on='id',right_on='match_id',how='left').drop('id',axis=1)

bowlers.head()
total_wickets = bowlers[bowlers.dismissal_kind !='0']

total_wickets['dismissal_kind'].value_counts()
plt.figure(figsize=(15,6))

temp = sns.countplot(x='season',data=total_wickets)

for i in temp.patches:

    temp.annotate(format(i.get_height()),(i.get_x()+0.20, i.get_height()),fontsize=15)

plt.title('Wickets Fallen each season',fontsize=20)

plt.show()
plt.figure(figsize=(14,6))

temp= total_wickets['dismissal_kind'].value_counts().plot(marker='o', color='red')

#for i in temp.patches:

 #   temp.annotate(format(i.get_height()),(i.get_x()+0.20, i.get_height()),fontsize=15)



plt.xticks(rotation='vertical')

plt.show()
dismissals = ['caught','bowled','lbw','stumped','caught and bowled','hit wicket']

wickets = bowlers[bowlers['dismissal_kind'].isin(dismissals)]

wickets['dismissal_kind'].value_counts()
temp = wickets['bowler'].value_counts()[:10]

temp
plt.figure(figsize=(14,6))

temp = wickets['bowler'].value_counts()[:10].plot(kind='bar', color=sns.color_palette('autumn',10))

for i in temp.patches:

    temp.annotate(format(i.get_height()),(i.get_x()+0.20, i.get_height()),fontsize=15)

plt.title('Highest Wicket Takers',fontsize=20)

plt.show()
plt.figure(figsize=(15,6))

temp = sns.countplot(x='season',data=wickets)

for i in temp.patches:

    temp.annotate(format(i.get_height()),(i.get_x()+0.20, i.get_height()),fontsize=15)

plt.title('Wickets Taken by bowlers each season',fontsize=20)

plt.show()
batsmen = matches_data[['id','season']].merge(deliveries_data, left_on='id', right_on='match_id', how='left')

batsmen = batsmen.drop('id',axis=1)

batsmen.head()
batsmen.columns
temp = batsmen.groupby('batsman')['batsman_runs'].sum().reset_index()

temp = temp.sort_values('batsman_runs', ascending=False)[:10]

temp.reset_index(drop=True)    # reset_index(drop=True) will reset the index column 
temp = temp.plot(kind='bar', x='batsman', y='batsman_runs', width=0.8, color=sns.color_palette('summer',20))

for i in temp.patches:

    temp.annotate(format(i.get_height()),(i.get_x()+0.20, i.get_height()),fontsize=15)



fig=plt.gcf()

fig.set_size_inches(14,6)

plt.show()
batsmen.groupby('season')['total_runs'].sum().plot(marker='o')

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.title('Runs scored every season')

plt.show()
temp = batsmen.groupby('season')['total_runs'].sum().plot(kind='bar',color= sns.color_palette('summer'))      

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.title('Runs scored every season')

plt.show()
batsmen['batsman_runs'].value_counts()
boundary = ['4']

fours = batsmen[batsmen['batsman_runs'].isin(boundary)]

fours['batsman'].value_counts()[:10]
plt.figure(figsize=(15,6))

temp = sns.countplot(x='season',data=fours)

for i in temp.patches:

    temp.annotate(format(i.get_height()),(i.get_x()+0.20, i.get_height()),fontsize=15)

plt.title('4"s every season',fontsize=20)

plt.show()
six = ['6']

sixes = batsmen[batsmen['batsman_runs'].isin(six)]

sixes['batsman'].value_counts()[:10]
plt.figure(figsize=(15,6))

temp = sns.countplot(x='season',data=sixes)

for i in temp.patches:

    temp.annotate(format(i.get_height()),(i.get_x()+0.20, i.get_height()),fontsize=15)

plt.title('6"s every season',fontsize=20)

plt.show()
a=sixes.groupby("season")["batsman_runs"].agg(lambda x : x.sum()).reset_index()



b=fours.groupby("season")["batsman_runs"].agg(lambda x: x.sum()).reset_index()



boundaries=a.merge(b,left_on='season',right_on='season',how='left')



boundaries=boundaries.rename(columns={'batsman_runs_x':'6"s','batsman_runs_y':'4"s'})



boundaries.set_index('season')[['6"s','4"s']].plot(marker='o',color=['red','green'])

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()