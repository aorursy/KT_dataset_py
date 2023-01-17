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
df = pd.read_csv("/kaggle/input/ipldata/matches.csv")
runs = 10
wicket= 3

team ='Royal Challengers Bangalore'
team_df =  df.query(' (team1==@team or team2 ==@team)  and ( (win_by_runs<=@runs and win_by_runs!=0)  or (win_by_wickets!=0 and win_by_wickets<=@wicket))  ')
team_df['results']=np.where(team_df['winner']== team, "Won", "Lost")

print(team_df.groupby(['season','results'])[["id"]].agg(['count']).head(30))
team ='Kolkata Knight Riders'
team_df =  df.query(' (team1==@team or team2 ==@team)  and ( (win_by_runs<=@runs and win_by_runs!=0)  or (win_by_wickets!=0 and win_by_wickets<=@wicket))  ')
team_df['results']=np.where(team_df['winner']== team, "Won", "Lost")

print(team)
print(team_df.groupby(['season','results'])[["id"]].agg(['count']).head(30))

team ='Mumbai Indians'
team_df =  df.query(' (team1==@team or team2 ==@team)  and ( (win_by_runs<=@runs and win_by_runs!=0)  or (win_by_wickets!=0 and win_by_wickets<=@wicket))  ')
team_df['results']=np.where(team_df['winner']== team, "Won", "Lost")

print(team)
print(team_df.groupby(['season','results'])[["id"]].agg(['count']).head(30))
team ='Delhi Daredevils'
team_df =  df.query(' (team1==@team or team2 ==@team)  and ( (win_by_runs<=@runs and win_by_runs!=0)  or (win_by_wickets!=0 and win_by_wickets<=@wicket))  ')
team_df['results']=np.where(team_df['winner']== team, "Won", "Lost")

print(team)
print(team_df.groupby(['season','results'])[["id"]].agg(['count']).head(30))

team ='Sunrisers Hyderabad'
team_df =  df.query(' (team1==@team or team2 ==@team)  and ( (win_by_runs<=@runs and win_by_runs!=0)  or (win_by_wickets!=0 and win_by_wickets<=@wicket))  ')
team_df['results']=np.where(team_df['winner']== team, "Won", "Lost")

print(team)
print(team_df.groupby(['season','results'])[["id"]].agg(['count']).head(30))
team ='Kings XI Punjab'
team_df =  df.query(' (team1==@team or team2 ==@team)  and ( (win_by_runs<=@runs and win_by_runs!=0)  or (win_by_wickets!=0 and win_by_wickets<=@wicket))  ')
team_df['results']=np.where(team_df['winner']== team, "Won", "Lost")

print(team)
print(team_df.groupby(['season','results'])[["id"]].agg(['count']).head(30))

team ='Rajasthan Royals'
team_df =  df.query(' (team1==@team or team2 ==@team)  and ( (win_by_runs<=@runs and win_by_runs!=0)  or (win_by_wickets!=0 and win_by_wickets<=@wicket))  ')
team_df['results']=np.where(team_df['winner']== team, "Won", "Lost")

print(team)
print(team_df.groupby(['season','results'])[["id"]].agg(['count']).head(30))
team ='Chennai Super Kings'
team_df =  df.query(' (team1==@team or team2 ==@team)  and ( (win_by_runs<=@runs and win_by_runs!=0)  or (win_by_wickets!=0 and win_by_wickets<=@wicket))  ')
team_df['results']=np.where(team_df['winner']== team, "Won", "Lost")

print(team)
print(team_df.groupby(['season','results'])[["id"]].agg(['count']).head(30))
close_df = df.query(' ( (win_by_runs<=@runs and win_by_runs!=0)  or (win_by_wickets!=0 and win_by_wickets<=@wicket))  ')
print(close_df.groupby(['venue'])[["id"]].count().sort_values(['id'], ascending=False).head(30))
team_df =  df.query(' ((win_by_runs<=@runs and win_by_runs!=0)  or (win_by_wickets!=0 and win_by_wickets<=@wicket))  ')

print(team_df.groupby(['winner'])[["id"]].count().sort_values(['id'], ascending=False).head(30))
team_df =  df.query(' ( (win_by_runs<=@runs and win_by_runs!=0)  or (win_by_wickets!=0 and win_by_wickets<=@wicket))  ')
team_df["losser"] = np.where(team_df['winner']!= team_df['team1'], team_df['team1'], team_df['team2'])
print(team_df.groupby(['losser'])[["id"]].count().sort_values(['id'], ascending=False).head(30))
team_df =  df.query(' ( (win_by_runs<=@runs and win_by_runs!=0)  or (win_by_wickets!=0 and win_by_wickets<=@wicket))  ')
team_df["losser"] = np.where(team_df['winner']!= team_df['team1'], team_df['team1'], team_df['team2'])
print(team_df.groupby(['winner','losser'])[["id"]].count().sort_values(['winner'], ascending=False).head(50))
