import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
#import deliveries and matches files
matches = pd.read_csv('../input/matches.csv')
deliveries = pd.read_csv('../input/deliveries.csv')
#check head of each files
deliveries.head()
matches.head()
matches.columns
#total No of matches
print(matches.shape[0])
#seasons of IPL
print(matches['season'].unique())
#plot showing no of matches per season
sns.countplot(x='season',data=matches)
#No of matches conducted at a given venue
sns.countplot(x='venue',data=matches)
plt.xticks(rotation='vertical')
#no of duckwoth lewis applied

dl_applied=pd.DataFrame(matches['dl_applied'].value_counts())
dl_applied
#each team winnings in all seasons
matches_won=pd.DataFrame(matches['winner'].value_counts())
matches_won
sns.countplot(x='winner',data=matches)
plt.xticks(rotation='vertical')
player_of_match=pd.DataFrame(matches['player_of_match'].value_counts())
player_of_match
#no of tosses each team has won in all seasons
sns.countplot(x='toss_winner',data=matches)
plt.xticks(rotation='vertical')
toss_decision=matches.toss_decision.value_counts()
toss_decision
toss_decision.plot(kind='bar')
sns.countplot(x='toss_decision',hue='season',data=matches)

sns.countplot(x='season',hue='toss_decision',data=matches)
#bog margin wins i.e won by 100> runs or won by > 5 wickets
big_margin = matches[(matches['win_by_runs'] > 100) | (matches['win_by_wickets'] > 5)]
sns.countplot(x='winner', data=big_margin)
plt.xticks(rotation = 'vertical')
deliveries.head()
runs_scored=deliveries.groupby(['batsman'])['batsman_runs'].sum()
runs_scored=runs_scored.sort_values(ascending=False)
runs_scored
top_10_run_scorers=pd.DataFrame(runs_scored.head(10))
top_10_run_scorers
top_10_run_scorers.plot(kind='bar')
wickets_taken=deliveries.groupby(['bowler'])['player_dismissed'].count()
wickets_taken=wickets_taken.sort_values(ascending=False)
wickets_taken
top_10_wicket_takers=pd.DataFrame(wickets_taken.head(10))
top_10_wicket_takers
top_10_wicket_takers.plot(kind='barh')
runs_given_by_bowler=deliveries.groupby(['bowler'])['total_runs'].sum()
runs_given_by_bowler=runs_given_by_bowler.sort_values(ascending=False)
runs_given_by_bowler.head(10).plot(kind='bar')
extra_runs_given_by_bowler=deliveries.groupby(['bowler'])['extra_runs'].sum()
extra_runs_given_by_bowler=extra_runs_given_by_bowler.sort_values(ascending=False)
extra_runs_given_by_bowler.head(10).plot(kind='bar')
deliveries.head(10)
batsman_dismissals=deliveries.groupby(['batsman'])['dismissal_kind'].count()
batsman_dismissals=batsman_dismissals.sort_values(ascending=False)
batsman_dismissals
batsman_dismissals.head(10).plot(kind='bar')
batsman_dismissals_df=pd.DataFrame(batsman_dismissals)
batsman_dismissals_df
#batsman wise dismissal kinds
batsman_dismissals1=deliveries.groupby(['batsman','dismissal_kind'])
batsman_dismissals1_df=pd.DataFrame(batsman_dismissals1.size())
batsman_dismissals1_df
total_dismissal_types=deliveries.groupby(['dismissal_kind'])['batsman'].count()
total_dismissal_types
#team wise dismissals
team_dismissals=deliveries.groupby(['batting_team'])['dismissal_kind'].count()
team_dismissals
#team wise dismissals and its types
team_dismissals=deliveries.groupby(['batting_team','dismissal_kind'])
#team_dismissals_df=pd.DataFrame(team_dismissals.size())
#team_dismissals_df
team_dismissals.size()
