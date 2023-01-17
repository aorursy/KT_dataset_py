# Import statements for all of the packages that I plan to use
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
import sqlite3
import scipy.stats as stats
%matplotlib inline
# Import data with an SQL query to combine all data from Team table with goals scored and team data from Match table
# After cleaning I will use this dataframe to compute team win records and performance over the years
with sqlite3.connect('database.sqlite') as con:
    df_1_per = pd.read_sql("SELECT t.*, m.date, m.match_api_id, m.home_team_api_id, m.away_team_api_id, m.home_team_goal, m.away_team_goal FROM Team t, Match m WHERE t.team_api_id = m.home_team_api_id OR t.team_api_id = m.away_team_api_id", con)
df_1_per.head(1)
# Import all data from Team Attributes table to be plotted later against team performance to see if we can find strong correlations. 
# Will combine with df_1_per table to look for correlations once it is better wrangled.
with sqlite3.connect('database.sqlite') as con:
    df_2_att = pd.read_sql_query("SELECT * FROM Team_Attributes", con)
df_2_att.head(1)
# Change date to datetime type for Team Attributes table
# Look at unique values per column
df_2_att['date'] = pd.to_datetime(df_2_att['date'])
df_2_att.nunique()
# No duplicates
sum(df_2_att.duplicated())
# Here I try to understand why there are so many id and so few team_api_id in the Team Attribute table
df_2_att.groupby('team_api_id').count().head(4)
# Here we see that teams are given different attributes for different years. Therefore, we'll have to compare team attributes 
# to the team win record for each specific year
df_2_att.query('team_api_id==1601')
# We see that we only have team attributes between 2010 and 2015, and for 2010 to 2013 there is no data for buildUpPlayDribbling
df_2_att.groupby(df_2_att['date'].dt.year).count()
# Drop buildUpPlayDribbling attribute here
df_2_att.drop('buildUpPlayDribbling', inplace=True, axis=1)
# Clean table by changing date to only include year
df_2_att['date'] = df_2_att['date'].dt.year
df_2_att.rename(columns={"date":"year"}, inplace=True)
df_2_att.head(3)
# Change date to datetime type
# Look at unique values for each column
df_1_per['date'] = pd.to_datetime(df_1_per['date'])
df_1_per.nunique()
# No duplicates
sum(df_1_per.duplicated())
# Change date to year
df_1_per['date'] = df_1_per['date'].dt.year
df_1_per.rename(columns={"date":"year"}, inplace=True)
df_1_per.head(2)
# 2008 and 2016 don't have complete year datasets. 2009 doesn't have team attribute data. 
# Therefore I will drop data from 2008,2009,and 2016 and do my analysis on only the data from 2010 to 2015.
df_1_per.drop(df_1_per[df_1_per.year == 2008].index, inplace=True)
df_1_per.drop(df_1_per[df_1_per.year == 2009].index, inplace=True)
df_1_per.drop(df_1_per[df_1_per.year == 2016].index, inplace=True)
df_1_per.nunique()
# Get all winning pairs between team api id and win
df_1_home_wins=df_1_per.query('team_api_id==home_team_api_id').query('home_team_goal > away_team_goal')
df_1_away_wins=df_1_per.query('team_api_id==away_team_api_id').query('home_team_goal < away_team_goal')
df_1_wins = pd.concat([df_1_home_wins,df_1_away_wins])
df_1_wins.head(2)
# Get all losing pairs between team api id and loss
df_1_home_losses=df_1_per.query('team_api_id==home_team_api_id').query('home_team_goal < away_team_goal')
df_1_away_losses=df_1_per.query('team_api_id==away_team_api_id').query('home_team_goal > away_team_goal')
df_1_losses = pd.concat([df_1_home_losses,df_1_away_losses])
df_1_losses.head(2)
# Get all drawing pairs between team api id and draw
df_1_home_draws=df_1_per.query('team_api_id==home_team_api_id').query('home_team_goal == away_team_goal')
df_1_away_draws=df_1_per.query('team_api_id==away_team_api_id').query('home_team_goal == away_team_goal')
df_1_draws = pd.concat([df_1_home_draws,df_1_away_draws])
df_1_draws.head(2)
# Drop unneeded columns
columns = ['id','team_fifa_api_id','team_short_name','home_team_api_id','away_team_api_id','home_team_goal','away_team_goal']
df_1_wins.drop(columns, inplace=True, axis=1)
df_1_losses.drop(columns, inplace=True, axis=1)
df_1_draws.drop(columns, inplace=True, axis=1)
# Check unique values for each column 
df_1_wins.nunique()
# Create wins, losses, and draws columns
# Create dataframes by grouping data by teams and years
df_1_wins.rename(columns={"match_api_id":"wins"}, inplace=True)
df_1_wins_by_year=df_1_wins.groupby(['team_api_id','year','team_long_name'], as_index=False).count()

df_1_losses.rename(columns={"match_api_id":"losses"}, inplace=True)
df_1_losses_by_year=df_1_losses.groupby(['team_api_id','year','team_long_name'], as_index=False).count()

df_1_draws.rename(columns={"match_api_id":"draws"}, inplace=True)
df_1_draws_by_year=df_1_draws.groupby(['team_api_id','year','team_long_name'], as_index=False).count()
# Wins table
df_1_wins_by_year.head(2)
# loss table
df_1_losses_by_year.head(2)
# Draws table
df_1_draws_by_year.head(2)
# Merge the wins and losses dataframes
df_1_wins_losses = pd.merge(df_1_wins_by_year,df_1_losses_by_year,how='inner',on=['team_api_id','year','team_long_name'])
# Merge the win/loss and draws dataframes
df_1_records_by_year = pd.merge(df_1_wins_losses,df_1_draws_by_year,how='inner',on=['team_api_id','year','team_long_name'])
# Create new columns for points, games played and points_per_game
df_1_records_by_year['points'] = df_1_records_by_year['wins']*3 + df_1_records_by_year['draws']
df_1_records_by_year['games_played'] = df_1_records_by_year['wins'] + df_1_records_by_year['losses'] + df_1_records_by_year['draws']
df_1_records_by_year['points_per_game'] = df_1_records_by_year['points']/df_1_records_by_year['games_played']
# View table and sort by points
df_1_records_by_year.sort_values(by='points', ascending=False).head(5)
# Merge performance dataframe with attribute dataframe on team_api_id and year
df_q2_wins_att=pd.merge(df_1_records_by_year, df_2_att,  how='inner', on=['team_api_id','year'])
df_q2_wins_att.head(3)
# Basic statistics for our table
df_1_records_by_year.describe()
# We have 283 teams before we apply our constraints
df_1_records_by_year.nunique()
# Get list of teams that have records for at least 4 of 6 years with a minimum of 20 games played
df_mingames = df_1_records_by_year.drop(df_1_records_by_year[df_1_records_by_year.games_played < 20].index)
df=df_mingames.groupby('team_long_name', as_index = False).count()
df.drop(df[df.points < 4].index, inplace=True)
df.drop(df[df.points > 6].index, inplace=True)
columns = ['team_api_id','wins','losses','draws','points','games_played','points_per_game']
df.drop(columns, inplace=True, axis=1)
df.rename(columns={"year":"year_count"}, inplace=True)
# We are now comparing only 152 from the original 283 teams
df.nunique()
# Merge list of teams with performance dataframe
df_trends=pd.merge(df_1_records_by_year, df,  how='inner', on='team_long_name')
df_trends.head(3)
df_trends.nunique()
# Who has been the best from 2010 to 2015 based on our new criteria?
df_trends.groupby('team_long_name').mean().sort_values(by='points_per_game', ascending=False).head(5)
# What is the spectrum of points per game that teams average?
plt.subplots(figsize=(8, 5))
plt.hist(df_trends['points_per_game'])
plt.title('Histogram of Avg Points per game per year. Constraint: minimum 20 games per year and 4 years played between 2010 to 2015')
plt.xlabel('Avg points per game')
plt.ylabel('Number of teams/years')
# Now to find the teams that have improved the most
# Here we use scipy.stats to get the line of best fit of the points_per_game vs years for each team
# This slope will tell us who has improved the most based on our criteria of a minimum of 4 years and 20 games per year of play
# List our 10 most improved teams
df_points_trends = df_trends.groupby('team_long_name').apply(lambda v: stats.linregress(v.year,v.points_per_game)[0])
df_points_trends.sort_values(axis=0, ascending=False).head(10)
# Details of Aberdeen's improvement
df_trends.query('team_long_name=="Aberdeen"')
# Plot points per game for the most improved team "Aberdeen"
plt.subplots(figsize=(8, 5))
plt.plot(df_trends.query('team_long_name=="Aberdeen"')['year'],df_trends.query('team_long_name=="Aberdeen"')['points_per_game'])
plt.title('Aberdeen Improvement from 2010 to 2015')
plt.xlabel('Year')
plt.ylabel('Average Points per Game')
# Details of Paris Saint-Germain's improvement
df_trends.query('team_long_name=="Paris Saint-Germain"')
# Plot points per game for the 2nd most improved team "Paris Saint-Germain"
plt.subplots(figsize=(8, 5))
plt.plot(df_trends.query('team_long_name=="Paris Saint-Germain"')['year'],df_trends.query('team_long_name=="Paris Saint-Germain"')['points_per_game'])
plt.title('Paris Saint-Germain Improvement from 2010 to 2015')
plt.xlabel('Year')
plt.ylabel('Average Points per Game')
# Look at the basic statistics
df_q2_wins_att.describe()
# Do a pearson correlation analysis on the numerical columns
df_q2_wins_att.corr()
# Defence pressure is the attribute most correlated with performance
# Here we explore defencePressure
# Use a scatter plot to visualize the correlation of defencePressure vs points per game
plt.subplots(figsize=(8, 5))
plt.scatter(df_q2_wins_att['defencePressure'],df_q2_wins_att['points_per_game'])
plt.title('Team Attribute Rating: defencePressure vs Points per Game')
plt.xlabel('Team Attribute Rating: defencePressure')
plt.ylabel('Average Points per Game')
# Explore the difference in defencePressure rating among teams
plt.subplots(figsize=(8, 5))
plt.hist(df_q2_wins_att['defencePressure'])
plt.title('Histogram of defencePressure attribute rating')
plt.xlabel('DefencePressure attribute rating')
plt.ylabel('Number of teams/seasons')
# Here we look at the correlation coefficient and p-value for the 3 numerical attributes most correlated to points per game
stats.pearsonr(df_q2_wins_att['points_per_game'], df_q2_wins_att['defencePressure'])
stats.pearsonr(df_q2_wins_att['points_per_game'], df_q2_wins_att['chanceCreationShooting'])
stats.pearsonr(df_q2_wins_att['points_per_game'], df_q2_wins_att['defenceAggression'])
# Average points per game per team
df_q2_wins_att['points_per_game'].mean()
df_q2_wins_att.groupby('buildUpPlaySpeedClass')['points_per_game'].mean()
df_q2_wins_att.groupby('buildUpPlayDribblingClass')['points_per_game'].mean()
df_q2_wins_att.groupby('buildUpPlayPassingClass')['points_per_game'].mean()
df_q2_wins_att.groupby('buildUpPlayPositioningClass')['points_per_game'].mean()
df_q2_wins_att.groupby('chanceCreationPassingClass')['points_per_game'].mean()
df_q2_wins_att.groupby('chanceCreationShootingClass')['points_per_game'].mean()
df_q2_wins_att.groupby('defencePressureClass')['points_per_game'].mean()
df_q2_wins_att.groupby('defenceAggressionClass')['points_per_game'].mean()
df_q2_wins_att.groupby('defenceTeamWidthClass')['points_per_game'].mean()
df_q2_wins_att.groupby('defenceDefenderLineClass')['points_per_game'].mean()
# This bar plot shows that most teams prefer to use a Mixed buildUpPlayPassingClass even though those teams who choose a short
# approach have a much higher points per game average.
fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.3
df_q2_wins_att.groupby('buildUpPlayPassingClass')['points_per_game'].mean().plot(kind='bar', color='red', ax=ax, width=width, position=1, legend=True)
df_q2_wins_att['buildUpPlayPassingClass'].value_counts().plot(kind='bar', color='blue', ax=ax2, width=width, position=0, legend=True)

plt.title('Comparison of # of teams using buildUpPlayPassingClass strategies vs Avg Points/game')
ax.set_ylabel('Points per Game')
ax.set_yticks(np.arange(0,2,0.2))
ax.legend(bbox_to_anchor=(0.4, 0.98))
ax2.set_ylabel('# of Teams using Strategy')
ax2.set_yticks(np.arange(0, 1500,300))
ax2.legend(bbox_to_anchor=(1, 0.98))

plt.show()
#Here I'll compute t-test on categories to see if points_per_game are significantly different based on buildUpPlayPositioningClass
Free_Form = df_q2_wins_att[df_q2_wins_att['buildUpPlayPositioningClass']=='Free Form']
Organised = df_q2_wins_att[df_q2_wins_att['buildUpPlayPositioningClass']=='Organised']

stats.ttest_ind(Free_Form['points_per_game'], Organised['points_per_game'],equal_var=False)
# Here I'll compute t-test on categories to see if the points_per_game are significantly different based on buildUpPlayPassingClass
Short = df_q2_wins_att[df_q2_wins_att['buildUpPlayPassingClass']=='Short']
Mixed = df_q2_wins_att[df_q2_wins_att['buildUpPlayPassingClass']=='Mixed']

stats.ttest_ind(Short['points_per_game'], Mixed['points_per_game'],equal_var=False)
Short = df_q2_wins_att[df_q2_wins_att['buildUpPlayPassingClass']=='Short']
Long = df_q2_wins_att[df_q2_wins_att['buildUpPlayPassingClass']=='Long']

stats.ttest_ind(Short['points_per_game'], Long['points_per_game'],equal_var=False)
#Here I'll compute t-test on categories to see if points_per_game are significantly different based on chanceCreationPassingClass
Risky = df_q2_wins_att[df_q2_wins_att['chanceCreationPassingClass']=='Risky']
Normal = df_q2_wins_att[df_q2_wins_att['chanceCreationPassingClass']=='Normal']

stats.ttest_ind(Risky['points_per_game'], Normal['points_per_game'],equal_var=False)
Risky = df_q2_wins_att[df_q2_wins_att['chanceCreationPassingClass']=='Risky']
Safe = df_q2_wins_att[df_q2_wins_att['chanceCreationPassingClass']=='Safe']

stats.ttest_ind(Risky['points_per_game'], Safe['points_per_game'],equal_var=False)
# Plot points per game for the most improved team "Aberdeen"
plt.subplots(figsize=(8, 5))
plt.plot(df_trends.query('team_long_name=="Aberdeen"')['year'],df_trends.query('team_long_name=="Aberdeen"')['points_per_game'])
plt.title('Aberdeen Improvement from 2010 to 2015')
plt.xlabel('Year')
plt.ylabel('Average Points per Game')