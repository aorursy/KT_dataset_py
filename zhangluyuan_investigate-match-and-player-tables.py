#import necessary tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
import sqlite3
import scipy.stats as stats
%matplotlib inline
# Establish connection to database, and load the tables.
conn=sqlite3.connect('database.sqlite')
match=pd.read_sql_query('SELECT * FROM match;', conn)
team_attributes=pd.read_sql_query('SELECT * FROM team_attributes',conn)
player_attributes=pd.read_sql_query('SELECT * FROM player_attributes;', conn)
#check for missing values
np.array(match.isnull().sum())
# no duplicates in match
match.duplicated().sum()
#view first few rows of match
match.head()
#most columns in match are player columns
np.array(match.columns)
#view first few rows
team_attributes.head()
#no duplicates
team_attributes.duplicated().sum()
#check for missing data
team_attributes.info()
#check for missing values
player_attributes.isnull().sum()
#count number of rows that contains missing values
player_attributes['null_count']=player_attributes.isnull().sum(axis=1)
(player_attributes['null_count']!=0).sum()
#calculate ther percentage of number of rows that contain NaN
(player_attributes['null_count']!=0).sum()/player_attributes.shape[0]
# Check the first few rows of the table player_attributes
player_attributes.head()
#no duplicates
player_attributes.duplicated().sum()
#load other necessary tables
team=pd.read_sql_query('SELECT * FROM team',conn)
player=pd.read_sql_query('SELECT * FROM player',conn)
# view first few rows of team
team.head()
# no duplicates
team.duplicated().sum()
# check for missing values
team.isnull().sum()
#view first few rows of the table player
player.head()
# no duplicates
player.duplicated().sum()
# no missing values
player.isnull().sum()
team_attributes.drop(columns='buildUpPlayDribbling', inplace=True)
#drop rows containing NaN
player_attributes.dropna(inplace=True)
#merge player and player_attributes
player_data=pd.merge(player_attributes, player, on=['player_api_id', 'player_fifa_api_id'])
#change date and birthday to datetime
player_data.date=pd.to_datetime(player_data.date)
player_data.birthday=pd.to_datetime(player_data.birthday)
#calculate player age
player_data['age']=(player_data['date']-player_data['birthday']).dt.days/365.25
team.drop(columns='team_fifa_api_id', inplace=True)
# extract first few columns of match
match=match[['id', 'country_id', 'league_id', 'season', 'stage', 'date','match_api_id', 'home_team_api_id', 'away_team_api_id','home_team_goal', 'away_team_goal']]
#add 2 columns that calculates the points of home_team and away_team
match['home_points']= (match.home_team_goal > match.away_team_goal)*3 + (match.home_team_goal == match.away_team_goal)*1
match['away_points']= (match.home_team_goal < match.away_team_goal)*3 + (match.home_team_goal == match.away_team_goal)*1
match.head()
#split the match table into home dataframe and away dataframe, then append them into a single dataframe
homedf=match[['home_team_api_id', 'date','home_team_goal','home_points']]
homedf.columns=['team_api_id', 'date', 'goal','points']
awaydf=match[['away_team_api_id', 'date','away_team_goal','away_points']]
awaydf.columns=['team_api_id', 'date', 'goal','points']
df1=pd.DataFrame(homedf.append(awaydf))
#create a year column
df1['date']=pd.to_datetime(df1['date'])
df1['year']=df1['date'].dt.year
#create a new dataframe team_data, that aggregates points into point per game for each year
team_data=pd.DataFrame()
team_data['total_goals']=df1.groupby(['team_api_id','year'])['goal'].sum()
team_data['num_of_games']=df1.groupby(['team_api_id','year']).goal.count()
team_data['total_points']=df1.groupby(['team_api_id','year']).points.sum()
team_data['points_per_game']=team_data['total_points'] / team_data['num_of_games']
team_data.reset_index(inplace=True)
#merge with the table team to include team long name, and drop other unnecessary columns
team=pd.read_sql_query("SELECT * FROM team;", conn)
team_data=pd.merge(team_data, team, on='team_api_id')
team_data.drop(columns=['id', 'team_fifa_api_id','team_short_name'], inplace=True)
#drop the rows where the year is before 2010
team_data=team_data[team_data.year>2009]
team_data.head()
#create a dataframe that counts how many year each team played
num_years=team_data.groupby('team_api_id').year.count()
num_years=pd.DataFrame(num_years)
num_years.columns=['num_years']
num_years.reset_index(inplace=True)
num_years.head()
#merge the team_per and num_year dataframes
team_data=pd.merge(team_data, num_years, on='team_api_id')
#filter the dataframe to select only teams who has played at least 5 years 
team_data=team_data.query('num_years>4')
#calculate the percentage of home team win, lose and draw, and save them to a list
home_results=[]
home_results.append(sum((match.home_points==3)))
home_results.append((match.home_points==0).sum())
home_results.append((match.home_points==1).sum())
home_results=home_results/sum(home_results)
home_results=np.array(home_results)
# A bar diagram is the most straightforward visualization for the differences between win, lose and draw chances.
xlabel=['win', 'lose','draw']
plt.bar(xlabel,home_results);
plt.ylabel('percentage')
plt.title('Home team result');
# extract my sample from match table, where the game result is home win or home lose
hypotest_df=match.query('home_points!=1')['home_points']==3
hypotest_df.count()
# bootstrap my sample 10000 times
boot_means=[]
for _ in range(10000):
    boot_sample=hypotest_df.sample(hypotest_df.count(), replace=True)
    boot_means.append(boot_sample.mean())
boot_means=np.array(boot_means)
sample_mean=boot_means.mean()
sample_std=boot_means.std()
sample_mean, sample_std
Null_val=np.random.normal(0.5, sample_std, hypotest_df.count())
lowbound=0.5-(sample_mean-0.5)
p=(Null_val>sample_mean).mean() + (Null_val < lowbound).mean()
p
# calculate the slope of points_per_game vs. year, and save it to team_improvement 
team_improvement = team_data.groupby('team_long_name').apply(lambda v: stats.linregress(v.year,v.points_per_game)[0])
team_improvement.sort_values(axis=0, ascending=False).head(10)
#to check if these values of improvements are indeed at the upper end, I plot a histogram of team imporvements
team_improvement.hist(bins=50)
plt.xlabel('improvement points/game/year')
plt.ylabel('number of teams')
plt.title('team improvements');
#plot the points_per_game vs year to view improvement of the top 3 teams
x1=team_data.query('team_long_name=="Southampton"')['year']
y1=team_data.query('team_long_name=="Southampton"')['points_per_game']
x2=team_data.query('team_long_name=="Dundee FC"')['year']
y2=team_data.query('team_long_name=="Dundee FC"')['points_per_game']
x3=team_data.query('team_long_name=="Juventus"')['year']
y3=team_data.query('team_long_name=="Juventus"')['points_per_game']
plt.plot(x1,y1, label='Southampton')
plt.plot(x2,y2, label="Dundee FC")
plt.plot(x3,y3, label="Juventus")
plt.legend()
plt.xlabel('Year')
plt.ylabel('Points_per_Game')
plt.title("Southampton_Improvement_from_2010_to_2016");
#create a year column in team_attributes
team_attributes['year']=pd.to_datetime(team_attributes['date']).dt.year
#merge table team_data and team_attributes on team_api_id and year
team_data=pd.merge(team_data, team_attributes, on=['team_api_id', 'year'])
#view properties of columns
team_data.info()
#create a dataframe that contains only quantitative attributes
num_df=team_data[['team_api_id', 'team_long_name', 'year', 'points_per_game','buildUpPlaySpeed','buildUpPlayPassing','chanceCreationPassing','chanceCreationCrossing','chanceCreationShooting','defencePressure','defenceAggression','defenceTeamWidth' ]]
#Get a first overall glance of the correlation between quantitative columns.
num_df.corr()
#visual inpection of potential correlation between points_per_game and defencePressure
plt.scatter(num_df.defencePressure, num_df.points_per_game)
plt.title('team defencePressure vs points per game')
plt.xlabel('team defencePressure')
plt.ylabel('team points per game');
#visual inpection of potential correlation between points_per_game and buildUpPlayPassing
plt.scatter(num_df.buildUpPlayPassing, num_df.points_per_game)
plt.title('team buildUpPlayPassing vs points per game')
plt.xlabel('team buildUpPlayPassing')
plt.ylabel('team points per game');
#visual inpection of potential correlation between points_per_game and chanceCreationShooting
plt.scatter(num_df.chanceCreationShooting, num_df.points_per_game)
plt.title('team chanceCreationShooting vs points per game')
plt.xlabel('team chanceCreationShooting')
plt.ylabel('team points per game');
# Pearson correlation between points_per_game and defencePressure
stats.pearsonr(num_df.points_per_game, num_df.defencePressure)
# Pearson correlation between points_per_game and buildUpPlayPassing
stats.pearsonr(num_df.points_per_game, num_df.buildUpPlayPassing)
# Pearson correlation between points_per_game and chanceCreationShooting
stats.pearsonr(num_df.points_per_game, num_df.chanceCreationShooting)
# unique value and counts of buildUpPlaySpeedClass
team_data['buildUpPlaySpeedClass'].value_counts()
# mean points_per_game of each category value
buildUpPlaySpeedClass_mean_balanced=team_data.query('buildUpPlaySpeedClass=="Balanced"')['points_per_game'].mean()
buildUpPlaySpeedClass_mean_fast=team_data.query('buildUpPlaySpeedClass=="Fast"')['points_per_game'].mean()
buildUpPlaySpeedClass_mean_slow=team_data.query('buildUpPlaySpeedClass=="Slow"')['points_per_game'].mean()
buildUpPlaySpeedClass_mean_balanced, buildUpPlaySpeedClass_mean_fast, buildUpPlaySpeedClass_mean_slow
# unique value and counts of buildUpPlayDribblingClass
team_data['buildUpPlayDribblingClass'].value_counts()
# mean points_per_game of each category value
buildUpPlayDribblingClass_mean_little=team_data.query('buildUpPlayDribblingClass=="Little"')['points_per_game'].mean()
buildUpPlayDribblingClass_mean_Normal=team_data.query('buildUpPlayDribblingClass=="Normal"')['points_per_game'].mean()
buildUpPlayDribblingClass_mean_lots=team_data.query('buildUpPlayDribblingClass=="Lots"')['points_per_game'].mean()
buildUpPlayDribblingClass_mean_little, buildUpPlayDribblingClass_mean_Normal, buildUpPlayDribblingClass_mean_lots
# unique value and counts of buildUpPlayPassingClass
team_data['buildUpPlayPassingClass'].value_counts()
# mean points_per_game of each category value
buildUpPlayPassingClass_mean_mixed=team_data.query('buildUpPlayPassingClass=="Mixed"')['points_per_game'].mean()
buildUpPlayPassingClass_mean_short=team_data.query('buildUpPlayPassingClass=="Short"')['points_per_game'].mean()
buildUpPlayPassingClass_mean_long=team_data.query('buildUpPlayPassingClass=="Long"')['points_per_game'].mean()
buildUpPlayPassingClass_mean_mixed,buildUpPlayPassingClass_mean_short, buildUpPlayPassingClass_mean_long
# unique value and counts of buildUpPlayPositioningClass
team_data['buildUpPlayPositioningClass'].value_counts()
# mean points_per_game of each category value
buildUpPlayPositioningClass_mean_organised=team_data.query('buildUpPlayPositioningClass=="Organised"')['points_per_game'].mean()
buildUpPlayPositioningClass_mean_freeform=team_data.query('buildUpPlayPositioningClass=="Free Form"')['points_per_game'].mean()
buildUpPlayPositioningClass_mean_organised, buildUpPlayPositioningClass_mean_freeform
# unique value and counts of chanceCreationPassingClass
team_data['chanceCreationPassingClass'].value_counts()
# mean points_per_game of each category value
chanceCreationPassingClass_mean_normal=team_data.query('chanceCreationPassingClass=="Normal"')['points_per_game'].mean()
chanceCreationPassingClass_mean_risky=team_data.query('chanceCreationPassingClass=="Risky"')['points_per_game'].mean()
chanceCreationPassingClass_mean_safe=team_data.query('chanceCreationPassingClass=="Safe"')['points_per_game'].mean()
chanceCreationPassingClass_mean_normal, chanceCreationPassingClass_mean_risky, chanceCreationPassingClass_mean_safe
# unique value and counts of chanceCreationCrossingClass
team_data['chanceCreationCrossingClass'].value_counts()
# mean points_per_game of each category value
chanceCreationCrossingClass_mean_normal=team_data.query('chanceCreationCrossingClass=="Normal"')['points_per_game'].mean()
chanceCreationCrossingClass_mean_lots=team_data.query('chanceCreationCrossingClass=="Lots"')['points_per_game'].mean()
chanceCreationCrossingClass_mean_little=team_data.query('chanceCreationCrossingClass=="Little"')['points_per_game'].mean()
chanceCreationCrossingClass_mean_normal, chanceCreationCrossingClass_mean_lots, chanceCreationCrossingClass_mean_little
# unique value and counts of chanceCreationShootingClass
team_data['chanceCreationShootingClass'].value_counts()
# mean points_per_game of each category value
chanceCreationShootingClass_mean_normal=team_data.query('chanceCreationShootingClass=="Normal"')['points_per_game'].mean()
chanceCreationShootingClass_mean_lots=team_data.query('chanceCreationShootingClass=="Lots"')['points_per_game'].mean()
chanceCreationShootingClass_mean_little=team_data.query('chanceCreationShootingClass=="Little"')['points_per_game'].mean()
chanceCreationShootingClass_mean_normal, chanceCreationShootingClass_mean_lots, chanceCreationShootingClass_mean_little
# unique value and counts of chanceCreationPositioningClass
team_data['chanceCreationPositioningClass'].value_counts()
# mean points_per_game of each category value
chanceCreationPositioningClass_mean_organised=team_data.query('chanceCreationPositioningClass=="Organised"')['points_per_game'].mean()
chanceCreationPositioningClass_mean_freeform=team_data.query('chanceCreationPositioningClass=="Free Form"')['points_per_game'].mean()
chanceCreationPositioningClass_mean_organised,chanceCreationPositioningClass_mean_freeform
# unique value and counts of defencePressureClass
team_data['defencePressureClass'].value_counts()
# mean points_per_game of each category value
defencePressureClass_mean_Medium=team_data.query('defencePressureClass=="Medium"')['points_per_game'].mean()
defencePressureClass_mean_Deep=team_data.query('defencePressureClass=="Deep"')['points_per_game'].mean()
defencePressureClass_mean_High=team_data.query('defencePressureClass=="High"')['points_per_game'].mean()
defencePressureClass_mean_Medium,defencePressureClass_mean_Deep, defencePressureClass_mean_High
# unique value and counts of defenceAggressionClass
team_data['defenceAggressionClass'].value_counts()
# mean points_per_game of each category value
defenceAggressionClass_mean_Press=team_data.query('defenceAggressionClass=="Press"')['points_per_game'].mean()
defenceAggressionClass_mean_Double=team_data.query('defenceAggressionClass=="Double"')['points_per_game'].mean()
defenceAggressionClass_mean_Contain=team_data.query('defenceAggressionClass=="Contain"')['points_per_game'].mean()
defenceAggressionClass_mean_Press,defenceAggressionClass_mean_Double, defenceAggressionClass_mean_Contain
# unique value and counts of defenceTeamWidthClass
team_data['defenceTeamWidthClass'].value_counts()
# mean points_per_game of each category value
defenceTeamWidthClass_mean_Normal=team_data.query('defenceTeamWidthClass=="Normal"')['points_per_game'].mean()
defenceTeamWidthClass_mean_Wide=team_data.query('defenceTeamWidthClass=="Wide"')['points_per_game'].mean()
defenceTeamWidthClass_mean_Narrow=team_data.query('defenceTeamWidthClass=="Narrow"')['points_per_game'].mean()
defenceTeamWidthClass_mean_Normal,defenceTeamWidthClass_mean_Wide, defenceTeamWidthClass_mean_Narrow
# unique value and counts of defenceDefenderLineClass
team_data['defenceDefenderLineClass'].value_counts()
# mean points_per_game of each category value
defenceDefenderLineClass_mean_Cover=team_data.query('defenceDefenderLineClass=="Cover"')['points_per_game'].mean()
defenceDefenderLineClass_mean_OffsideTrap=team_data.query('defenceDefenderLineClass=="Offside Trap"')['points_per_game'].mean()
defenceDefenderLineClass_mean_Cover,defenceDefenderLineClass_mean_OffsideTrap
#histogram of points_per_game of each category value of chanceCreationPositioningClass
team_data.query('chanceCreationPositioningClass=="Organised"')['points_per_game'].hist(color='green',alpha=0.5,bins=20, density=True, label='Organised')
team_data.query('chanceCreationPositioningClass=="Free Form"')['points_per_game'].hist(color='red',alpha=0.5,bins=20, density=True, label='Free Form')
plt.legend()
plt.title('points_per_game of chanceCreationPositioningClass categories')
plt.xlabel('points_per_game');
# t-test on chanceCreationPositioningClass Organized and Free Form
stats.ttest_ind(team_data.query('chanceCreationPositioningClass=="Organised"')['points_per_game'],team_data.query('chanceCreationPositioningClass=="Free Form"')['points_per_game'],equal_var=False)
#histogram of points_per_game of each category value of buildUpPlayPassingClass
team_data.query('buildUpPlayPassingClass=="Mixed"')['points_per_game'].hist(color='green',alpha=0.5,bins=20, density=True, label='Mixed')
team_data.query('buildUpPlayPassingClass=="Short"')['points_per_game'].hist(color='red',alpha=0.5,bins=20, density=True, label='Short')
team_data.query('buildUpPlayPassingClass=="Long"')['points_per_game'].hist(color='blue',alpha=0.5,bins=20, density=True, label='Long')
plt.legend()
plt.title('points_per_game of buildUpPlayPassingClass categories')
plt.xlabel('points_per_game');
# t-test on buildUpPlayPassingClass Mixed and Short
stats.ttest_ind(team_data.query('buildUpPlayPassingClass=="Mixed"')['points_per_game'],team_data.query('buildUpPlayPassingClass=="Short"')['points_per_game'],equal_var=False)
# t-test on buildUpPlayPassingClass Mixed and Long
stats.ttest_ind(team_data.query('buildUpPlayPassingClass=="Mixed"')['points_per_game'],team_data.query('buildUpPlayPassingClass=="Long"')['points_per_game'],equal_var=False)
# t-test on buildUpPlayPassingClass Short and Long
stats.ttest_ind(team_data.query('buildUpPlayPassingClass=="Short"')['points_per_game'],team_data.query('buildUpPlayPassingClass=="Long"')['points_per_game'],equal_var=False)
#histogram of points_per_game of each category value of buildUpPlayPositioningClass
team_data.query('buildUpPlayPositioningClass=="Organised"')['points_per_game'].hist(color='green',alpha=0.5,bins=20, density=True, label='Organised')
team_data.query('buildUpPlayPositioningClass=="Free Form"')['points_per_game'].hist(color='red',alpha=0.5,bins=20, density=True, label='Free Form')
plt.legend()
plt.title('points_per_game of chanceCreationPositioningClass categories')
plt.xlabel('points_per_game');
# t-test on buildUpPlayPositioningClass Organised and Free Form
stats.ttest_ind(team_data.query('buildUpPlayPositioningClass=="Organised"')['points_per_game'],team_data.query('buildUpPlayPositioningClass=="Free Form"')['points_per_game'],equal_var=False)
###Count individual players number of penalties
player_data.groupby('player_name')['penalties'].sum().sort_values(ascending=False).head()
###find player age range
player_data['age'].describe()
#plot a histogram of player age distribution
player_data.age.hist(bins=50, alpha=0.9, label='player ages')
plt.xlabel('player age')
plt.ylabel('number of player')
plt.title('distribution of player ages')
plt.legend();
#create listw of columns that are quantitative
num_cols=['potential', 'crossing', 'finishing', 'heading_accuracy','short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy','long_passing', 'ball_control', 'acceleration', 'sprint_speed','agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina','strength', 'long_shots', 'aggression', 'interceptions', 'positioning','vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle','gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning','gk_reflexes', 'height', 'weight','age']
# create a list containing Pearson's correlation between 'overall_rating' with each column in num_cols
correlations =pd.Series([ player_data['overall_rating'].corr(player_data[f]) for f in num_cols ], index=num_cols)
correlations.sort_values(ascending=False)
# scatter plot of overall_rating vs reactions
plt.scatter(player_data['reactions'], player_data['overall_rating'])
plt.xlabel('player overall_rating')
plt.ylabel('reaction')
plt.title('overall_rating vs reaction');
# scatter plot of overall_rating vs potential
plt.scatter(player_data['potential'], player_data['overall_rating'])
plt.xlabel('player overall_rating')
plt.ylabel('potential')
plt.title('overall_rating vs potential');
# scatter plot of overall_rating vs short_passing
plt.scatter(player_data['short_passing'], player_data['overall_rating'])
plt.xlabel('player overall_rating')
plt.ylabel('short_passing')
plt.title('overall_rating vs short_passing');
# scatter plot of overall_rating vs ball_control
plt.scatter(player_data['ball_control'], player_data['overall_rating'])
plt.xlabel('player overall_rating')
plt.ylabel('ball_control')
plt.title('overall_rating vs ball_control');
# scatter plot of overall_rating vs long_passing
plt.scatter(player_data['long_passing'], player_data['overall_rating'])
plt.xlabel('player overall_rating')
plt.ylabel('long_passing')
plt.title('overall_rating vs long_passing');
#Pearson correlation between overall_rating and reactions
stats.pearsonr(player_data['overall_rating'],player_data['reactions'])
#pearson correlation between overall_rating and potential
stats.pearsonr(player_data['overall_rating'],player_data['potential'])
#pearson correlation between overall_rating and short_passing
stats.pearsonr(player_data['overall_rating'],player_data['short_passing'])
#pearson correlation between overall_rating and ball_control
stats.pearsonr(player_data['overall_rating'],player_data['ball_control'])
#pearson correlation between overall_rating and long_passing
stats.pearsonr(player_data['overall_rating'],player_data['long_passing'])
#check unique values of preferred_foot
player_data['preferred_foot'].unique()
#histograms of the overall_ratings of players who prefer left foot and who prefer right food are completely overlapped
player_data.query('preferred_foot=="left"')['overall_rating'].hist(color='red', alpha=0.5,label='left foot', bins=50, density=True)
player_data.query('preferred_foot=="right"')['overall_rating'].hist(color='blue', alpha=0.5,label='right foot', bins=50, density=True)
plt.xlabel('overall_rating')
plt.xlim(50,90)
plt.ylabel('percentage of players')
plt.title('overall_rating vs preferred_foot')
plt.legend();
stats.ttest_ind(player_data.query('preferred_foot=="left"')['overall_rating'], player_data.query('preferred_foot=="right"')['overall_rating'],equal_var=False)
#check unique values of the categorical attribute "attacking_work_rate".
#Majority of the players fall into medium, high, low or None category. So I will only consider players who are in these four values.
player_data.attacking_work_rate.value_counts()
#create seperate dataframe for the 4 unique values of attacking_work_rate, that represent majority of players
medium_attack=player_data.query('attacking_work_rate=="medium"')
high_attack=player_data.query('attacking_work_rate=="high"')
low_attack=player_data.query('attacking_work_rate=="low"')
None_attack=player_data.query('attacking_work_rate=="None"')
#first check in histogram of high_attack and low_attack if there is visible difference.
high_attack['overall_rating'].hist(color='red', alpha=0.5,label='high attacking_work_rate', bins=50, weights=np.ones_like(high_attack['overall_rating'])/len(high_attack['overall_rating']))
low_attack['overall_rating'].hist(color='blue', alpha=0.3,label='low attacking_work_rate', bins=50, weights=np.ones_like(low_attack['overall_rating'])/len(low_attack['overall_rating']))
plt.xlabel('overall_rating')
plt.xlim(50,90)
plt.ylabel('percentage of players')
plt.title('histogram of overall_rating vs. attacking_work_rate')
plt.legend();
stats.ttest_ind(high_attack['overall_rating'], medium_attack['overall_rating'],equal_var=False)
stats.ttest_ind(high_attack['overall_rating'], low_attack['overall_rating'],equal_var=False)
stats.ttest_ind(high_attack['overall_rating'], None_attack['overall_rating'],equal_var=False)
stats.ttest_ind(medium_attack['overall_rating'], low_attack['overall_rating'],equal_var=False)
stats.ttest_ind(medium_attack['overall_rating'], None_attack['overall_rating'],equal_var=False)
stats.ttest_ind(low_attack['overall_rating'], None_attack['overall_rating'],equal_var=False)
#check unique values of defensive_work_rate
player_data.defensive_work_rate.value_counts()
#I will only consider the players whose value of defensive_work_rate are medium, high or low
medium_defense=player_data.query('defensive_work_rate=="medium"')
high_defense=player_data.query('defensive_work_rate=="high"')
low_defense=player_data.query('defensive_work_rate=="low"')
#plot a hightogram of high and low defensive_work_rate
high_defense['overall_rating'].hist(color='red', alpha=0.5,label='high defensive_work_rate', bins=50, weights=np.ones_like(high_defense['overall_rating'])/len(high_defense['overall_rating']) )
low_defense['overall_rating'].hist(color='blue', alpha=0.3,label='low defensive_work_rate', bins=50, weights=np.ones_like(low_defense['overall_rating'])/len(low_defense['overall_rating']) )
plt.xlabel('overall_rating')
plt.ylabel('percentage of players')
plt.title('player defensive_work_rate vs overall_rating')
plt.xlim(50,90)
plt.legend();
stats.ttest_ind(high_defense['overall_rating'], medium_defense['overall_rating'], equal_var=False)
stats.ttest_ind(high_defense['overall_rating'], low_defense['overall_rating'], equal_var=False)
stats.ttest_ind(medium_defense['overall_rating'], low_defense['overall_rating'], equal_var=False)
#close connection to database
conn.close()