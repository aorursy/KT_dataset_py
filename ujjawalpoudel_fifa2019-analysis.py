# Data manipulation

import numpy as np

import pandas as pd



# Data visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Date

import datetime



# Counter

from collections import Counter as counter
# Read CSV file from local disk 

df_fifa=pd.read_csv("../input/data.csv") 

df_fifa.head() # Give top 5 rows of given dataset
# Find how many null value is there in particular column

df_fifa.isnull().sum()
# Describe our dataset with some statistical terms

df_fifa.describe()
# Choose columns for analysis

chosen_columns = ['ID', 'Name', 'Age', 'Nationality','Overall', 'Potential', 'Club','Value', 'Wage', 'Special',

                  'Preferred Foot', 'International Reputation', 'Weak Foot','Skill Moves', 'Work Rate',

                  'Body Type', 'Position','Jersey Number','Height', 'Weight', 'LS', 'ST', 'RS', 'LW',

                  'LF', 'CF', 'RF', 'RW','LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM',

                  'LWB', 'LDM','CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Crossing','Finishing',

                  'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling','Curve', 'FKAccuracy', 

                  'LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility', 'Reactions', 

                  'Balance', 'ShotPower','Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

                  'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure','Marking', 

                  'StandingTackle', 'SlidingTackle', 'Release Clause','GKDiving','GKHandling','GKKicking',

                  'GKPositioning','GKReflexes','Joined','Contract Valid Until']
# Created DataFrame with chosen columns

# Use previous dataframe and use only selected columns as we mention in above code



df = pd.DataFrame(df_fifa, columns = chosen_columns)

df.head()
# Correlation heatmap



plt.rcParams['figure.figsize']=(25,16)

hm=sns.heatmap(df[['Age', 'Overall', 'Potential', 'Value', 'Wage',

                'Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl', 

                'Body Type','Composure', 'Crossing','Dribbling', 'FKAccuracy', 'Finishing', 

                'HeadingAccuracy', 'Interceptions','International Reputation',

                'Joined', 'Jumping', 'LongPassing', 'LongShots',

                'Marking', 'Penalties', 'Position', 'Positioning',

                'ShortPassing', 'ShotPower', 'Skill Moves', 'SlidingTackle',

                'SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Vision',

                'Volleys']].corr(), annot = True, linewidths=.5, cmap='Blues')

hm.set_title(label='Heatmap of dataset', fontsize=20);
# Histogram: number of players's age



sns.set(style ="dark") # style dark means there is no any square box on histogram plot

plt.figure(figsize=(12,6))

ax = sns.distplot(df.Age, bins = 58, kde = False) #  bins => gap between two bar graph

ax.set_xlabel(xlabel="Player\'s age", fontsize=16)

ax.set_ylabel(ylabel='Number of players', fontsize=16)

ax.set_title(label='Histogram of players age', fontsize=20)

plt.show()
# The ten eldest players



eldest = df.sort_values('Age', ascending = False)[['Name','Club', 'Nationality', 'Age']].head(10)

eldest.set_index('Name', inplace=True)

print(eldest)
# The ten youngest players



eldest = df.sort_values('Age', ascending = True)[['Name','Club', 'Nationality', 'Age']].head(10)

eldest.set_index('Name', inplace=True)

print(eldest)
# The longest membership in the club



now = datetime.datetime.now()

df['Join_year'] = df.Joined.dropna().map(lambda x: x.split(',')[1].split(' ')[1])

df['Years_of_member'] = (df.Join_year.dropna().map(lambda x: now.year - int(x))).astype('int').dropna()

membership = df[['Name', 'Club', 'Years_of_member']].sort_values(by = 'Years_of_member', ascending = False).dropna().head(10)

membership.set_index('Name', inplace=True)

membership
# Number of players in particular position



sns.set(style="darkgrid")

ax = sns.countplot(x = 'Position' ,data = df) 

# countplot is use for showing the counts of observations in each categorical bin using bars.

ax.set_title(label='Count of players on the position', fontsize=30);
# The best player per position



# idxmax() returns index where you get maximum value but max() returns particular maximum value not index



top_players=df.iloc[df.groupby(df['Position'])['Overall'].idxmax()][['Name', 'Position','Club']]

top_players.set_index('Name', inplace=True)

top_players
# Top three features per position



player_features = [

    'Acceleration', 'Aggression', 'Agility', 

    'Balance', 'BallControl', 'Composure', 

    'Crossing', 'Dribbling', 'FKAccuracy', 

    'Finishing', 'GKDiving', 'GKHandling', 

    'GKKicking', 'GKPositioning', 'GKReflexes', 

    'HeadingAccuracy', 'Interceptions', 'Jumping', 

    'LongPassing', 'LongShots', 'Marking', 'Penalties'

    ]



# It group the data according to Position and find mean of particular grouped columns having player_features



for i, val in df.groupby(df['Position'])[player_features].mean().iterrows():

    print('Position {}: {}, {}, {}'.format(i, *tuple(val.nlargest(3).index)))
# Top 10 left-footed players



left_foot=df[df['Preferred Foot'] == 'Left'][['Name','Overall','Club','Position','Nationality']].head(10)

left_foot.set_index('Name',inplace=True)

left_foot
# Top 10 right-footed players



right_foot=df[df['Preferred Foot'] == 'Right'][['Name','Overall','Club','Position','Nationality']].head(10)

right_foot.set_index('Name',inplace=True)

right_foot
# The clubs, where have players mainly from one country



clubs_coherency = pd.Series()

for club, players in df.groupby(['Club'])['Nationality'].count().items():

    coherency = df[df['Club'] == club].groupby(['Nationality'])['Club'].count().max() / players * 100

    clubs_coherency[club] = coherency



clubs_coherency.sort_values(ascending = False).head(20)
# The clubs with largest number of different countries players

# nunique => count distinct observations over requested axis



df.groupby(['Club'])['Nationality'].nunique().sort_values(ascending = False).head(10)
# The value has some non numeric mark(K or M) so extract rigth value

# Calculate value of club with summation of every player's value from particular club



def value_to_int(df_value):

    try:

        value = float(df_value[1:-1]) # This return 110.5 from €110.5M

        suffix = df_value[-1:] # This return M or K

        if suffix == 'M':

            value = value * 1000000

        elif suffix == 'K':

            value = value * 1000

    except ValueError:

        value = 0

    return value



df['Value_float'] = df['Value'].apply(value_to_int)



# Top five the most expensive clubs

df.groupby(['Club'])['Value_float'].sum().sort_values(ascending = False).head(10)
# Top ten the less expensive clubs



df.groupby(['Club'])['Value_float'].sum().sort_values().head(10)
# Maximum number of player from particular country

# Counter return as dictionary that is key:'Country_Name' values:'Total_Number_Of_Players'

# most_common return list in descending order with respect to Total_Number_Of_Players

# most_common()[:11] this return top 10 



plt.figure(1 , figsize = (15 , 7))

countries = []

c = counter(df['Nationality']).most_common()[:11]

for n in range(11):

    countries.append(c[n][0])



# value_counts() return a series containing counts of unique values



sns.countplot(x  = 'Nationality' ,

              data = df[df['Nationality'].isin(countries)] ,

              order  = df[df['Nationality'].isin(countries)]['Nationality'].value_counts().index , 

             palette = 'rocket') 

plt.xticks(rotation = 90)

plt.title('Maximum number footballers belong to which country' )

plt.show()
# Player with great shot power



gshot=df.sort_values(by = 'ShotPower' , ascending = False)[['Name' , 'Club' , 'Nationality' ,'ShotPower' ]].head(10)

gshot.set_index('Name',inplace=True)

gshot
# Player with best long pass



blpass=df.sort_values(by = 'LongPassing' , ascending = False)[['Name' , 'Club' , 'Nationality' , 'LongPassing']].head(10)

blpass.set_index('Name',inplace=True)

blpass
# Player with great Vision



gvision=df.sort_values(by = 'Vision' , ascending = False)[['Name' , 'Club' , 'Nationality' ,'Vision' ]].head(10)

gvision.set_index('Name',inplace=True)

gvision
# Quality should have in players having particular position



df_postion  = pd.DataFrame()

for position_name, features in df.groupby(df['Position'])[player_features].mean().iterrows():

    top_features = dict(features.nlargest(5))

    df_postion[position_name] = tuple(top_features)

df_postion
# Display name of player having age less than 25 years , Overall rating is more than 75 and from country Brazil



filtering_data = df.Nationality == "Brazil"

filtering_data2 = df.Age < 25

filtering_data3 = df.Overall > 75

best_brazil=(df[filtering_data & filtering_data2 & filtering_data3].sort_values(by = 'Overall' , ascending = False))[['Name','Age','Club','Nationality']]

best_brazil.set_index('Name',inplace=True)

best_brazil