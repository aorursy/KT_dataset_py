# data manipulation

import pandas as pd

# data visualization

import matplotlib.pyplot as plt

import seaborn as sns 

# display propertice*-

# expand to see more column

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

# Date

import datetime

# Maps

# import geopandas as gpd

# import pycountry



from math import pi

%matplotlib inline
df_fifa19 = pd.read_csv('../input/data.csv')
df_fifa19.head()
df_fifa19.info()
list_col = list(df_fifa19)
print(list_col)
chosen_col = ['ID', 'Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value', 'Wage', 'Special', 'Preferred Foot',

             'Weak Foot','Position', 'Jersey Number','Height', 'Weight', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 

             'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 

             'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 

             'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle']
df_chosen = pd.DataFrame(df_fifa19, columns = chosen_col)


# Correlation heatmap

plt.rcParams['figure.figsize']=(25,16)

hm=sns.heatmap(df_chosen[['Age', 'Overall', 'Potential', 'Value', 'Wage',

                'Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl', 

                'Composure', 'Crossing','Dribbling', 'FKAccuracy', 'Finishing', 

                'HeadingAccuracy', 'Interceptions', 'Jumping', 'LongPassing', 'LongShots',

                'Marking', 'Penalties', 'Position', 'Positioning',

                'ShortPassing', 'ShotPower', 'SlidingTackle',

                'SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Vision',

                'Volleys']].corr(), annot = True, linewidths=.5, cmap='Blues')

hm.set_title(label='Heatmap of dataset', fontsize=20)

hm;
# Scatter plot to show correlation between BallContr'ol and other chosen features



def make_scatter(df):

    lists = ('Agility','Crossing','Dribbling','Finishing','LongPassing','LongShots', 'Positioning','Vision','Stamina')

    

    for index, list in enumerate(lists):

        plt.subplot(len(lists)/9 + 1, 9, index + 1)

        ax = sns.regplot(x='BallControl', y = list, data = df)

        

    plt.figure(figsize =(20,20))

    plt.subplots_adjust(hspace = 0.4)



make_scatter(df_chosen)
# Histogram to show number of plaer's age

sns.set(style="dark", palette="colorblind", color_codes=True)

x = df_chosen.Age

plt.figure(figsize=(12,8))

ax = sns.distplot(x, bins= 58,kde= False, color='g')

ax.set_xlabel(xlabel="Player\'s Age", fontsize=16)

ax.set_ylabel(ylabel='Number of players', fontsize= 16)

ax.set_title(label='Histogram of player age', fontsize=20)

plt.show()
eldest = df_chosen.sort_values('Age', ascending= False)[['Name','Nationality','Age']].head(10)

eldest.set_index('Name', inplace= True)

print(eldest)
youngest = df_chosen.sort_values('Age', ascending= True)[['Name','Nationality','Age']].head(10)

youngest.set_index('Name', inplace = True)

print(youngest)
# Compare 8 clubs in UEFA Final in relation to age

clubs = ('Juventus', 'Liverpool', 'Manchester United', 'Manchester City', 'FC Barcelona' ,'FC Porto', 'Tottenham Hotspur','Ajax')

df_club = df_chosen.loc[df_chosen['Club'].isin(clubs) & df_chosen['Age']]



fig, ax = plt.subplots()

fig.set_size_inches(20, 10)

ax = sns.violinplot(x="Club", y="Age", data=df_club);

ax.set_title(label='Distribution of age in some clubs', fontsize=20);

# All position

ax = sns.countplot(x = 'Position', data = df_chosen, palette = 'hls');

ax.set_title(label='Count of players on the position', fontsize=20);
# The best player in each position

best_player_position = df_chosen.iloc[df_chosen.groupby(df_chosen['Position'])['Overall'].idxmax()][['Name', 'Position','Overall']]

best_player_position.set_index('Name', inplace=True)

print(best_player_position)
# Top 5 left foot player

left_player = df_chosen[df_chosen['Preferred Foot'] == 'Left'][['Name', 'Position','Overall']].head()

left_player.set_index('Name', inplace= True)

print(left_player)
# Top 5 Right foot player

left_player = df_chosen[df_chosen['Preferred Foot'] == 'Right'][['Name', 'Position','Overall']].head()

left_player.set_index('Name', inplace= True)

print(left_player)