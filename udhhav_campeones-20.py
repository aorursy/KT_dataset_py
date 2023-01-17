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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

sns.set_style(style='darkgrid')

pd.set_option('display.max_columns',None)
df = pd.read_csv('../input/fifa19/data.csv')

df.head()
df.info()
df.shape
df.columns
df.drop(columns=['Unnamed: 0','ID','Photo','Flag','Club Logo'],axis=1,inplace=True)
df.shape
df.head()
plt.figure(figsize =(10,10))

sns.countplot(df['Age'])
df_young =df[df['Age']<=20]
df_young.shape[0]
df_young.sample()
print('There are around {}% young players.'.format(np.round((len(df_young)/len(df))*100,2)))
youngest = df.sort_values('Age', ascending = True)[['Name', 'Nationality', 'Age','Club','Overall','Position']].head(30)

youngest.set_index('Name', inplace=True)

print(youngest)
df_strikers = df[df['Position']=='ST']
df_strikers.shape[0]
df_strikers.sample(5)
df_strikers.columns
sns.countplot(df_strikers['Preferred Foot'])
df.columns
plt.figure(figsize =(25,16))

matrix = np.triu(df_strikers[['Age', 'Overall', 'Potential', 'Value', 'Wage',

                'Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl', 

                'Body Type','Composure', 'Crossing','Dribbling', 'FKAccuracy', 'Finishing', 

                'HeadingAccuracy', 'Interceptions','International Reputation',

                'Joined', 'Jumping', 'LongPassing', 'LongShots',

                'Marking', 'Penalties', 'Position', 'Positioning',

                'ShortPassing', 'ShotPower', 'Skill Moves', 'SlidingTackle',

                'SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Vision',

                'Volleys']].corr())

sns.heatmap(df_strikers[['Age', 'Overall', 'Potential', 'Value', 'Wage',

                'Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl', 

                'Body Type','Composure', 'Crossing','Dribbling', 'FKAccuracy', 'Finishing', 

                'HeadingAccuracy', 'Interceptions','International Reputation',

                'Joined', 'Jumping', 'LongPassing', 'LongShots',

                'Marking', 'Penalties', 'Position', 'Positioning',

                'ShortPassing', 'ShotPower', 'Skill Moves', 'SlidingTackle',

                'SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Vision',

                'Volleys']].corr(), annot=True, mask=matrix,cmap='Blues')
features_y = ['Agility', 'Balance', 'Dribbling', 'SprintSpeed','BallControl']

for ft in features_y:

    sns.regplot(x='Acceleration',y=ft,data=df_strikers)

    plt.show()
my_clubs = [ 'Real Madrid', 'Paris Saint-Germain', 'FC Barcelona', 'Juventus', 'Manchester United']

df_club = df.loc[df['Club'].isin(my_clubs) & df['Age']]

df_club.head()
plt.figure(figsize =(10,10))

sns.barplot(x=df_club['Club'], y=df_club['Overall'], palette="rocket")
plt.figure(figsize =(10,10))

sns.barplot(x=df_club['Club'], y=df_club['Age'], palette="rocket")
my_team = df[df['Club']=='Real Madrid']

my_team.shape
my_team.sample(5)
my_team.sort_values(by='Age').head()
my_team.nlargest(5,'Potential')
plt.figure(figsize =(15,10))

sns.countplot(my_team['Nationality'])
plt.figure(figsize =(15,10))

sns.countplot(my_team['Position'])