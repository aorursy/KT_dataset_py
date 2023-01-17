#Importing the Libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#Turning off the warnings

import warnings

warnings.filterwarnings('ignore')
#To display all the columns of the dataframe

pd.set_option('display.max_columns',70)
#Reading the FIFA20 dataset

fifa = pd.read_csv("/kaggle/input/fifa-player-stats-database/FIFA20_official_data.csv")
#Dimension of the dataset

fifa.shape
#head of the dataset

fifa.head()
#columns of the dataframe

fifa.columns
#INFO() function

fifa.info()
#Some initial stats for the df

fifa.describe()
#Copy of our pulled in df

fifa_copy = fifa.copy()
fifa.head()
#Dropping some of the columns - ID, Photo, Flag, Club Logo

fifa.drop(['ID','Photo','Flag','Club Logo'],axis=1,inplace=True)
#Dropping Real Face column

fifa.drop(['Real Face'],axis=1,inplace=True)
#Filtering for rows which have Loaned From column not NULL 

fifa.loc[~fifa['Loaned From'].isnull()][:5]
#Checking whether all values are defined in Euros or not

fifa.loc[fifa['Value'].str.startswith('€')].shape[0]
#Checking the same for Wages.

fifa.loc[fifa['Wage'].str.startswith('€')].shape[0]
#Splitting the value column to get just the numeric

fifa['Value'] = fifa['Value'].str.split('€')

fifa['Value'] = fifa['Value'].apply(lambda x:x[1])
#Splitting the wage column to get just the numeric

fifa['Wage'] = fifa['Wage'].str.split('€')

fifa['Wage'] = fifa['Wage'].apply(lambda x:x[1])
#Converting the player value in thousand Euros to Million Euros and then stripping the end denote.

fifa_value_K = fifa.loc[fifa['Value'].str.endswith('K')]

fifa_value_K['Value'] = fifa_value_K['Value'].apply(lambda x: x[:-1])

fifa_value_K['Value'] = fifa_value_K['Value'].astype('float64')

fifa_value_K['Value'] = fifa_value_K['Value'] / 1000
#Stripping the end denote for Million Euros Player value

fifa_value_M = fifa.loc[fifa['Value'].str.endswith('M')]

fifa_value_M['Value'] = fifa_value_M['Value'].apply(lambda x: x[:-1])

fifa_value_M['Value'] = fifa_value_M['Value'].astype('float64')
#Converting the player wage in thousand Euros to Million Euros and then stripping the end denote.

fifa_value_K['Wage'] = fifa_value_K['Wage'].apply(lambda x: x[:-1] if x.endswith('K') else x)

fifa_value_K['Wage'] = fifa_value_K['Wage'].astype('float64')

fifa_value_K['Wage'] = fifa_value_K['Wage'] / 1000
fifa_value_M.loc[fifa_value_M['Wage'].str.endswith("M")]
#Converting the player wage in thousand Euros to Million Euros and then stripping the end denote.

fifa_value_M['Wage'] = fifa_value_M['Wage'].apply(lambda x: x[:-1] if x.endswith('K') else x)

fifa_value_M['Wage'] = fifa_value_M['Wage'].astype('float64')

fifa_value_M['Wage'] = fifa_value_M['Wage'] / 1000
#Concatenating both the splitted up dataframes

fifa2 = pd.concat([fifa_value_M,fifa_value_K])
fifa2.shape
fifa2['Position'] = fifa2['Position'].str.split(">")

fifa2['Position'] = fifa2['Position'].apply(lambda x:x[1])
fifa2['Weight'] = fifa2['Weight'].apply(lambda x : x[:-3])

fifa2['Weight'] = fifa2['Weight'].astype('int64')
foot = fifa2['Preferred Foot'].value_counts()

foot
foot_right = foot[0]/fifa2['Preferred Foot'].count()*100

foot_left = foot[1]/fifa2['Preferred Foot'].count()*100

foot_df = pd.DataFrame({'Percentage':[foot_right,foot_left]},index=['Right Foot','Left Foot'])

foot_df.style.background_gradient(cmap='Purples')
#Barplot for the classes

plt.title("Foot Preference")

sns.barplot(x=foot_df.index,y=foot_df['Percentage'],palette='Blues')

plt.show()
fifa2.Age.mean()
fifa2.head()
fifa_overall = fifa2.sort_values(['Overall'],ascending=False)[:10]

fifa_overall[['Name','Overall','Potential','Club','Preferred Foot','Position']].style.background_gradient(cmap='Greens')
fifa_potential = fifa2.sort_values(['Potential'],ascending=False)[:10]

fifa_potential[['Name','Overall','Potential','Club','Preferred Foot','Position']].style.background_gradient(cmap='Reds')
fifa2.sort_values(['Overall'],ascending=True)[:10]
fifa2.sort_values(['Potential'],ascending=True)[:10]
fifa2_germany = fifa2.loc[fifa2.Nationality=='Germany']

fifa2_germany.sort_values(['Overall'],ascending=False)[:5]
fifa2.loc[fifa2.Age >40]
plt.figure(figsize=(10,30))

sns.barplot(y=fifa2['Nationality'],x=fifa2['Age'])

plt.plot()
fifa2.sort_values(['Value'],ascending=False)[:5]
#To find the range of values for Weak Foot

print(fifa2['Weak Foot'].min())

print(fifa2['Weak Foot'].max())
#To find the range of values for Skill Moves

print(fifa2['Skill Moves'].min())

print(fifa2['Skill Moves'].max())
fifa2.loc[((fifa2['Weak Foot']==fifa2['Weak Foot'].max()) &(fifa2['Skill Moves']==fifa2['Skill Moves'].max()))]
fifa_pos = fifa2['Position'].value_counts()

plt.figure(figsize=(15,5))

sns.barplot(fifa_pos.index,fifa_pos.values)

plt.show()
fifa2.loc[fifa2['Position']=='CF'].count()[0]
fifa2.sort_values(['Weight'],ascending=False)[:1][['Name','Weight']]
plt.figure(figsize=(15,8))

sns.boxplot(y=fifa2['Value'],x=fifa2['Position'])

plt.plot()
fifa3 = fifa2[['Crossing', 'Finishing', 'HeadingAccuracy',

       'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',

       'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility',

       'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',

       'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision']]
plt.figure(figsize=(20,7))

sns.heatmap(round(fifa3.corr(),2),annot=True,cmap='Blues')

plt.show()