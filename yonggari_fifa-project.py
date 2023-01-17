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

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')







fifa = pd.read_csv('../input/fifa19/data.csv')



#SEE IF NaN VALUES EXISTS

print(fifa.isna().any())



print(fifa.info())

print(fifa.head())

print(fifa.dtypes)

print(fifa.describe())









#FILL MISSING VALUES

fifa['ShortPassing'].fillna(fifa['ShortPassing'].mean(), inplace = True)

fifa['Volleys'].fillna(fifa['Volleys'].mean(), inplace = True)

fifa['Dribbling'].fillna(fifa['Dribbling'].mean(), inplace = True)

fifa['Curve'].fillna(fifa['Curve'].mean(), inplace = True)

fifa['FKAccuracy'].fillna(fifa['FKAccuracy'], inplace = True)

fifa['LongPassing'].fillna(fifa['LongPassing'].mean(), inplace = True)

fifa['BallControl'].fillna(fifa['BallControl'].mean(), inplace = True)

fifa['HeadingAccuracy'].fillna(fifa['HeadingAccuracy'].mean(), inplace = True)

fifa['Finishing'].fillna(fifa['Finishing'].mean(), inplace = True)

fifa['Crossing'].fillna(fifa['Crossing'].mean(), inplace = True)

fifa['Weight'].fillna('200lbs', inplace = True)

fifa['Contract Valid Until'].fillna(2019, inplace = True)

fifa['Height'].fillna("5'11", inplace = True)

fifa['Loaned From'].fillna('None', inplace = True)

fifa['Joined'].fillna('Jul 1, 2018', inplace = True)

fifa['Jersey Number'].fillna(8, inplace = True)

fifa['Body Type'].fillna('Normal', inplace = True)

fifa['Position'].fillna('ST', inplace = True)

fifa['Club'].fillna('No Club', inplace = True)

fifa['Work Rate'].fillna('Medium/ Medium', inplace = True)

fifa['Skill Moves'].fillna(fifa['Skill Moves'].median(), inplace = True)

fifa['Weak Foot'].fillna(3, inplace = True)

fifa['Preferred Foot'].fillna('Right', inplace = True)

fifa['International Reputation'].fillna(1, inplace = True)

fifa['Wage'].fillna('€200K', inplace = True)

















#AGE BREAKDOWN

print(fifa['Age'].value_counts())





#NATIONALITY BREAKDOWN

print(fifa['Nationality'].value_counts().head(10))















#KOREAN PLAYERS

def kor(x):

    return fifa[fifa['Nationality'] == x][['Name','Overall','Potential','Position']]

print(kor('Korea Republic'))







#REAL MADRID CLUB BREAKDOWN

def club(x):

    return fifa[fifa['Club'] == x][['Name','Jersey Number','Position','Overall','Nationality','Age','Wage',

                                    'Value','Contract Valid Until']]

print(club('Real Madrid'))

















#RIGHTY? OR LEFTY?

plt.rcParams['figure.figsize'] = (10, 5)

sns.countplot(fifa['Preferred Foot'], palette = 'pink')

plt.title('Dominant Foot', fontsize = 20)

plt.show()



#TOP 10 LEFT FOOTED

print(fifa[fifa['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(10))



#TOP 10 RIGHT FOOTED

print(fifa[fifa['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10))













#SKILL MOVES PLOT

plt.figure(figsize = (10, 8))

ax = sns.countplot(x = 'Skill Moves', data = fifa, palette = 'pastel')

ax.set_title(label = 'Count of Number of Skill Moves', fontsize = 20)

ax.set_xlabel(xlabel = 'Number of Skill Moves', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()













#SKILL MOVES BY HEIGHT

plt.figure(figsize = (13, 8))

ax = sns.countplot(x = 'Height', data = fifa, palette = 'dark')

ax.set_title(label = 'Count of Number of Skill Moves by Height', fontsize = 20)

ax.set_xlabel(xlabel = 'Height in Foot per inch', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()



















#HISTOGRAM OF AGE OF PLAYERS

sns.set(style = "dark", palette = "colorblind", color_codes = True)

x = fifa.Age

plt.figure(figsize = (15,8))

ax = sns.distplot(x, bins = 58, kde = False, color = 'g')

ax.set_xlabel(xlabel = "Player\'s age", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)

ax.set_title(label = 'Player Age Breakdown', fontsize = 20)

plt.show()

















#TOP 10 CLUBS THAT HAVE THE HIGHEST NUMBER OF DIFFERNENT NATIONALITIES

print(fifa.groupby(fifa['Club'])['Nationality'].nunique().sort_values(ascending = True).head(10))





#TOP CLUBS WITH LOWEST FOREIGNER PLAYERS

print(fifa.groupby(['Club'])['Nationality'].nunique().sort_values().head())





#TOP 5 TEAMS WITH BEST PLAYERS

print(fifa.groupby(['Club'])['Overall'].max().sort_values(ascending = False).head())


