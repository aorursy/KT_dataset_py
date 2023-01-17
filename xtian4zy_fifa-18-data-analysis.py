# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# basic operations
import numpy as np
import pandas as pd 

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


# reading the complete data and also checking the computation time

%time df = pd.read_csv('../input/fifa19/data.csv')

print(df.shape)
#See the sample of our data
df.head(5)
#Lets check for the columns
df.columns
#Delete unwanted columns
df.drop(columns=['Unnamed: 0','Photo','Club Logo','Flag'], inplace = True)
df.columns
#Check for columns with Missing data
df.loc[:, df.isnull().any()].columns
#Lets see how many missing values in total
df.isnull().values.sum()
#Get the columns with missing values and the percentages
df.loc[:,list(df.loc[:,df.isnull().any()].columns)].isnull().sum()/(len(df))*100
#Handling Missing Values
df['ShortPassing'].fillna(df['ShortPassing'].mean(), inplace = True)
df['Volleys'].fillna(df['Volleys'].mean(), inplace = True)
df['Dribbling'].fillna(df['Dribbling'].mean(), inplace = True)
df['Curve'].fillna(df['Curve'].mean(), inplace = True)
df['FKAccuracy'].fillna(df['FKAccuracy'], inplace = True)
df['LongPassing'].fillna(df['LongPassing'].mean(), inplace = True)
df['BallControl'].fillna(df['BallControl'].mean(), inplace = True)
df['HeadingAccuracy'].fillna(df['HeadingAccuracy'].mean(), inplace = True)
df['Finishing'].fillna(df['Finishing'].mean(), inplace = True)
df['Crossing'].fillna(df['Crossing'].mean(), inplace = True)
df['Weight'].fillna('200lbs', inplace = True)
df['Contract Valid Until'].fillna(2019, inplace = True)
df['Height'].fillna("5'11", inplace = True)
df['Loaned From'].fillna('None', inplace = True)
df['Joined'].fillna('Jul 1, 2018', inplace = True)
df['Jersey Number'].fillna(8, inplace = True)
df['Body Type'].fillna('Normal', inplace = True)
df['Position'].fillna('ST', inplace = True)
df['Club'].fillna('No Club', inplace = True)
df['Work Rate'].fillna('Medium/ Medium', inplace = True)
df['Skill Moves'].fillna(df['Skill Moves'].median(), inplace = True)
df['Weak Foot'].fillna(3, inplace = True)
df['Preferred Foot'].fillna('Right', inplace = True)
df['International Reputation'].fillna(1, inplace = True)
df['Wage'].fillna('€150K', inplace = True)
#How many players names are listed?
#Lets check how many unique players are listed
total_player_list = pd.value_counts(df.Name).sum()
print('Total names of players on list =', total_player_list)
#pd.unique(df.Name).sum()
total_unique_players = len(df['Name'].unique()) 
print('Total unique names of players on list =', total_unique_players)
print('---Lets see the number of times a particular name shows up--->'  )
print ('Players Count\n', pd.value_counts(df.Name))
#Lets format the 'Wage' column so we can make it numeric 
df['wage_new'] = df['Wage'].str.replace(r'\D', '').astype(int)
#Check to see our new column data
df['wage_new'][:5]
#Get players and earnings column only
df_wage_only = df.loc[:, ['Name', 'Wage', 'wage_new']]

#Sort the dataFrame top down
df_wage_only = df_wage_only.sort_values(by=['wage_new'], ascending=False)

#Top 10 highest earners
print('--------Top 10 Earners--------->')
print(df_wage_only[:10])
#Bar Chart for Top 10 earners
players = df_wage_only['Name'][:10]
y_pos = np.arange(len(players))
wage =  df_wage_only['wage_new'][:10]
plt.figure(figsize=(20,5))
plt.bar(y_pos, wage, align='center', alpha=0.5)
plt.xticks(y_pos, players)
plt.ylabel('Wage')
plt.title('Top 10 Earning Players')

plt.show()