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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('/kaggle/input/data.csv',index_col=0)
df.head()
df.columns # some of these are unnecessary, and we'll get rid of them
df.drop(columns=['ID','Photo','Flag','Club Logo','Work Rate','Weak Foot','Preferred Foot','Body Type','Real Face',

                 'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until',

                 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',

       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',

       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB'], inplace=True)
df.dropna(subset=['Height','Weight'], inplace=True) # drop all rows where either of height or weight is missing
def to_inches(row):

    lst = row.split("'")

    lst = pd.to_numeric(lst)

    new_height = (lst[0] * 12) + lst[1]

    return new_height
df['Height'] = df['Height'].apply(to_inches)
def weight_in_num(row):

    lst = row.split('lbs')

    lst = pd.to_numeric(lst)

    new_weight = lst[0]

    return new_weight
df['Weight'] = df['Weight'].apply(weight_in_num)
def value_change(row): #converting everything in terms of 'thousands of euros'

    if row[-1] == 'M':

        row = row[1:-1] # getting rid of euro and million sign

        num = pd.to_numeric(row)

        num = num * 1000

        return num

    else:

        return pd.to_numeric(row[1:-1])
df['Value'] = df['Value'].apply(value_change)
def release_clause_change(row):

    if row[-1] == 'M':

        row = row[1:-1]

        num = pd.to_numeric(row)

        num = num * 1000

        return num

    else:

        return pd.to_numeric(row[1:-1])
df['Release Clause'] = df[df['Release Clause'].notnull()]['Release Clause'].apply(release_clause_change)
def wage_change(row): # make wages in thousands of euros

    return pd.to_numeric(row[1:-1])
df['Wage'] = df['Wage'].apply(wage_change)
df.head(2)
sns.scatterplot(y='Special',x='SprintSpeed',data=df) # players with more 'special' rating tend to be 'speedier'
sns.scatterplot(x='Age',y='Value',data=df) # almost represents a normal distribution, which is good to see. Mr. Gauss would be a man.
top_class_players = df[df['Value'] > 40000] # players with a market value of more than 40 million euros

top_class_players.head()
top_class_players.groupby('Club')['Name'].count().sort_values(ascending=False) # No wonder Madrid won 4 out of the last 6 Champions Leagues!
sns.boxplot(y='Vision',x='Height',data=df) # reaffirms the notion that shorter players are more creative, for eg. Cazorla, Messi, Mata, Silva etc.
sns.boxplot(y='Balance',x='Height',data=df) # and this is why shorter players are quicker, they can balance themseles better!
sns.boxplot(y='Strength',x='Height',data=df) # reaffirms the notion that shorter players are kinda weaker in duels
df.corr()['Age'] # no visible correlation anywhere! reaffirms the notion that age is just a number!
df_raw_players_with_potential = df[(df['Overall'] <= 70) & (df['Potential'] > 80)]
df_raw_players_with_potential.sort_values(by='Potential',ascending=False)  # watch out for these players!
df_raw_players_with_potential.groupby('Club')['Club'].count().sort_values(ascending=False)
df_worldcup2022_probables = df[(df.Age<=30) & (df.Potential>75)]
df_worldcup2022_probables.sample()
df_worldcup2022_probables.groupby('Nationality')['Potential'].count()
df_worldcup2022_probables.groupby('Nationality').filter(lambda x: len(x) > 20)['Nationality'].value_counts()
df_worldcup2022_probables = df_worldcup2022_probables[df_worldcup2022_probables.groupby('Nationality')['Nationality'].transform('size') > 20]
df_worldcup2022_probables.groupby('Nationality')['Potential'].mean().sort_values(ascending=False)