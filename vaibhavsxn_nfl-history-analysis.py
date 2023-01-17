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
data = pd.read_csv('/kaggle/input/superbowl-history-1967-2020/superbowl.csv')
data
data.Winner.mode()
pd.DataFrame(data.Winner.value_counts()).reset_index().rename(columns = {'index' : 'Team'})
import matplotlib.pyplot as plt

import seaborn as sns
pd.DataFrame(data.Loser.value_counts()).reset_index().rename(columns = {'index' : 'Team'})
pd.DataFrame(data.MVP.value_counts()).reset_index().rename(columns = {'index' : 'Player'})
data.info()
print('The maximum winning points scored in a game is' + str(' ') + str(data['Winner Pts'].max()))

print('The maximum losing points scored in a game is' + str(' ') + str(data['Loser Pts'].max()))
data['Score Difference'] = data['Winner Pts'] - data['Loser Pts']
data['Score Difference']
plt.figure(figsize=(40,18))

#subgrade_order = data['Score Difference']

sns.countplot(x='Score Difference',data=data,palette='coolwarm')


plt.figure(figsize=(50,18))

sns.countplot(x = 'Stadium', data = data)


plt.figure(figsize=(50,18))

sns.countplot(x = 'City', data = data)


plt.figure(figsize=(50,18))

sns.countplot(x = 'State', data = data)
finals = list(set(list(data.Winner) + list(data.Loser)))

print('The teams that have made it to the SuperBowl are: \n' + str(finals) + str('\n\n\n') + str('Total count: ') + str(len(finals)))
finals_ap = list(data.Winner) + list(data.Loser)
max(set(finals_ap), key=finals_ap.count)
data['Times'] = 1
data.groupby('Winner')['Times'].sum().reset_index(drop = False).sort_values(by = 'Times',ascending = False)
data.groupby('Loser')['Times'].sum().reset_index(drop = False).sort_values(by = 'Times',ascending = False)
losers = data.groupby('Loser')['Times'].sum().reset_index(drop = False).sort_values(by = 'Times',ascending = False)

winners = data.groupby('Winner')['Times'].sum().reset_index(drop = False).sort_values(by = 'Times',ascending = False)

losers = losers.rename(columns = {'Loser' : 'Team'})

winners = winners.rename(columns = {'Winner' : 'Team'})
most_finals = losers.merge(winners, on = 'Team', how = 'left')
most_finals['Times_y'].fillna(0, inplace = True)
most_finals['most appeareances'] = (most_finals['Times_x'] + most_finals['Times_y']).astype(int)
most_finals.drop(['Times_x', 'Times_y'], axis = 1, inplace = True)
most_finals.sort_values(by = 'most appeareances', ascending = False)
plt.figure(figsize=(40,20))

#subgrade_order = sorted(df['sub_grade'].unique())

sns.countplot(x='most appeareances',data=most_finals.sort_values(by = 'most appeareances', ascending = False),palette='coolwarm')