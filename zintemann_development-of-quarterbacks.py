# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
qbs = pd.read_csv('../input/Game_Logs_Quarterback.csv')
qbs = qbs[qbs['Season']=='Regular Season']
qbs = qbs[qbs['Passes Attempted'] != '--']
qbs.tail()
qbs['Team Points'] = qbs['Score'].str.split(" ").str.get(0)

qbs['Team Points'] = qbs['Team Points'].astype('int')

qbs.head()

qbs.info()
qbs['Passing Yards'] = qbs['Passing Yards'].astype('int')

qbs['TD Passes'] = qbs['TD Passes'].astype('int')

qbs['Ints'] = qbs['Ints'].astype('int')

qbs['Passes Completed'] = qbs['Passes Completed'].astype('int')

qbs['Passes Attempted'] = qbs['Passes Attempted'].astype('int')
qbs = qbs[qbs['Passes Attempted'] >= 10]
qbs['Completion Percentage'] = (qbs['Passes Completed'] / qbs['Passes Attempted']) * 100

qbs['Points by TD Pass'] = qbs['TD Passes'] * 7
qbseasons = qbs.groupby(['Name', 'Player Id', 'Year']).agg({'Ints' : 'sum', 'Passer Rating' : 'mean', 

                                                              'TD Passes' : 'sum', 'Passes Completed' : 'sum',

                                                             'Passing Yards' : 'sum',

                                                           'Team Points' : 'mean',

                                                           'Passes Attempted' : 'sum',

                                                           'Completion Percentage' : 'mean',

                                                           'Points by TD Pass' : 'mean'})
qbseasons.head()
seasonaverages = qbseasons.groupby('Year').agg({'Passer Rating' : 'mean', 'Passes Completed': 'mean',

                                               'TD Passes' : 'mean', 'Ints' : 'mean', 'Passing Yards' : 'mean',

                                               'Team Points' : 'mean',

                                               'Passes Attempted' : 'mean',

                                               'Completion Percentage' : 'mean',

                                               'Points by TD Pass' : 'mean'})
seasonaverages
plt.style.use('seaborn')

fig = plt.figure()

fig = seasonaverages[['Passer Rating', 'Completion Percentage']].plot(cmap='coolwarm')

fig.set_title('Average Game QB Passer Rating and Pass Completion over the Years')

plt.savefig('Quaterbacks 1.png')
qbs2 = pd.read_csv('../input/Game_Logs_Quarterback.csv')
qbs2 = qbs[qbs['Season']=='Regular Season']
qbs2['Team Points'] = qbs2['Score'].str.split(" ").str.get(0)

qbs2['Team Points'] = qbs2['Team Points'].astype('int')
qbs2['TD Passes'] = qbs2['TD Passes'].astype('int')
scores = qbs2.groupby(['Year', 'Week', 'Opponent', 'Game Date']).agg({'Team Points' : 'sum', 'TD Passes' : 'sum'})

scores.head()
scores.info()
season_scores = scores.groupby('Year').agg({'Team Points' : 'mean', 'TD Passes' : 'mean'})
season_scores['Points by TD Pass'] = season_scores['TD Passes'] * 7
plt.style.use('seaborn')

fig2 = plt.figure()

fig2 = season_scores[['Team Points', 'Points by TD Pass']].plot(cmap='viridis')

fig2.set_title('Average Team Points and Points by TD Pass per Game')

plt.savefig('Quaterbacks 2.png')
season_scores['Percentage of Points by TD Pass'] = (season_scores['Points by TD Pass'] / season_scores['Team Points']) * 100
fig3 = plt.figure()

fig3 = season_scores['Percentage of Points by TD Pass'].plot(c='brown')

fig3.set_title('Percentage of Points by Touchdown Pass')

plt.savefig('Quaterbacks 3.png')