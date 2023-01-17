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
data = pd.read_csv('/kaggle/input/nba-all-star-game-20002016/NBA All Stars 2000-2016 - Sheet1.csv')

#games = pd.read_excel('/kaggle/input/nba-all-star-game-20002016/NBA All Star Games (1).xlsx')
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
data
top_10 =pd.DataFrame(data.Player.value_counts()[:10]).reset_index(drop = False).rename(columns = {'Player' : 'Times', 'index' : 'Player'})

top_10
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Player',data=data, order = data.Player.value_counts().iloc[:20].index)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize = 30)

#ax.set_yticklabels(ax.get_yticklabels(), ha = 'center', fontsize = 15)

plt.tight_layout()
data.Pos.value_counts()
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Pos',data=data, order = data.Pos.value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), fontsize = 30)

plt.tight_layout()
kobe = data[data.Player == 'Kobe Bryant']

kobe
height = data.HT.unique().tolist()

height
print('The maximum height of an all star player in NBA is' ,max(height))

print('The min height of an all star player in NBA is' ,min(height))
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='HT',data=data, order = data.HT.value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), fontsize = 30)

plt.tight_layout()
guard = data[data.Pos == 'G']

guard
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Player',data=guard, order = guard.Player.value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)

plt.tight_layout()
data.Pos.unique()
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Player',data=data[data.Pos == 'SG'], order = data['Player'][data.Pos == 'SG'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)

plt.tight_layout()
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Player',data=data[data.Pos == 'SF'], order = data['Player'][data.Pos == 'SF'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)

plt.tight_layout()
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Player',data=data[data.Pos == 'F'], order = data['Player'][data.Pos == 'F'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)

plt.tight_layout()
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Player',data=data[data.Pos == 'PF'], order = data['Player'][data.Pos == 'PF'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)

plt.tight_layout()
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Player',data=data[data.Pos == 'C'], order = data['Player'][data.Pos == 'C'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)

plt.tight_layout()
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Player',data=data[data.Pos == 'GF'], order = data['Player'][data.Pos == 'GF'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)

plt.tight_layout()
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Player',data=data[data.Pos == 'PG'], order = data['Player'][data.Pos == 'PG'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)

plt.tight_layout()
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Player',data=data[data.Pos == 'FC'], order = data['Player'][data.Pos == 'FC'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)

plt.tight_layout()
data
team = pd.DataFrame(data.Team.value_counts()).reset_index().rename(columns = {'Team' : 'Times', 'index' : 'Team'})

team
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Team',data=data, order = data.Team.value_counts().iloc[:20].index)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize = 30)

#ax.set_yticklabels(ax.get_yticklabels(), ha = 'center', fontsize = 15)

plt.tight_layout()
data.WT.unique()
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='WT',data=data, order = data.WT.value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), fontsize = 30)

plt.tight_layout()
weight_200 = data[data.WT > 200]

weight_200
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Player',data=weight_200, order = weight_200.Player.value_counts().iloc[:20].index)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)

plt.tight_layout()
weight_not_200 = data[data.WT < 200]
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Player',data=weight_not_200, order = weight_not_200.Player.value_counts().iloc[:20].index)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)

plt.tight_layout()
data['NBA Draft Status'].value_counts()
data['Player'][data['NBA Draft Status'] == '1996 Rnd 1 Pick 13'].loc[38]
foren = data[data.Nationality != 'United States']

foren
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Player',data=foren, order = foren.Player.value_counts().iloc[:20].index)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)

plt.tight_layout()
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Nationality',data=foren, order = foren.Nationality.value_counts().iloc[:20].index)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)

plt.tight_layout()