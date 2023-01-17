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
df = pd.read_csv('/kaggle/input/epl-results-19932018/EPL_Set.csv')

df.head()
df.dropna(inplace=True)

df.drop('Div',axis=1, inplace=True)

df.reset_index(inplace=True, drop=True)

df.head()
df.HTAG = df.HTAG.apply(int)

df.HTHG = df.HTHG.apply(int)

df.Date = pd.to_datetime(df.Date, dayfirst=True)
df['dayofweek'] = df.Date.apply(lambda x: x.dayofweek)

df.head()
df.dayofweek.value_counts() # Most of the matches have been played on a Saturday
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize=(12,6))

sns.countplot('dayofweek',data=df,hue='FTR')



# For some reason, teams tend to love the prospects of playing at Home on a Saturday

# Maybe because they're relatively fresh, and so are the majority Home supporters.
x = df[(df.HTHG) - (df.HTAG) >= 3].FTR.value_counts()

print(x)

print(x / x.sum())
x = df[(df.HTAG) - (df.HTHG) >= 3].FTR.value_counts()

print(x)

print(x / x.sum())
x = df[(df.HTHG) - (df.HTAG) == 2].FTR.value_counts()

print(x)

print(x / x.sum())
x = df[(df.HTAG) - (df.HTHG) == 2].FTR.value_counts()

print(x)

print(x / x.sum())
x = df[(df.HTHG) - (df.HTAG) == 1].FTR.value_counts()

print(x)

print(x / x.sum())
x = df[(df.HTAG) - (df.HTHG) == 1].FTR.value_counts()

print(x)

print(x / x.sum())
x = df[(df.HTHG) - (df.HTAG) == 0].FTR.value_counts()

print(x)

print(x / x.sum())
ars_home = df[(df.HomeTeam == 'Arsenal')]

ars_away = df[(df.AwayTeam == 'Arsenal')]

plt.figure(figsize=(12,6))

plt.subplot(211)

sns.countplot('dayofweek',data=ars_home,hue='FTR', hue_order=['H','D','A'])

plt.title('Arsenal Home Results')

plt.figure(figsize=(12,6))

plt.subplot(212)

sns.countplot('dayofweek',data=ars_away,hue='FTR', hue_order=['H','D','A'])

plt.title('Arsenal Away Results')

plt.show()
liv_home = df[(df.HomeTeam == 'Liverpool')]

liv_away = df[(df.AwayTeam == 'Liverpool')]



plt.figure(figsize=(12,6))

plt.subplot(2,1,1)

sns.countplot('dayofweek',data=liv_home,hue='FTR', hue_order=['H','D','A'])

plt.title('Liverpool Home Results')

plt.show()

plt.figure(figsize=(12,6))

plt.subplot(2,1,2)

sns.countplot('dayofweek',data=liv_away,hue='FTR', hue_order=['H','D','A'])

plt.title('Liverpool Away Results')

plt.show()
mutd_home = df[(df.HomeTeam == 'Man United')]

mutd_away = df[(df.AwayTeam == 'Man United')]



plt.figure(figsize=(12,6))

plt.subplot(2,1,1)

sns.countplot('dayofweek',data=mutd_home,hue='FTR', hue_order=['H','D','A'])

plt.title('Man United Home Results')

plt.show()

plt.figure(figsize=(12,6))

plt.subplot(2,1,2)

sns.countplot('dayofweek',data=mutd_away,hue='FTR', hue_order=['H','D','A'])

plt.title('Man United Away Results')

plt.show()
#ARSENAL: There's nearly a 90% chance of Arsenal winning a game in which they've led at HT.



x = ars_home[(ars_home.HTR == 'H')].FTR.value_counts()

print(x)

print(x * 100 / x.sum())
#LIVERPOOL: There's a tiny 0.5% (zero-point-five!) chance of Liverpool losing a game in which they've led at HT! WOW!



x = liv_home[(liv_home.HTR == 'H')].FTR.value_counts()

print(x)

print(x * 100/ x.sum())
#MAN UNITED: Thev've NEVER lost a game in which they've led at HT! DAMNNNNN!



x = mutd_home[(mutd_home.HTR == 'H')].FTR.value_counts()

print(x)

print(x * 100 / x.sum())
# ARSENAL: All results are possible when Arsenal are losing at home at HT



x = ars_home[(ars_home.HTR == 'A')].FTR.value_counts()

print(x)

print(x * 100 / x.sum())
# LIVERPOOL: There's more than 50% of a chance of the away team winning, when they've led at HT at Anfield



x = liv_home[(liv_home.HTR == 'A')].FTR.value_counts()

print(x)

print(x * 100 / x.sum())
# MAN UTD: Surprisingly, United's record at Old Trafford ain't as good when they've been losing at HT



x = mutd_home[(mutd_home.HTR == 'A')].FTR.value_counts()

print(x)

print(x * 100 / x.sum())
#ARSENAL: When Arsenal are losing at HT away from home, there's only a 12% chance they'll win the game.



x = ars_away[(ars_away.HTR == 'H')].FTR.value_counts()

print(x)

print(x * 100 / x.sum())
#LIVERPOOL : When Liverpool are losing at HT away from home, there's only a 8% chance they'll win the game



x = liv_away[(liv_away.HTR == 'H')].FTR.value_counts()

print(x)

print(x * 100/ x.sum())
# MAN UTD: When Man Utd are losing at HT away from home, there's a 20% chance they'll come back and the game! WHOAAA!

x = mutd_away[(mutd_away.HTR == 'H')].FTR.value_counts()

print(x)

print(x * 100/ x.sum())
#ARSENAL: Whenever Arsenal are leading at HT away from home, there's an 80% chance they'll eventually win the game



x = ars_away[(ars_away.HTR == 'A')].FTR.value_counts()

print(x)

print(x *100/ x.sum())
#LIVERPOOL: Whenever Liverpool are leading away from home at HT, there's a 77% chance they'll end up winning



x = liv_away[(liv_away.HTR == 'A')].FTR.value_counts()

print(x)

print(x *100/ x.sum())
#MAN UTD: Whenever the Red Devils are leading at HT away from home, there's only a 2.5% chance the opponent will stage a coma=eback and win it!



x = mutd_away[(mutd_away.HTR == 'A')].FTR.value_counts()

print(x)

print(x *100/ x.sum())