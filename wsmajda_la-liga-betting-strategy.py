# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter




# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

plt.rcParams["figure.figsize"] = (20,10)
df = pd.read_csv('/kaggle/input/laliga_full.csv')
df
def bet_equal(row):
    return 1

def bet_martingale_home(row):
    return 2/row['B365H']

df['bet_value'] = df.apply (lambda row: bet_equal(row), axis=1)

def bet_on_home (row):
    if row['result'] == 'H' :
        return row['bet_value'] * (row['B365H'])
    return 0

df['net'] = df.apply (lambda row: bet_on_home(row), axis=1)
print("bet = " + str(df['bet_value'].sum()))
print("net = " + str(df['net'].sum()))
print("bet/net =" + str(df['net'].sum() / df['bet_value'].sum()))

df.head()
def real_result(series):
    homes = 0
    non_homes = 0
    for k, value in series.items():
        if 'H' in value:
            homes = homes + 1
        else:
            non_homes = non_homes + 1        
    return  (homes / (homes + non_homes))
def readible_bin(row):
    return "[%.2f, %.2f] -> %s - %s" % (1/row['home_bin'].right, 1/row['home_bin'].left, row['home_bin'].left, row['home_bin'].right)
 
df['home_propability'] = [1/n for n in df['B365H']]
df['draw_propability'] = [1/n for n in df['B365D']]
df['away_propability'] = [1/n for n in df['B365A']]

df['home_bin'] = pd.qcut(df['B365H'], 8)
df['away_bin'] = pd.qcut(df['B365A'], 8)

fig, ax = plt.subplots(figsize=(15,7))

grouped = df.groupby(['season','home_bin'])

# new series with formatted bins
df['home_bin_str'] = df.apply (readible_bin, axis=1)

# main graph
df.groupby(['season','home_bin_str'])['result'].aggregate(real_result).unstack().plot(ax=ax, kind="line")

# upper probability values for bin
for cat in df['away_bin'].apply(lambda row: 1/row.left).unique():
    ax.hlines(y=cat, xmin=0.0, xmax=7.0, color='gray', linestyle="--")

ax.axhline(y=1, color='r', linestyle='-')
ax.legend()

df.head()

#totals 
def total(df, *args): 
    total_dir = {}

    for col_name in args: 
        col = df[col_name]
        
        for entry in col:
            if entry in total_dir.keys():
                total_dir[entry] += 1
            else:
                total_dir[entry] = 1
        return total_dir

total_h_win = total(df,"result")
plt.bar(*zip(*total_h_win.items()))
plt.show()
df2 = df[df['result'] == 'H']
total_h_win_club = total(df2,"home_team")

# c = Counter(total_h_win_club)
# # print(c.most_common())
# # sort_values
# x = sorted(total_h_win_club.values())
# print(x)

df3 = df[df['result'] == 'A']
total_a_win_club = total(df3,"away_teame")
d = Counter(total_a_win_club)
print(d.most_common())
x = total_h_win_club.update(total_a_win_club)
print(x)
#todo
df5 = df[(df['home_team'] == 'BAR') | (df['away_teame'] == 'BAR')]
y = df5.groupby(['season','home_bin_str'])['result'].aggregate(real_result).unstack()
df5.head()
y = df5.groupby(['season','home_bin_str'])['result']
y.head()