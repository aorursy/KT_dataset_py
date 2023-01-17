# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('tab10')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Load data.

metras = pd.read_csv("/kaggle/input/marble-racing/marbles.csv")
print(metras.head())
## Add column with pole position to numeric.

metras['p_position'] = metras['pole'].str.lstrip('P')
metras['p_position'] = pd.to_numeric(metras['p_position'])

print(metras.head())
## Subset dataset.

metras_sum = metras.groupby('marble_name').sum().reset_index()
team_sum = metras.groupby('team_name').sum().reset_index()
## Pole position performance by marble.

plt.figure(figsize=(32, 6))
plt.bar(metras_sum.marble_name, metras_sum.p_position)
plt.title('Pole position performance by marble', fontsize=20)
plt.ylabel('Lower number = Better performance', fontsize=16);
## Points performance by marble.

plt.figure(figsize=(32, 6))
plt.bar(metras_sum.marble_name, metras_sum.points)
plt.title('Points performance by marble', fontsize=20)
plt.ylabel('Higher number = Better performance', fontsize=16);
## Pole position performance by team.

plt.figure(figsize=(32, 6))
plt.bar(team_sum.team_name, team_sum.p_position)
plt.title('Pole position performance by team', fontsize=20)
plt.ylabel('Lower number = Better performance', fontsize=16);
## Points performance by team.

plt.figure(figsize=(32, 6))
plt.bar(team_sum.team_name, team_sum.points)
plt.title('Points performance by team', fontsize=20)
plt.ylabel('Higher number = Better performance', fontsize=16);
## Pole position comparison by marble.

plt.figure(figsize=(32, 7))
ax = sns.boxplot(data=metras, x='marble_name', y='p_position')
ax.axhline(y=10,  ls=':', color='r',linewidth=4)
plt.title('Pole position Performance by marble', fontsize=20)
plt.xlabel(None)
plt.ylabel('Pole position Performance', fontsize=16);
## Pole position comparison by team.

plt.figure(figsize=(32, 7))
ax = sns.boxplot(data=metras, x='team_name', y='p_position')
ax.axhline(y=10,  ls=':', color='r',linewidth=4)
plt.title('Pole position Performance by team', fontsize=20)
plt.xlabel(None)
plt.ylabel('Pole position Performance', fontsize=16);
## Points comparison by Marbles.

plt.figure(figsize=(32, 7))
ax = sns.boxplot(data=metras, x='marble_name', y='points')
ax.axhline(np.mean(metras.points),  ls='--', color='r',linewidth=2)
plt.title('Points performance by marble', fontsize=20)
plt.xlabel(None)
plt.ylabel('Points', fontsize=16);
## Points comparison by teams

plt.figure(figsize=(32, 7))
ax = sns.boxplot(data=metras, x='team_name', y='points')
ax.axhline(np.mean(metras.points), ls=':', color='r',linewidth=4 )
plt.title('Points performance by team', fontsize=20)
plt.xlabel(None)
plt.ylabel("Points", fontsize=16);
## Average time lap by marble.

plt.figure(figsize=(32, 7))
ax = sns.boxplot(data=metras, x='marble_name', y='avg_time_lap')
ax.axhline(np.mean(metras.avg_time_lap), ls='--', color='r',linewidth=3)
plt.title('Average time lap by marble', fontsize=20)
plt.xlabel(None)
plt.ylabel('Average time lap', fontsize=16);
## Average time lap by team

plt.figure(figsize=(32, 7))
ax = sns.boxplot(data=metras, x='team_name', y='avg_time_lap')
plt.title('Average time lap by team', fontsize=20)
plt.xlabel(None)
plt.ylabel("Average time lap", fontsize=16);
## Setting xticks and yticks for the following plots.

f_pos = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
y_pos = range(len(f_pos))
r_place_odd = ['Savage Speedway',  'Momotorway', 'Greenstone',  'Razzway']
r_pos_odd = range(len(r_place_odd))
r_place_even = ["O'raceway", 'Hivedrive','Short Circuit','Midnight Bay']
r_pos_even = range(len(r_place_even))
r_names = ['Savage Speedway', "O'raceway", 'Momotorway', 'Hivedrive',
       'Greenstone', 'Short Circuit', 'Razzway', 'Midnight Bay']
r_pos_names = range(len(r_names));
## Team's Pole position by race.

sns.relplot(x= 'race', y='p_position', data = metras, kind='line', 
            hue='team_name', height=10, aspect=2.7, marker='o', linewidth=5, markersize=14)
plt.title('Pole postion comparison', fontsize=22)
plt.xlabel('Race' , fontsize=16)
plt.xticks(r_pos_names, r_names, fontsize=16)
plt.ylabel('Pole position', fontsize=16)
plt.show();
## Team wins by race.

sns.relplot(x= 'race', y='points', data = metras, kind='line', 
            hue='team_name', height=10, aspect=2.7, marker='o', linewidth=5, markersize=14)
plt.yticks(y_pos, f_pos,fontsize=16)
plt.title('Wins comparison', fontsize=22)
plt.xlabel('Race' , fontsize=16)
plt.xticks(r_pos_names, r_names, fontsize=16)
plt.ylabel('Final position by points', fontsize=16)
plt.show();
## Marbel with the most pole position.

p_position = (metras.pole == 'P1')
pole_p = metras[p_position]

f, ax = plt.subplots(figsize=(15, 5))
ax = sns.countplot(x = 'marble_name', data = pole_p)
ax.set_yticks([0,1,2,3])
ax.set_yticklabels([0,1,2,3], fontsize = 14)
ax.set_xticklabels(pole_p.marble_name, fontsize = 14)
ax.set_xlabel(None)
ax.set_ylabel("Pole Position", fontsize = 14)
ax.set_title('Marble with the most Pole Position', fontsize = 16)
plt.show()
## Team with the most pole position in the Marbula 1.

t_position = (metras.pole == 'P1')
tp_p = metras[t_position]

f, ax = plt.subplots(figsize=(15, 5))
ax = sns.countplot(x = 'team_name', data = tp_p)
ax.set_yticks([0,1,2,3])
ax.set_yticklabels([0,1,2,3], fontsize = 14)
ax.set_xticklabels(tp_p.team_name, fontsize = 14)
ax.set_xlabel(None)
ax.set_ylabel("Pole Position", fontsize = 14)
ax.set_title('Team with the most Pole Position', fontsize = 16)
plt.show()
## Marbel with the most wins position in the Marbula 1.

mf_position = (metras.points >= 25)
mf_p = metras[mf_position]

f, ax = plt.subplots(figsize=(15, 5))
ax = sns.countplot(x = 'marble_name', data = mf_p)
ax.set_yticks([0,1,2,3])
ax.set_yticklabels([0,1,2,3], fontsize = 14)
ax.set_xticklabels(mf_p.marble_name, fontsize = 14)
ax.set_xlabel(None)
ax.set_ylabel("Wins", fontsize = 14)
ax.set_title('Marble with the most Wins', fontsize = 16)
plt.show()
## Team with the most wins in the Marbula 1

f_position = (metras.points >= 25)
final_p = metras[f_position]

f, ax = plt.subplots(figsize=(15, 5))
ax = sns.countplot(x = 'team_name', data = final_p)
ax.set_yticks([0,1,2,3])
ax.set_yticklabels([0,1,2,3], fontsize = 14)
ax.set_xticklabels(final_p.team_name, fontsize = 14)
ax.set_xlabel(None)
ax.set_ylabel("Wins", fontsize = 14)
ax.set_title('Team with the most Wins', fontsize = 16)
plt.show()

