# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set(style="darkgrid")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/fifa19/data.csv')
df.head()
# Let's use the value_counts() to determine the frequency of the values present in club
df['Club'].value_counts()
df_c=df.groupby(['Club']).count()
df_c.head()
# Let's check the club wise count
df_c=df.head(100)
plt.figure(figsize=(7,7))
sns.set(style="darkgrid")
ax = sns.countplot(y="Club", data=df.head(100))
# Let's see country wise count of players
df_p=df.head(100)
plt.figure(figsize=(7,7))
sns.set(style="darkgrid")
ax = sns.countplot(y="Nationality", data=df_p.head(200))
# how about age
df_p=df.head(100)
plt.figure(figsize=(7,7))
sns.set(style="darkgrid")
ax = sns.countplot(y="Age", data=df_p.head(200))
# Let's check with bar plot of Players and their potential
sns.barplot(x="Potential", y="Name", data=df[:15])
# Age Vs Aggression
sns.barplot(x="Age", y="Aggression", data=df[:25])
# Let's check the acceleration
sns.distplot(df_c['Acceleration'].dropna(),bins=100,color='purple')
# Age Vs Jumping
sns.barplot(x="Age", y="Jumping", data=df[:25])
# Age Vs strength
sns.barplot(x="Age", y="Strength", data=df[:25])
# Similarly let's check Age Vs Stamina
sns.barplot(x="Age", y="Stamina", data=df[:25])
plt.scatter(df_p['Age'].head(10),df_p['Nationality'].head(10),color='R',label='TIME TABLE',marker='*',s=80)
# Calculating Data Automatically

x_values = df_p['Nationality']
y_values = df_p['Club']

plt.scatter(x_values, y_values, s=40)

# set the range for each axis
plt.axis([0, 5, 0, 10])

plt.show()
# Overall distplot
sns.distplot(df['Overall'])
# scatter plot
sns.relplot(x="Name", y="Potential", data=df[:200], kind="scatter");
# Top 20 players with value
sns.relplot(x="Value", y="Name", hue="Value",data=df[:20]);
# bubble plot for top players based on value
sns.relplot(x="Value", y="Name", data=df[:30], kind="scatter", size="Value", hue="Value");
# bubble plot for top players based on Stamina
sns.relplot(x="Stamina", y="Name", data=df[:30], kind="scatter", size="Stamina", hue="Stamina");
# bubble plot for top players based on Stamina
sns.relplot(x="Strength", y="Name", data=df[:30], kind="scatter", size="Strength", hue="Strength");
# Categoriacal plot basd on potential / Club
sns.catplot(x="Potential", y="Club", kind='strip',data=df[:100]);
# Categoriacal plot based on Nationality and Overall
sns.catplot(x="Overall", y="Nationality", kind='strip',data=df[:100]);
sns.catplot(x="Vision", y="Club", kind='strip',data=df[:100]);
# Top 20 players with ball control
sns.catplot(x="BallControl", y="Name", kind='strip',data=df[:20]);
# Top 50 players with Penality accuracy
sns.catplot(x="Penalties", y="Name", kind='strip',data=df[:50]);
# Violin plots
sns.catplot(x="Finishing", y="BallControl",kind="violin",data=df);
sns.catplot(x="Stamina", y="Aggression",kind="point",data=df);
sns.catplot(x="Stamina", y="Strength",kind="bar",data=df);
# distribution
plt.figure(figsize=(10,10))
sns.kdeplot(df['Balance'], shade=True);
plt.figure(figsize=(10,10))
sns.distplot(df['Strength']);
plt.figure(figsize=(10,10))
sns.distplot(df['Interceptions']);
data = df[['Age', 'Stamina']]

# generate subplots
fig, ax = plt.subplots()

# add labels to x axis
plt.xticks(ticks=[1,2], labels=['Age', 'Stamina'])

# make the violinplot
plt.violinplot(data.values);
# set label of axes 
plt.xlabel('Strength')
plt.ylabel('Stamina')

# set title
plt.title('Strength vs Stamina')

# plot
plt.scatter(df["Strength"][:100], df["Stamina"][:100], s=df["Stamina"][:100]*50, c='red')
# Top 15 clubs and their players age
sns.barplot(x="Club", y="Age", data=df[:15])
# line plot using relplot
sns.lineplot(x="Club", y="Aggression",data=df[:30]);
# Clubs and their Aggression
sns.barplot(x="Aggression", y="Club", data=df[:20])
# Clubs and their BallControl
sns.barplot(x="BallControl", y="Club", data=df[:12])
# Clubs and their Strength
sns.barplot(x="Strength", y="Club", data=df[:12])
df1 = df[['Club', 'Potential','Stamina','Strength']]
sns.pairplot(df1, hue='Stamina', height=2.5);
df.columns
# To be continued.....