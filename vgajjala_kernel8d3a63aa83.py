import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")
df = pd.read_csv('../input/vgsales.csv')
df.info()
df = df[df.Year <= 2018.0]
df.info()
df.isnull().sum()
df = df.dropna()
df.info()
df.isnull().sum()
y = df.groupby(['Year']).sum()

y = y['Global_Sales']

x = y.index.astype(int)



plt.figure(figsize = (10,5))

ax = sns.barplot(x = x, y = y)

ax.set_title("Games Sales in Millions ($)", fontsize = 25)

ax.set_xlabel("Year", fontsize=10)

ax.set_ylabel("Millions ($)", fontsize = 10)

ax.set_xticklabels(labels = x, fontsize=10, rotation=90)

plt.show()
y = df.groupby(['Year']).sum()

x = y.index.astype(int)

NA = y['NA_Sales']

EU = y['EU_Sales']

JP = y['JP_Sales']

O = y['Other_Sales']

fig = plt.figure(figsize = (15,10))

ax1 = fig.add_subplot(141)

sns.barplot(x = x, y = NA)

ax1.set_title("North America Game Sales in Millions ($)")

ax1.set_xlabel("Year", fontsize=10)

ax1.set_ylabel("Millions ($)", fontsize = 10)

ax1.set_xticklabels(labels = x, fontsize=6, rotation=90)

ax2 = fig.add_subplot(142)

sns.barplot(x = x, y = EU)

ax2.set_title("Europe Game Sales in Millions ($)")

ax2.set_xlabel("Year", fontsize=10)

ax2.set_ylabel("Millions ($)", fontsize = 10)

ax2.set_xticklabels(labels = x, fontsize=6, rotation=90)

ax3 = fig.add_subplot(143)

sns.barplot(x = x, y = JP)

ax3.set_title("Japan Game Sales in Millions ($)")

ax3.set_xlabel("Year", fontsize=10)

ax3.set_ylabel("Millions ($)", fontsize = 10)

ax3.set_xticklabels(labels = x, fontsize=6, rotation=90)

ax4 = fig.add_subplot(144)

sns.barplot(x = x, y = O)

ax4.set_title("Other Game Sales in Millions ($)")

ax4.set_xlabel("Year", fontsize=10)

ax4.set_ylabel("Millions ($)", fontsize = 10)

ax4.set_xticklabels(labels = x, fontsize=6, rotation=90)

plt.show()
table1 = df.pivot_table('Global_Sales', index='Genre', columns='Year', aggfunc='sum')

Genre = table1.idxmax()

sales1 = table1.max()

years1 = table1.columns.astype(int)

data1 = pd.concat([Genre, sales1], axis=1)

data1.columns = ['Genre', 'Global Sales']

table2 = df.pivot_table('Global_Sales', index='Publisher', columns='Year', aggfunc='sum')

Publisher = table2.idxmax()

sales2 = table2.max()

years2 = table2.columns.astype(int)

data2 = pd.concat([Publisher, sales2], axis=1)

data2.columns = ['Publisher', 'Global Sales']

fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(121)

sns.pointplot(y = 'Global Sales', x = years1, hue='Genre', data=data1, size=15)

ax1.set_xlabel(xlabel='Year', fontsize=12)

ax1.set_ylabel(ylabel='Global Sales Per Year', fontsize=12)

ax1.set_title(label='Highest Genre Revenue in Millions($) Per Year', fontsize=20)

ax1.set_xticklabels(labels = years1, fontsize=12, rotation=90)

ax2 = fig.add_subplot(122)

sns.pointplot(y = 'Global Sales', x = years2, hue='Publisher', data=data2, size=15)

ax2.set_xlabel(xlabel='Year', fontsize=12)

ax2.set_ylabel(ylabel='Global Sales Per Year', fontsize=12)

ax2.set_title(label='Highest Publisher Revenue in Millions($) Per Year', fontsize=20)

ax2.set_xticklabels(labels = years2, fontsize=12, rotation=90)

plt.show();
data = df.groupby(['Platform']).count().iloc[:,0]

data = pd.DataFrame(data.sort_values(ascending=False))[0:5]

Platform = data.index

data.columns = ['Releases']

colors = sns.color_palette("BuGn_r", len(data))

plt.figure(figsize=(12,8))

ax = sns.barplot(y = Platform , x = 'Releases', data=data, orient='h', palette=colors)

ax.set_xlabel(xlabel='Number of Releases', fontsize=16)

ax.set_ylabel(ylabel='Platform', fontsize=16)

ax.set_title(label='Top 5 Total Platform Games Released', fontsize=20)

ax.set_yticklabels(labels = Platform, fontsize=14)

plt.show();
top5 = ['DS', 'PS2', 'PS3', 'Wii', 'X360']

table = df.pivot_table('Global_Sales', columns='Platform', index='Year', aggfunc='sum')

data = [table[i] for i in top5]

data = np.array(data)

data = pd.DataFrame(np.reshape(data, (5, 38)))

years = table.index.astype(int)

plt.figure(figsize=(12,8))

ax = sns.heatmap(data)

ax.set_xticklabels(labels = years, fontsize=12, rotation=90)

ax.set_yticklabels(labels = top5[::-1], fontsize=14, rotation=0)

ax.set_xlabel(xlabel='Year', fontsize=18)

ax.set_ylabel(ylabel='Platform', fontsize=18)

ax.set_title(label='Total Revenue Per Year in Millions($) for Top 5 Platforms', fontsize=20)

plt.show();
sns.distplot(df['Year'],

                 kde=True,

                 rug = False,

                 bins = 50

                 )
data = df[df['Publisher'] == 'Nintendo']

sns.jointplot("Year", "Global_Sales",data=data, color='red')
ax = sns.regplot(x='Year', y='Global_Sales', data=data, color='orange')

ax.set_title('Nintendo Sales', fontsize = 20)
data = df.drop(['Rank', 'Other_Sales', 'Global_Sales'], axis=1)

sns.pairplot(data,hue='Genre', palette="husl")
plt.figure(figsize=(15,10))

ax = sns.violinplot("Year", "Global_Sales", data=df )

ax.set_xlabel("Year", fontsize=10)

ax.set_ylabel("Millions ($)", fontsize = 15)

ax.set_xticklabels(labels = x, fontsize=15, rotation=90)