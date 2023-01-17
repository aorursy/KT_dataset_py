import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
df = pd.read_csv('../input/videogamesales/vgsales.csv')
df.info()
df.describe()
df.head()
sns.heatmap(df.isnull(),cmap='Blues',cbar=False,yticklabels=False)
df = df.dropna(how='any')
print("Number of games: ", len(df))
print("Number of publishers: ", len(df['Publisher'].unique()))
print("Number of platforms: ", len(df['Platform'].unique()))
print("Number of genres: ", len(df['Genre'].unique()))
plt.figure(figsize=(8,5))
plt.title('Sales distribution as per year')
ax = sns.distplot(df['Year'], color = 'g')
plt.figure(figsize=(12, 8))
ax = sns.countplot(x="Genre", data=df, order = df['Genre'].value_counts().index, palette="rocket")
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext= (0, 10), textcoords = 'offset points')
ax.set_title(label='Number of Games made as per Genre type', fontsize=20)
ax.set_xlabel(xlabel='Genre', fontsize=16)
ax.set_ylabel(ylabel='Number of Games', fontsize=16)
plt.xticks(rotation=50)
y = df.groupby(['Year']).sum()
y = y['Global_Sales']
x = y.index.astype(int)

plt.figure(figsize=(14,8))
ax = sns.barplot(y = y, x = x,facecolor=(0, 0, 0, 0),
                   linewidth=3,
                   edgecolor=sns.color_palette("dark", 10))
ax.set_xlabel(xlabel='Year', fontsize=16)
ax.set_xticklabels(labels = x, fontsize=12, rotation=50)
ax.set_ylabel(ylabel='$ Millions', fontsize=16)
ax.set_title(label='Game Sales in $ Millions Per Year', fontsize=20)
plt.show();
plt.figure(figsize=(14, 8))
plt.style.use('ggplot')
ax = sns.countplot(x="Year", data=df, order = df.groupby(by=['Year'])['Name'].count().sort_values(ascending=False).index,facecolor=(0, 0, 0, 0),
                   linewidth=3,
                   edgecolor=sns.color_palette("dark", 10))
ax.set_xlabel(xlabel='Year', fontsize=16)
ax.set_ylabel(ylabel='Number of Games relased', fontsize=16)
ax.set_title(label='Number of Games released Per Year', fontsize=20)
plt.xticks(rotation=50)
df_to_pie = df.drop(['Name','Platform', 'Genre', 'Publisher','Global_Sales','Rank'], axis = 1).groupby('Year').agg('sum')
df_to_pie.head()
index = ['NA_Sales','EU_Sales','JP_Sales','Other_Sales']
series = pd.DataFrame({'2006': df_to_pie.loc[[2006],:].values.tolist()[0],
                      '2007': df_to_pie.loc[[2007],:].values.tolist()[0],
                      '2008': df_to_pie.loc[[2008],:].values.tolist()[0],
                      '2009': df_to_pie.loc[[2009],:].values.tolist()[0],
                       '2010': df_to_pie.loc[[2010],:].values.tolist()[0],
                       '2011': df_to_pie.loc[[2011],:].values.tolist()[0]}, index=index)
series.plot.pie(y='2006',figsize=(9, 9), autopct='%1.1f%%', colors=['orange', 'palegreen', 'aqua', 'blue'], fontsize=18, legend=False, title='2006 Sales Distribution as per regions').set_ylabel('')
series.plot.pie(y='2007',figsize=(9, 9), autopct='%1.1f%%', colors=['orange', 'palegreen', 'aqua', 'blue'], fontsize=18, legend=False, title='2007 Sales Distribution as per regions').set_ylabel('')
series.plot.pie(y='2008',figsize=(9, 9), autopct='%1.1f%%', colors=['orange', 'palegreen', 'aqua', 'blue'], fontsize=18, legend=False, title='2008 Sales Distribution as per regions').set_ylabel('')
series.plot.pie(y='2009',figsize=(9, 9), autopct='%1.1f%%', colors=['orange', 'palegreen', 'aqua', 'blue'], fontsize=18, legend=False, title='2009 Sales Distribution as per regions').set_ylabel('')
series.plot.pie(y='2010',figsize=(9, 9), autopct='%1.1f%%', colors=['orange', 'palegreen', 'aqua', 'blue'], fontsize=18, legend=False, title='2010 Sales Distribution as per regions').set_ylabel('')
series.plot.pie(y='2011',figsize=(9, 9), autopct='%1.1f%%', colors=['orange', 'palegreen', 'aqua', 'blue'], fontsize=18, legend=False, title='2011 Sales Distribution as per regions').set_ylabel('')
table = df.pivot_table('Global_Sales', index='Name', columns='Year')
table.columns = table.columns.astype(int)
games = table.idxmax()
sales = table.max()
years = table.columns
data = pd.concat([games, sales], axis=1)
data.columns = ['Game', 'Global Sales']


colors = sns.color_palette("deep", len(years))
plt.figure(figsize=(14,10))
ax = sns.barplot(y = years , x = 'Global Sales', data=data, orient='h', palette=colors)
ax.set_xlabel(xlabel='Global Sales Per Year', fontsize=16)
ax.set_ylabel(ylabel='Year', fontsize=16)
ax.set_title(label='Highest Revenue Per Game in $ Millions Per Year', fontsize=20)
plt.show();
data
genre_comparison = df[['Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
comparison = genre_comparison.groupby(by=['Genre']).sum()
plt.figure(figsize=(12, 8))
sns.set(font_scale=1)
sns.heatmap(comparison,cmap="YlGnBu",linewidths=.5, annot=True)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(ylabel='Genre', fontsize=16)
plt.title('Sales comparison by Genre in different regions', fontsize=20)
plt.show()
comparison_table = comparison.reset_index()
comparison_table = pd.melt(comparison_table, id_vars=['Genre'], value_vars=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], var_name='Sale_Region', value_name='Sale_Price')
comparison_table.head()
plt.figure(figsize = (15,8))
sns.barplot(x='Genre', y = 'Sale_Price', hue = 'Sale_Region', data = comparison_table, palette='deep')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(xlabel='Genre', fontsize=16)
plt.ylabel(ylabel='Sale_Price', fontsize=16)
plt.title('Sales comparison of different regions based on genre type', fontsize=20)
top_sale_reg = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
top_sale_reg = top_sale_reg.sum().reset_index()
top_sale_reg = top_sale_reg.rename(columns={"index": "region", 0: "sale"})
top_sale_reg
plt.figure(figsize = (10,7))
ax = sns.barplot(x='region',y='sale',data=top_sale_reg)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(xlabel='Regions', fontsize=16)
plt.ylabel(ylabel='Sales in total', fontsize=16)
plt.title('Overall Sales comparison of different regions.', fontsize=20)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext= (0, 10), textcoords = 'offset points')
top_publishers = df.groupby(['Publisher']).count().iloc[:,0]
top_publishers = pd.DataFrame(top_publishers.sort_values(ascending=False))[0:10]
publishers = top_publishers.index
top_publishers.columns = ['Releases']

colors = sns.color_palette("summer", len(top_publishers))
plt.figure(figsize=(13,8))
ax = sns.barplot(y = publishers , x = 'Releases', data=top_publishers, orient='h', palette=colors)
ax.set_xlabel(xlabel='Number of Releases', fontsize=16)
ax.set_ylabel(ylabel='Publisher', fontsize=17)
ax.set_title(label='Top 10 Total Publisher Games Released', fontsize=20)
ax.set_yticklabels(labels = publishers, fontsize=14)
plt.show();
top_publishers_rev = df.groupby(['Publisher']).sum()['Global_Sales']
top_publishers_rev = pd.DataFrame(top_publishers_rev.sort_values(ascending=False))[0:10]
publishers = top_publishers_rev.index
top_publishers_rev.columns = ['Global Sales']

colors = sns.color_palette("winter", len(top_publishers_rev))
plt.figure(figsize=(13,8))
ax = sns.barplot(y = publishers , x = 'Global Sales', data=top_publishers_rev, orient='h', palette=colors)
ax.set_xlabel(xlabel='Revenue in $ Millions', fontsize=16)
ax.set_ylabel(ylabel='Publisher', fontsize=17)
ax.set_title(label='Top 10 Total Publisher Game Revenue', fontsize=20)
ax.set_yticklabels(labels = publishers, fontsize=14)
plt.show();
rev = df.groupby(['Genre']).sum()['Global_Sales']
rev = pd.DataFrame(rev.sort_values(ascending=False))
genres = rev.index
rev.columns = ['Revenue']

colors = sns.color_palette('Set3', len(rev))
plt.figure(figsize=(12,8))
ax = sns.barplot(y = genres , x = 'Revenue', data=rev, orient='h', palette=colors)
ax.set_xlabel(xlabel='Revenue in $ Millions', fontsize=16)
ax.set_ylabel(ylabel='Genre', fontsize=16)
ax.set_title(label='Genres by Total Revenue Generated in $ Millions', fontsize=20)
ax.set_yticklabels(labels = genres, fontsize=14)
plt.show();

data = pd.concat([df['Name'][0:10], df['Global_Sales'][0:10]], axis=1)

plt.figure(figsize=(12,8))
colors = sns.color_palette("CMRmap", len(data))
ax = sns.barplot(y = 'Name' , x = 'Global_Sales', data=data, orient='h', palette=colors)
ax.set_xlabel(xlabel='Revenue in $ Millions', fontsize=16)
ax.set_ylabel(ylabel='Name', fontsize=16)
ax.set_title(label='Top 10 Games by Revenue Generated in $ Millions', fontsize=20)
ax.set_yticklabels(labels = games, fontsize=14)
plt.style.use('ggplot')
plt.show();
EU = df.pivot_table('EU_Sales', columns='Name', index='Year', aggfunc='sum').sum(axis=1)
NA = df.pivot_table('NA_Sales', columns='Name', index='Year', aggfunc='sum').sum(axis=1)
JP = df.pivot_table('JP_Sales', columns='Name', index='Year', aggfunc='sum').sum(axis=1)
Other = df.pivot_table('Other_Sales', columns='Name', index='Year', aggfunc='sum').sum(axis=1)
years = Other.index.astype(int)
regions = ['European Union','Japan','North America','Other']

plt.figure(figsize=(12,8))
ax = sns.pointplot(x=years, y=EU, color='mediumslateblue', scale=0.7)
ax = sns.pointplot(x=years, y=JP, color='orchid', scale=0.7)
ax = sns.pointplot(x=years, y=NA, color='midnightblue', scale=0.7)
ax = sns.pointplot(x=years, y=Other, color='thistle', scale=0.7)
ax.set_xticklabels(labels=years, fontsize=12, rotation=50)
ax.set_xlabel(xlabel='Year', fontsize=16)
ax.set_ylabel(ylabel='Revenue in $ Millions', fontsize=16)
ax.set_title(label='Distribution of Total Revenue Per Region by Year in $ Millions', fontsize=20)
ax.legend(handles=ax.lines[::len(years)+1], labels=regions, fontsize=18)
plt.style.use('ggplot')
plt.show();
data = df
data = pd.DataFrame([data['EU_Sales'], data['JP_Sales'], data['NA_Sales'], data['Other_Sales']]).T
regions = ['European Union', 'Japan', 'North America', 'Other']
q = data.quantile(0.90)
data = data[data < q]
plt.figure(figsize=(12,8))

colors = sns.color_palette("deep", len(data))
ax = sns.boxplot(data=data, orient='h', palette=colors)
ax.set_xlabel(xlabel='Revenue per Game in $ Millions', fontsize=16)
ax.set_ylabel(ylabel='Region', fontsize=16)
ax.set_title(label='Distribution of Sales Per Game in $ Millions Per Region', fontsize=20)
ax.set_yticklabels(labels=regions, fontsize=14)
plt.style.use('ggplot')
plt.show()

plt.figure(figsize=(13,10))
sns.heatmap(df.corr(), cmap = "Blues", annot=True, linewidth=3)
data_pair = df.loc[:,["Year","Platform", "Genre", "NA_Sales","EU_Sales", "Other_Sales"]]
data_pair.head(2)
plt.style.use('ggplot')
sns.pairplot(data_pair, hue='Genre')
