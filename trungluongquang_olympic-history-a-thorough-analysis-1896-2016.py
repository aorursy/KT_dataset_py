# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
athlete_events = pd.read_csv('../input/athlete_events.csv')
regions = pd.read_csv('../input/noc_regions.csv')
# we merge the two data frame into one so that we can extract the region and take a quick look
athlete_events = pd.merge(athlete_events, regions, on = 'NOC', how = 'left')
athlete_events.head()
athlete_events.isnull().sum()
# because we plan to analyze data for summer and winter Olympics separately,
# it would be better to prepare two dataframes accordingly
summer_athlete_events = athlete_events[athlete_events['Season'] == 'Summer']
winter_athlete_events = athlete_events[athlete_events['Season'] == 'Winter']
tmp_summer = summer_athlete_events.groupby('Year', as_index = False).count()
tmp_winter = winter_athlete_events.groupby('Year', as_index = False).count()
fig = plt.figure(figsize = (16, 6))

sns.lineplot(x = 'Year', y = 'Name', data = tmp_summer, label = 'Summer Olympics', color = 'r', marker = 'o')
sns.lineplot(x = 'Year', y = 'Name', data = tmp_winter, label = 'Winter Olympics', color = 'b', marker = 'o')
plt.xlabel('Year')
plt.xticks(athlete_events['Year'].unique(), rotation = 60)

plt.ylabel('Number of Athletes')
plt.title("Number of Athletes over years",color="b")
plt.grid(True,alpha=.2)

# remove chart junk
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().legend()
plt.gca().grid(False)

plt.show()
summer_data = (summer_athlete_events[['Year', 'NOC']].drop_duplicates(['Year', 'NOC']))['Year'].value_counts()
winter_data = (winter_athlete_events[['Year', 'NOC']].drop_duplicates(['Year', 'NOC']))['Year'].value_counts()
plt.figure(figsize = (16, 6))
sns.lineplot(x = summer_data.index.values, y = summer_data.values, color = 'r', label = 'Summer Olympics', marker = 'o')
sns.lineplot(x = winter_data.index.values, y = winter_data.values, color = 'b', label = 'Winter Olympics', marker = 'o')
plt.ylabel('Number of countries')
plt.title('Number of countries participating in the Olympics')

# remove chart junk
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
summer_mean_age = summer_athlete_events[pd.notnull(summer_athlete_events['Age'])]
winter_mean_age = winter_athlete_events[pd.notnull(winter_athlete_events['Age'])]
fig = plt.figure(figsize = (16, 6))
sns.lineplot(x = 'Year', y = 'Age', data = summer_mean_age, label = 'Summer Olympics', color = 'red', marker = 'o')
sns.lineplot(x = 'Year', y = 'Age', data = winter_mean_age, label = 'Winter Olympics', color = 'blue', marker = 'o')

plt.title('Average Age by Year')
plt.ylabel('Average Age')

# remove chart junk
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
summer_age = summer_athlete_events[pd.notnull(summer_athlete_events['Age'])].groupby('Sport', as_index = False)['Age'].mean()
winter_age = winter_athlete_events[pd.notnull(winter_athlete_events['Age'])].groupby('Sport', as_index = False)['Age'].mean()
summer_oldest = summer_athlete_events[summer_athlete_events['Age'] == summer_athlete_events['Age'].max()]
summer_youngest = summer_athlete_events[summer_athlete_events['Age'] == summer_athlete_events['Age'].min()]
winter_oldest = winter_athlete_events[winter_athlete_events['Age'] == winter_athlete_events['Age'].max()]
winter_youngest = winter_athlete_events[winter_athlete_events['Age'] == winter_athlete_events['Age'].min()]

print('In the Summer Olympics:')
print('-',summer_age.max()['Sport'], 'is the sport with the heightest average age with', summer_age.max()['Age'])
print('-',summer_age.min()['Sport'], 'is the sport with the lowest average age with', summer_age.min()['Age'])
print('- The oldest athelete is', summer_oldest['Name'].iloc[0], '(',summer_oldest['Age'].iloc[0],')', 'of', summer_oldest['Sport'].iloc[0], 'from', summer_oldest['Team'].iloc[0], 'in', summer_oldest['Games'].iloc[0])
print('- The youngest athelete is', summer_youngest['Name'].iloc[0], '(',summer_youngest['Age'].iloc[0],')', 'of', summer_youngest['Sport'].iloc[0], 'from', summer_youngest['Team'].iloc[0], 'in', summer_youngest['Games'].iloc[0])

print('')

print('In the Winter Olympics:')
print('-',winter_age.max()['Sport'], 'is the sport with the heightest average age with', winter_age.max()['Age'])
print('-',winter_age.min()['Sport'], 'is the sport with the lowest average age with', winter_age.min()['Age'])
print('- The oldest athelete is', winter_oldest['Name'].iloc[0], '(',winter_oldest['Age'].iloc[0],')', 'of', winter_oldest['Sport'].iloc[0], 'from', winter_oldest['Team'].iloc[0], 'in', winter_oldest['Games'].iloc[0])
print('- The youngest athelete is', winter_youngest['Name'].iloc[0], '(',winter_youngest['Age'].iloc[0],')', 'of', winter_youngest['Sport'].iloc[0], 'from', winter_youngest['Team'].iloc[0], 'in', winter_youngest['Games'].iloc[0])
bins_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]

plt.figure(figsize = (16, 6))

# Summer Olympics
plt.subplot(121)
sns.distplot(summer_athlete_events['Age'].dropna(), bins = bins_list)
plt.xticks(bins_list)
plt.xlabel("Age")
plt.title("Age distribution of Summer Olympics Atheletes")

# remove chart junk
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Winter Olympics

plt.subplot(122)
sns.distplot(winter_athlete_events['Age'].dropna(), bins = bins_list)
plt.xticks(bins_list)
plt.xlabel("Age")
plt.title("Age distribution of Winter Olympics Atheletes")

# remove chart junk
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
summer_gender = summer_athlete_events.groupby('Sex')
winter_gender = winter_athlete_events.groupby('Sex')
plt.figure(figsize = (10, 4.5))

plt.subplot(121)
summer_gender["ID"].nunique().plot.pie(autopct = "%1.0f%%",wedgeprops = {"linewidth":2,"edgecolor":"w"},
                                              explode = [0,.01],shadow = True , colors = ["royalblue","lawngreen"]) 
plt.ylabel("")
circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(circ)
plt.title("SUMMER OLYMPICS")

plt.subplot(122)
winter_gender["ID"].nunique().plot.pie(autopct = "%1.0f%%",wedgeprops = {"linewidth":2,"edgecolor":"w"},
                                              explode = [0,.01],shadow = True , colors = ["royalblue","lawngreen"]) 
plt.ylabel("")
circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(circ)
plt.title("WINTER OLYMPICS")
summer_athlete_height = summer_athlete_events[pd.notnull(summer_athlete_events['Height'])]
summer_athlete_height.drop_duplicates(subset=['Name', 'NOC', 'Year'], keep='first', inplace=True)

winter_athlete_height = winter_athlete_events[pd.notnull(winter_athlete_events['Height'])]
winter_athlete_height.drop_duplicates(subset=['Name', 'NOC', 'Year'], keep='first', inplace=True)
plt.figure(figsize = (16, 5))
bins_list = [110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]

plt.subplot(121)
plt.xticks(bins_list)
sns.distplot(summer_athlete_height['Height'], bins = bins_list)
plt.title('Height Distribution of Summer Olympics atheletes')
plt.xlabel('cm')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.subplot(122)
plt.xticks(bins_list)
sns.distplot(winter_athlete_height['Height'], bins = bins_list)
plt.title('Height Distribution of Winter Olympics atheletes')
plt.xlabel('cm')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
summer_athlete_weight = summer_athlete_events[pd.notnull(summer_athlete_events['Weight'])]
summer_athlete_weight.drop_duplicates(subset=['Name', 'NOC', 'Year'], keep='first', inplace=True)

winter_athlete_weight = winter_athlete_events[pd.notnull(winter_athlete_events['Weight'])]
winter_athlete_weight.drop_duplicates(subset=['Name', 'NOC', 'Year'], keep='first', inplace=True)
plt.figure(figsize = (16, 5))
bins_list = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]

plt.subplot(121)
plt.xticks(bins_list)
sns.distplot(summer_athlete_weight['Weight'], bins = bins_list)
plt.title('Weight Distribution of Summer Olympics atheletes')
plt.xlabel('kg')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.subplot(122)
plt.xticks(bins_list)
sns.distplot(winter_athlete_weight['Weight'], bins = bins_list)
plt.title('Weight Distribution of Winter Olympics atheletes')
plt.xlabel('kg')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
summer_medals = summer_athlete_events[pd.notnull(summer_athlete_events['Medal'])]
winter_medals = winter_athlete_events[pd.notnull(winter_athlete_events['Medal'])]
top_n_countries = pd.value_counts(summer_medals['region']).iloc[:10].index
tmp_country_medals = summer_medals[summer_medals['region'].isin(top_n_countries)]
tmp_country_medals
country_medals = pd.DataFrame(columns = ['region', 'Gold', 'Silver', 'Bronze', 'Total'])
for i in top_n_countries:
    tmp = summer_medals[summer_medals['region'] == i]
    medals = pd.value_counts(tmp['Medal'])
    country_medals = country_medals.append({'region': i, 'Gold': medals['Gold'], 'Silver': medals['Silver'], 
                                            'Bronze': medals['Bronze'], 'Total': medals['Gold'] + medals['Silver'] + medals['Bronze']
                                           },ignore_index=True)
    country_medals.sort_values('Total', ascending = False, inplace = True)

x = country_medals['region']    
    
plt.figure(figsize = (9, 6))

gold = plt.bar(x, country_medals['Gold'], color = 'yellow', label = 'Gold Medal')
silver = plt.bar(x, country_medals['Silver'], bottom = country_medals['Gold'], color = 'silver', label = 'Silver Medal')
bronze = plt.bar(x, country_medals['Bronze'], bottom = country_medals['Gold'] + country_medals['Silver'], color = 'orange', label = 'Bronze Medal')
plt.xticks(rotation = 60)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()
plt.xlabel('Country')
plt.title('Medal distribution by country in the Summer Olympics')
top_n_countries = pd.value_counts(winter_medals['region']).iloc[:10].index
tmp_country_medals = winter_medals[winter_medals['region'].isin(top_n_countries)]
tmp_country_medals
country_medals = pd.DataFrame(columns = ['region', 'Gold', 'Silver', 'Bronze', 'Total'])
for i in top_n_countries:
    tmp = winter_medals[winter_medals['region'] == i]
    medals = pd.value_counts(tmp['Medal'])
    country_medals = country_medals.append({'region': i, 'Gold': medals['Gold'], 'Silver': medals['Silver'], 
                                            'Bronze': medals['Bronze'], 'Total': medals['Gold'] + medals['Silver'] + medals['Bronze']
                                           },ignore_index=True)
    country_medals.sort_values('Total', ascending = False, inplace = True)

x = country_medals['region']    
    
plt.figure(figsize = (9, 6))

gold = plt.bar(x, country_medals['Gold'], color = 'yellow', label = 'Gold Medal')
silver = plt.bar(x, country_medals['Silver'], bottom = country_medals['Gold'], color = 'silver', label = 'Silver Medal')
bronze = plt.bar(x, country_medals['Bronze'], bottom = country_medals['Gold'] + country_medals['Silver'], color = 'orange', label = 'Bronze Medal')

plt.xticks(rotation = 60)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()
plt.xlabel('Country')
plt.title('Medal distribution by country in the Winter Olympics')