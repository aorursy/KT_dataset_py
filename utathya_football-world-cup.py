# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.decomposition import PCA

df_rank = pd.read_csv('../input/wc_rank.csv')
df_age = pd.read_csv('../input/wc_age.csv')
df_prog = pd.read_csv('../input/wc_progress.csv')
# Converting ['United States' 'Saudi Arabia' 'Yugoslavia' 'South Korea' 'South Africa']

rank_list = df_rank.Team.tolist()
change_list = ['United_States', 'Saudi_Arabia', 'FR_Yugoslavia', 'South_Korea', 'South_Africa']
for index, team in enumerate(['United States', 'Saudi Arabia', 'Yugoslavia', 'South Korea', 'South Africa']):
    idx = [i for i, x in enumerate(rank_list) if x == team]
    for i in idx:
        rank_list[i] = change_list[index]
    df_rank.Team = rank_list
    
#Converting ['Republic of Ireland' 'Costa Rica' 'China PR']

rank_list = df_rank.Team.tolist()
change_list = ['Republic_of_Ireland', 'Costa_Rica', 'China_PR']
for index, team in enumerate(['Republic of Ireland', 'Costa Rica', 'China PR']):
    idx = [i for i, x in enumerate(rank_list) if x == team]
    for i in idx:
        rank_list[i] = change_list[index]
    df_rank.Team = rank_list
    
# Converting ['Czech Republic' 'Serbia and Montenegro' 'Ivory Coast'\n 'Trinidad and Tobago']

rank_list = df_rank.Team.tolist()
change_list = ['Czech_Republic', 'Serbia_and_Montenegro','Ivory_Coast', 'Trinidad_and_Tobago']
for index, team in enumerate(['Czech Republic', 'Serbia and Montenegro','Ivory Coast', 'Trinidad and Tobago']):
    idx = [i for i, x in enumerate(rank_list) if x == team]
    for i in idx:
        rank_list[i] = change_list[index]
    df_rank.Team = rank_list
    
# Converting ['New Zealand' 'North Korea']

rank_list = df_rank.Team.tolist()
change_list = ['New_Zealand', 'North_Korea']
for index, team in enumerate(['New Zealand', 'North Korea']):
    idx = [i for i, x in enumerate(rank_list) if x == team]
    for i in idx:
        rank_list[i] = change_list[index]
    df_rank.Team = rank_list

# Converting ['Bosnia and Herzegovina']

rank_list = df_rank.Team.tolist()
change_list = ['Bosnia_and_Herzegovina']
for index, team in enumerate(['Bosnia and Herzegovina']):
    idx = [i for i, x in enumerate(rank_list) if x == team]
    for i in idx:
        rank_list[i] = change_list[index]
    df_rank.Team = rank_list
# Creating new datframe
df_wc = pd.DataFrame(columns = ['Team', '1998', '2002', '2006', '2010', '2014'])

# Adding Team column
df_wc['Team'] = df_age.Team.unique()

# Sorting by Team name and setting Team as index
df_wc = df_wc.sort_values(['Team']).reset_index(drop = True).set_index('Team')

# Assigning age to new dataframe
df_wc.loc[df_age[df_age.Year.isin(['1998'])].Team.unique().tolist(), '1998'] = df_age.Age[df_age.Year.isin(['1998'])].tolist()
df_wc.loc[df_age[df_age.Year.isin(['2002'])].Team.unique().tolist(), '2002'] = df_age.Age[df_age.Year.isin(['2002'])].tolist()
df_wc.loc[df_age[df_age.Year.isin(['2006'])].Team.unique().tolist(), '2006'] = df_age.Age[df_age.Year.isin(['2006'])].tolist()
df_wc.loc[df_age[df_age.Year.isin(['2010'])].Team.unique().tolist(), '2010'] = df_age.Age[df_age.Year.isin(['2010'])].tolist()
df_wc.loc[df_age[df_age.Year.isin(['2014'])].Team.unique().tolist(), '2014'] = df_age.Age[df_age.Year.isin(['2014'])].tolist()


df_wc.info()

# renaming columns
df_wc.columns = ['Age98', 'Age02', 'Age06', 'Age10', 'Age14']

# Adding Rank columns
df_wc['Rank98'] = np.nan
df_wc['Rank02'] = np.nan
df_wc['Rank06'] = np.nan
df_wc['Rank10'] = np.nan
df_wc['Rank14'] = np.nan

# Assigning right values to columns
df_wc.loc[df_rank[df_rank.Year.isin(['1998'])].Team.unique().tolist(), 'Rank98'] = df_rank.Rank[df_rank.Year.isin(['1998'])].tolist()
df_wc.loc[df_rank[df_rank.Year.isin(['2002'])].Team.unique().tolist(), 'Rank02'] = df_rank.Rank[df_rank.Year.isin(['2002'])].tolist()
df_wc.loc[df_rank[df_rank.Year.isin(['2006'])].Team.unique().tolist(), 'Rank06'] = df_rank.Rank[df_rank.Year.isin(['2006'])].tolist()
df_wc.loc[df_rank[df_rank.Year.isin(['2010'])].Team.unique().tolist(), 'Rank10'] = df_rank.Rank[df_rank.Year.isin(['2010'])].tolist()
df_wc.loc[df_rank[df_rank.Year.isin(['2014'])].Team.unique().tolist(), 'Rank14'] = df_rank.Rank[df_rank.Year.isin(['2014'])].tolist()

df_wc
# Adding means
df_wc.loc['ZMean'] = df_wc.mean()
df_wc['Age_Mean'] = df_wc.iloc[:, 0:5].mean(axis=1)
df_wc['Rank_Mean'] = df_wc.iloc[:, 5:].mean(axis=1)
df_wc.tail(10)
# Visualising data distribution
g = sns.FacetGrid(df_age, col="Year")
g.map(sns.distplot, "Age", bins=1)

m = sns.FacetGrid(df_rank, col="Year")
m.map(sns.distplot, "Rank", bins=1)
# Combine df_age and df_rank
df_comb = df_age.merge(df_rank, how='outer', on=['Team', 'Year'])

# Create a box plot for age
g1 = sns.boxplot(y=df_comb['Age'], x=df_comb['Year'])
g1.set_title("Average Age per Year")
g1.set_xticklabels(df_comb['Year'].unique(),rotation=20)
# Create a box plot for rank
g2 = sns.boxplot(y=df_comb['Rank'], x=df_comb['Year'])
g2.set_title("Fifa Rank per Year")
g2.set_xticklabels(df_comb['Year'].unique(),rotation=20)
# Add Champions data
df_comb.loc[df_comb.Year.isin(['1998']) & df_comb.Team.isin(['France']), 'Champion'] = 1
df_comb.loc[df_comb.Year.isin(['2002']) & df_comb.Team.isin(['Brazil']), 'Champion'] = 1
df_comb.loc[df_comb.Year.isin(['2006']) & df_comb.Team.isin(['Italy']), 'Champion'] = 1
df_comb.loc[df_comb.Year.isin(['2010']) & df_comb.Team.isin(['Spain']), 'Champion'] = 1
df_comb.loc[df_comb.Year.isin(['2014']) & df_comb.Team.isin(['Germany']), 'Champion'] = 1

df_comb.loc[df_comb['Champion'].isnull(), 'Champion'] = 0
df_comb.loc[:, 'Champion'] = df_comb['Champion'].astype(int)
df_comb.sample(5)

# View lmplot of Champions cluster on a scatterplot
sns.lmplot(x='Rank', y='Age', hue='Champion', data=df_comb, fit_reg=False)