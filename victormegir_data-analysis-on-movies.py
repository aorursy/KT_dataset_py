import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 16})
movies = pd.read_csv('../input/movies/movies.csv')
new_column_names = []

for col in movies.columns:
    new_column_names.append('_'.join(col.split()))

movies.columns = new_column_names
movies.describe()
movies.replace('Unknown',np.nan, inplace=True)
movies.isnull().sum()
movies.dropna(subset=['IMDB_Rating', 'IMDB_Votes', 'US_Gross', 'Worldwide_Gross'], inplace=True)
print('We have dropped {:.2f}% of the total datapoints.'.format((3201 - movies.shape[0]) *100 / 3201))
movies['US_Gross'] = pd.to_numeric(movies['US_Gross'])
movies['Worldwide_Gross'] = pd.to_numeric(movies['Worldwide_Gross'])
movies['IMDB_Votes'] = pd.to_numeric(movies['IMDB_Votes'])
sns.distplot(movies['Worldwide_Gross']/10**6, bins=100)
plt.xlabel('Worldwide Gross in million U.S. Dollars', fontsize=16);
sns.distplot(movies['IMDB_Rating'], bins=100)
plt.xlabel('IMDB Rating', fontsize=16);
sns.distplot(movies['IMDB_Votes']/10**3, bins=100)
plt.xlabel('Thousands of IMDB Votes', fontsize = 16);
sns.distplot(movies['Rotten_Tomatoes_Rating'], bins=100)
plt.xlabel('Rotten Tomatoes Rating', fontsize=16);
from scipy import stats
plt.rcParams['figure.figsize'] = [20, 10]
f, axes = plt.subplots(1, 2)

sns.distplot(movies['Worldwide_Gross'], bins=100, hist=False, fit=stats.powerlaw, ax=axes[0])
axes[0].set_xlabel('Worldwide Gross', fontsize=16)

sns.distplot(movies['IMDB_Votes'], hist=False, fit=stats.powerlaw, ax=axes[1])
axes[1].set_xlabel('IMDB Votes', fontsize=16);
f, axes = plt.subplots(1, 2)

sns.distplot(movies['IMDB_Rating'], bins=100, hist=False, fit=stats.norm, ax=axes[0])
axes[0].set_xlabel('IMDB Rating', fontsize=16);

sns.distplot(movies['Rotten_Tomatoes_Rating'], bins=100, hist=False, fit=stats.gennorm, ax=axes[1])
axes[1].set_xlabel('Rotten Tomatoes Rating', fontsize=16);
plt.rcParams['figure.figsize'] = [10, 10]
genre = movies.groupby('Major_Genre').count().sort_values(by='Title', ascending=False)
plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams.update({'font.size': 10})

sns.barplot(genre.index, genre['Title'], palette='GnBu_d')
plt.xlabel('Major Genre', fontsize=16)
plt.ylabel('Number of Movies', fontsize=16);
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 16})
plt.hist(movies['Worldwide_Gross']/10**6, bins=[0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560], rwidth=0.8)
plt.xlabel('Worldwide Gross in milions of U.S. dollars', fontsize = 16)
plt.ylabel('Number of movies', fontsize = 16);
plt.hist(movies['IMDB_Votes']/10**3, bins=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512], rwidth=0.8)
plt.xlabel('Thousands of IMDB Votes', fontsize = 16)
plt.ylabel('Number of movies', fontsize = 16);
sns.scatterplot(movies['IMDB_Votes']/10**3, movies['Worldwide_Gross']/10**6)
plt.ylabel('Worldwide Gross in million U.S. Dollars', fontsize=16)
plt.xlabel('Thousands of IMDB Votes', fontsize = 16);
splot = sns.regplot(movies['IMDB_Votes']/10**3, movies['Worldwide_Gross']/10**6, 
                    scatter_kws = {'color': 'blue', 'alpha': 0.2}, line_kws = {'color': 'red'})
splot.set(xscale='log')
splot.set(yscale='log')

plt.ylabel('Worldwide Gross in milion U.S. dollars', fontsize=16)
plt.xlabel('Thousands of IMDB Votes', fontsize = 16);
sns.pairplot(movies[['Worldwide_Gross', 'Rotten_Tomatoes_Rating', 'IMDB_Votes', 'IMDB_Rating']]);
from scipy import stats

def correlations(x, y):
    spearman = stats.spearmanr(x, y)
    pearson = stats.pearsonr(x, y)

    print("Spearman correlation coefficient for {} and {}:".format(x.name, y.name), spearman.correlation)
    print("Pearson correlation coefficient for {} and {}:".format(x.name, y.name), pearson[0])
    print("P-value correlation coefficient for {} and {}:".format(x.name, y.name), spearman.pvalue)
    print("")
no_null_tomatoes = movies.dropna(subset=['Rotten_Tomatoes_Rating'])

correlations(no_null_tomatoes['Rotten_Tomatoes_Rating'], no_null_tomatoes['IMDB_Rating'])
correlations(movies['IMDB_Votes'], movies['IMDB_Rating'])
correlations(no_null_tomatoes['IMDB_Votes'], no_null_tomatoes['Rotten_Tomatoes_Rating'])

correlations(no_null_tomatoes['Worldwide_Gross'], no_null_tomatoes['Rotten_Tomatoes_Rating'])
correlations(movies['Worldwide_Gross'], movies['IMDB_Rating'])
correlations(movies['Worldwide_Gross'], movies['IMDB_Votes'])
sns.regplot(no_null_tomatoes['Rotten_Tomatoes_Rating'], no_null_tomatoes['IMDB_Rating'], 
            scatter_kws = {'color': 'blue', 'alpha': 0.2}, line_kws = {'color': 'red'})

plt.ylabel('IMDB Rating', fontsize=16)
plt.xlabel('Rotten Tomatoes Rating', fontsize=16);
sns.regplot(movies['Worldwide_Gross']/10 ** 6, movies['IMDB_Votes']/10**3,
            scatter_kws = {'color': 'blue', 'alpha': 0.2}, line_kws = {'color': 'red'})

plt.ylabel('Thousands of IMDB Votes', fontsize = 16)
plt.xlabel('Worldwide Gross in milion U.S. dollars', fontsize=16);
plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams.update({'font.size': 10})

sns.barplot(movies['Major_Genre'], movies['Worldwide_Gross']/10 ** 6, palette='GnBu_d')
plt.ylabel('Worldwide Gross in milion U.S. dollars', fontsize=16)
plt.xlabel('Major Genre', fontsize=16);
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 16})
def t_test(genre1, genre2):
    x = movies[movies['Major_Genre'] == genre1]['Worldwide_Gross']
    y = movies[movies['Major_Genre'] == genre2]['Worldwide_Gross']
    
    print('The p-value for genres {} and {} is:'.format(genre1, genre2) ,stats.ttest_ind(x, y).pvalue)
t_test('Action', 'Adventure')
t_test('Action', 'Romantic Comedy')
t_test('Adventure', 'Romantic Comedy')
t_test('Action', 'Drama')
t_test('Documentary', 'Black Comedy')
t_test('Western', 'Horror')
t_test('Western', 'Drama')
t_test('Horror', 'Comedy')
movies['Release_Date'].replace('TBD', np.nan, inplace=True)
movies['Release_Date'].dropna()
def to_str(row):
    row_as_string = str(row)
    no_spaces = row_as_string.split()[-1]
    year = no_spaces[-2:]
    
    if not year.isdigit():
        return np.nan
    return year

movies['Year'] = movies['Release_Date'].apply(lambda row: to_str(row))
movies['Year'] = movies['Year'].dropna()
movies['Year'] = pd.to_numeric(movies['Year'])

def to_year(row):
    #1915 - 2011    
    if row >= 15:
        row += 1900
    else:
        row += 2000
    return row
    
movies['Year'] = movies['Year'].apply(lambda row: to_year(row))
movies['Year']
movies['Year'] = movies['Year'].fillna(0)
movies['Year'].astype('int')
movies = movies[movies['Year'] != 0]
sns.distplot(movies['Year'], bins=100);
def roundup(x):
    return int(math.floor(x / 10.0)) * 10

movies['Decade'] = movies['Year'].apply(lambda row: roundup(row))
sns.barplot(movies['Decade'], movies['Rotten_Tomatoes_Rating'], palette='GnBu_d')
plt.ylabel('Rotten Tomatoes Rating');
sns.barplot(movies['Decade'], movies['IMDB_Rating'], palette='GnBu_d')
plt.ylabel('IMDB Rating');
movies_after_20s = movies[movies['Decade'] > 1920]
difference_of_opinion = movies_after_20s['IMDB_Rating']*10 - movies_after_20s['Rotten_Tomatoes_Rating']

sns.barplot(movies_after_20s['Decade'], difference_of_opinion, palette='GnBu_d')
plt.xlabel('Decade after the 1920s')
plt.ylabel('Difference between IMDB and Rotten Tomatoes Rating');
no_null_directors = movies.dropna(subset=['Director'])
mean_stats = no_null_directors.groupby('Director')[['Worldwide_Gross', 'IMDB_Rating']].mean()

top_earning = mean_stats.sort_values(by='Worldwide_Gross', ascending=False)[:100]
top_rated  = mean_stats.sort_values(by='IMDB_Rating', ascending=False)[:100]
sns.distplot(top_earning['IMDB_Rating'], label='top earning')
sns.distplot(top_rated['IMDB_Rating'], label='top rated')
plt.legend();
sns.distplot(top_earning['Worldwide_Gross']/10**6, label='top earning')
sns.distplot(top_rated['Worldwide_Gross']/10**6, label='top rated')
plt.legend();
t1 = stats.ttest_ind(top_earning['Worldwide_Gross'], top_rated['Worldwide_Gross'])
t2 = stats.ttest_ind(top_earning['IMDB_Rating'], top_rated['IMDB_Rating'])

print('t-statistic: {}, p-value: {}'.format(t1.statistic, t1.pvalue))
print('t-statistic: {}, p-value: {}'.format(t2.statistic, t2.pvalue))
movies['Performance'] = movies['Worldwide_Gross'] - movies['Production_Budget']
movies['Flop'] = movies['Performance'].loc[movies['Performance'] < 0]
movies['Win'] = movies['Performance'].loc[movies['Performance'] > 0]

movies['Production_Budget'].fillna(0, inplace=True)
flops = movies.dropna(subset=['Flop'])
wins = movies.dropna(subset=['Win'])
plt.rcParams['figure.figsize'] = [20, 10]
f, axes = plt.subplots(1, 2)

sns.scatterplot(movies['Flop']*(-1), movies['Production_Budget'], ax=axes[0])
axes[0].set_xlabel('Flop', fontsize=16)
axes[0].set_ylabel('Production Budget', fontsize=16)

sns.scatterplot(movies['Win']/10**6, movies['Production_Budget'], ax=axes[1])
axes[1].set_xlabel('Win', fontsize=16)
axes[1].set_ylabel('Production Budget', fontsize=16);
correlations(flops['Production_Budget'], flops['Flop']*(-1))
correlations(wins['Production_Budget'], wins['Win'])
plt.rcParams['figure.figsize'] = [10, 10]

sns.regplot(movies['Worldwide_Gross']/10**6, movies['Production_Budget']/10**6, 
           scatter_kws = {'color': 'blue', 'alpha': 0.2}, line_kws = {'color': 'red'})
plt.xlabel('Worldwide Gross in milions of U.S. dollars', fontsize=16)
plt.ylabel('Production Budget in milions of U.S. dollars', fontsize=16)

correlations(movies['Worldwide_Gross'], movies['Production_Budget'])
plt.rcParams['figure.figsize'] = [20, 10]
f, axes = plt.subplots(1, 2)

sns.scatterplot(movies['IMDB_Rating'], movies['Flop']*(-1), ax=axes[0])
axes[0].set_xlabel('IMDB Rating', fontsize=16)
axes[0].set_ylabel('Flop', fontsize=16)

sns.scatterplot(movies['Rotten_Tomatoes_Rating'], movies['Flop']*(-1), ax=axes[1])
axes[1].set_xlabel('Rotten Toamtoes Rating', fontsize=16)
axes[1].set_ylabel('Flop', fontsize=16);
correlations(flops['IMDB_Rating'], flops['Flop'])
correlations(flops['Rotten_Tomatoes_Rating'].fillna(0), flops['Flop'])
plt.rcParams['figure.figsize'] = [10, 10]
flops_per_decade = flops.groupby('Decade').agg({'Flop' : 'count'})

sns.barplot(flops_per_decade.index, flops_per_decade['Flop'], palette='GnBu_d')
plt.ylabel('Flops', fontsize=16);
sns.barplot(flops['Decade'], flops['Flop']*(-1)/10**6, palette='GnBu_d')
plt.ylabel('Average loss for flops in U.S. dollars', fontsize=16);
sns.barplot(movies['Decade'], movies['Production_Budget']/10**6, palette='GnBu_d')
plt.ylabel('Production Budget in milions of U.S. dollars', fontsize=16);