import sys

import os

import ast

import time

import datetime

import warnings

import eli5



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import xgboost as xgb

import lightgbm as lgb



from PIL import Image

from numpy.linalg import norm

from collections import Counter

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor



%matplotlib inline

warnings.filterwarnings('ignore')

pd.set_option('max_rows', 10)

sns.set(style='white', context='notebook', palette='deep')
train_data = pd.read_csv('../input/tmdb-box-office-prediction/train.csv', )

test_data = pd.read_csv('../input/tmdb-box-office-prediction/test.csv', )



train_data.shape, test_data.shape
train_data.head()
train_data.select_dtypes(include=['int64','float64']).columns.values
# Summary statistics on numerical features, excluding the id

train_data.drop(labels=['id'], axis=1).describe()
# Let's visualize the distributions of numerical features

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

sns.set(color_codes=True, font_scale=1.3)



g = sns.distplot(train_data['budget'], color='blue', ax=axes[0,0])

skewness = train_data['budget'].skew()

axes[0,0].legend(["Skew : {:.2f}".format(skewness)])

plt.setp(g.get_legend().get_texts(), fontsize='16')



g = sns.distplot(train_data['popularity'], color='red', ax=axes[0,1])

skewness = train_data['popularity'].skew()

axes[0,1].legend(["Skew : {:.2f}".format(skewness)])

plt.setp(g.get_legend().get_texts(), fontsize='16')



g = sns.distplot(train_data['runtime'], color='green', ax=axes[1,0])

skewness = train_data['runtime'].skew()

axes[1,0].legend(["Skew : {:.2f}".format(skewness)])

plt.setp(g.get_legend().get_texts(), fontsize='16')



g = sns.distplot(train_data['revenue'], color='orange', ax=axes[1,1])

skewness = train_data['revenue'].skew()

axes[1,1].legend(["Skew : {:.2f}".format(skewness)])

plt.setp(g.get_legend().get_texts(), fontsize='16')



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

colors = sns.color_palette(n_colors=2)

sns.set(color_codes=True)



sns.regplot(x=train_data['budget'], y=train_data['revenue'], ax=axes[0], color=colors[0], scatter_kws={'alpha': 0.5})

correlation = np.corrcoef(train_data['budget'], train_data['revenue'])[0,1]

axes[0].set_title("Correlation {:.2f}".format(correlation))



sns.regplot(x=train_data['popularity'], y=train_data['revenue'], ax=axes[1], color=colors[1], scatter_kws={'alpha': 0.5})

correlation = np.corrcoef(train_data['popularity'], train_data['revenue'])[0,1]

axes[1].set_title("Correlation {:.2f}".format(correlation))
fig, ax = plt.subplots(figsize=(7,7))

colors = sns.color_palette(n_colors=1)

sns.set(color_codes=True)



sns.regplot(x=train_data['runtime'], y=train_data['revenue'], color='g', scatter_kws={'alpha': 0.5})

correlation = np.corrcoef(train_data['runtime'], train_data['revenue'])[0,1]

ax.set_title("Correlation {:.2f}".format(correlation))
features = ['budget', 'popularity', 'runtime', 'revenue']

sns.pairplot(train_data[features], kind='reg', diag_kind='kde', plot_kws={'scatter_kws': {'alpha': 0.2}}, size=3)

plt.show()
# Find categorical features

train_data.select_dtypes(include=['object']).columns.values
print("There are %i categorical features in total." % len(train_data.select_dtypes(include=['object']).columns.values))
# belongs_to_collection

percentage_missing = train_data['belongs_to_collection'].isnull().sum() / train_data.shape[0] * 100

print('%.2f percent of belongs_to_collection values are missing.' % percentage_missing)
# release_date

train_data['release_date']
# Status

fig = plt.figure(figsize=(9,6))

sns.set(font_scale=1.5)

sns.boxenplot(train_data['status'], train_data['revenue'])

plt.show()
train_data.loc[train_data['status']=='Rumored', ['original_title','overview','release_date','revenue']]
train_size = len(train_data)

all_data = pd.concat(objs=[train_data, test_data], axis=0).reset_index(drop=True).drop(labels=['id'], axis=1)

all_data.info()
all_data = all_data.fillna(np.nan)

missing_features = all_data.columns[all_data.isnull().any()]

missing_features.values
all_data[missing_features].isnull().sum()
# We extract collection name like this:

ast.literal_eval(all_data['belongs_to_collection'][1])[0]['name']
all_data['collection_name'] = all_data['belongs_to_collection'].apply(lambda x: 'None' if pd.isnull(x) else ast.literal_eval(x)[0]['name'])

all_data['collection_name']
all_data['collection_name'].describe()
all_data['collection_name'].value_counts().head(10)
all_data['in_collection'] = all_data['belongs_to_collection'].apply(lambda x: 'No' if pd.isnull(x) else 'Yes')
fig = plt.figure(figsize=(9,6))

sns.set(font_scale=1.5)

sns.boxenplot(all_data['in_collection'], all_data['revenue'])

plt.show()
# Extract genres

data_genres = all_data['genres'].apply(lambda y: {} if pd.isnull(y) else sorted(map(lambda x: x['name'], eval(y)))).map(lambda x: ','.join(map(str, x)))

data_genres
data_genres = data_genres.str.get_dummies(sep=',')

data_genres
number_genres = data_genres.sum(axis=1)
genres_sorted = data_genres.sum(axis=0).sort_values(ascending=False)
sns.set(font_scale=1.2)

fig, ax = plt.subplots(figsize=(20, 5))

sns.barplot(genres_sorted.index, genres_sorted.values)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize = (10, 6))

temp_genres = all_data['genres'].apply(lambda y: {} if pd.isnull(y) else sorted(map(lambda x: x['name'], eval(y))))

text = ' '.join([val for sublist in temp_genres for val in sublist])

wd = WordCloud(max_font_size=1000, background_color='white', collocations=False, width=1600, height=900).generate(text)



# Display the wordcloud

plt.imshow(wd, interpolation='bilinear')

plt.axis("off")

plt.show()
all_data = pd.concat([all_data, data_genres], axis=1, sort=False)

all_data['number_genres'] = number_genres
all_data['production_companies'][0]
data_companies = all_data['production_companies'].apply(lambda y: {} if pd.isnull(y) else sorted(map(lambda x: x['name'], eval(y)))).map(lambda x: ','.join(map(str, x)))

data_companies
data_companies[7393]
data_companies = data_companies.str.get_dummies(sep=',')
number_companies = data_companies.sum(axis=1)

number_companies
data_companies.sum(axis=0).sort_values(ascending=False).head(15)
data_companies = data_companies[data_companies.sum(axis=0).sort_values(ascending=False).head(15).index.values]
all_data = pd.concat([all_data, data_companies], axis=1, sort=False)

all_data['number_companies'] = number_companies
all_data['production_countries'][6]
data_countries = all_data['production_countries'].apply(lambda y: {} if pd.isnull(y) else sorted(map(lambda x: x['name'], eval(y)))).map(lambda x: ','.join(map(str, x)))

data_countries
data_countries = data_countries.str.get_dummies(sep=',')
data_countries.sum(axis=0).sort_values(ascending=False).head(16)
data_countries = data_countries[data_countries.sum(axis=0).sort_values(ascending=False).head(16).index.values]

data_countries
all_data = pd.concat([all_data, data_countries], axis=1, sort=False)
data_language = all_data['spoken_languages'].apply(lambda y: {} if pd.isnull(y) else sorted(map(lambda x: x['name'], eval(y)))).map(lambda x: ','.join(map(str, x)))

data_language
data_language = data_language.str.get_dummies(sep=',')
data_language.sum(axis=0).sort_values(ascending=False).head(20)
data_language = data_language[data_language.sum(axis=0).sort_values(ascending=False).head(20).index.values]

data_language
all_data = pd.concat([all_data, data_language], axis=1, sort=False)
Counter(all_data['status'])
all_data.loc[all_data['status'].isnull(), 'status'] = 'Released'
Counter(all_data['status'])
all_data['tagline'] = all_data['tagline'].apply(lambda x: 'no tagline' if pd.isnull(x) else 'has tagline')

all_data['tagline']
sns.set(font_scale=1.2)

fig, ax = plt.subplots(figsize=(12, 6))

sns.boxenplot(all_data['tagline'], all_data['revenue'])

plt.show()
all_data['Keywords'][100]
data_keywords = all_data['Keywords'].apply(lambda y: {} if pd.isnull(y) else sorted(map(lambda x: x['name'], eval(y)))).map(lambda x: ','.join(map(str, x)))

data_keywords
# Number of keywords

data_keywords[0]
data_keywords = data_keywords.str.get_dummies(sep=',')

data_keywords
number_keywords = data_keywords.sum(axis=1)
sns.set(font_scale=1.5)

fig, ax = plt.subplots(figsize=(10, 5))

sns.regplot(x=number_keywords[:3000], y=train_data['revenue'], scatter_kws={'alpha':0.5}, color='orange')

correlation = np.corrcoef(number_keywords[:3000], train_data['revenue'])[0,1]

ax.set_xlabel("number of keywords")

ax.set_title("Correlation {:.2f}".format(correlation))

plt.show()
data_keywords.sum(axis=0).sort_values(ascending=False).head(30)
data_keywords = data_keywords[data_keywords.sum(axis=0).sort_values(ascending=False).head(30).index.values]

data_keywords
all_data = pd.concat([all_data, data_keywords], axis=1, sort=False)

all_data['number_keywords'] = number_keywords
all_data['homepage'] = all_data['homepage'].apply(lambda x: 'no homepage' if pd.isnull(x) else 'has homepage')
sns.set(font_scale=1.2)

fig, ax = plt.subplots(figsize=(10, 5))

sns.distplot(all_data[:3000].loc[all_data['homepage']=='has homepage', 'revenue'])

sns.distplot(all_data[:3000].loc[all_data['homepage']=='no homepage', 'revenue'])

plt.legend(['has hompage', 'no homepage'])

plt.xticks(rotation=45)

plt.show()
sns.set(font_scale=1.5)

fig, ax = plt.subplots(figsize=(20, 5))

sns.countplot(all_data['original_language'])

plt.show()
all_data['in_English'] = all_data['original_language'].apply(lambda x: 'Yes' if x=='en' else 'No')
sns.set(font_scale=1.2)

fig, ax = plt.subplots(figsize=(10, 5))

sns.distplot(all_data['runtime'])

plt.xticks(rotation=45)

plt.show()
all_data.loc[all_data['runtime'].isnull(), 'runtime'] = all_data['runtime'].mode().values[0]
all_data['cast'][4]
cast_size = all_data['cast'].apply(lambda y: {} if pd.isnull(y) else sorted(map(lambda x: x['name'], eval(y)))).apply(lambda x: len(x))

cast_size
# Cast gender

cast_gender = all_data['cast'].apply(lambda y: {} if pd.isnull(y) else sorted(map(lambda x: x['gender'], eval(y))))

cast_gender
# Count each gender

cast_female_count = cast_gender.apply(lambda x: (pd.Series(x)==1).sum())

cast_male_count = cast_gender.apply(lambda x: (pd.Series(x)==2).sum())
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

sns.set(color_codes=True, font_scale=1.5)



sns.regplot(x=cast_female_count[:3000], y=train_data['revenue'], scatter_kws={'alpha':0.5}, color='pink', ax=axes[0])

correlation = np.corrcoef(cast_female_count[:3000], train_data['revenue'])[0,1]

axes[0].set_xlabel("female counts")

axes[0].set_title("Correlation {:.2f}".format(correlation))



sns.regplot(x=cast_male_count[:3000], y=train_data['revenue'], scatter_kws={'alpha':0.5}, color='blue', ax=axes[1])

correlation = np.corrcoef(cast_male_count[:3000], train_data['revenue'])[0,1]

axes[1].set_xlabel("male counts")

axes[1].set_title("Correlation {:.2f}".format(correlation))



plt.tight_layout()

plt.show()
sns.set(font_scale=1.5)

fig, ax = plt.subplots(figsize=(10, 5))

sns.regplot(x=cast_size[:3000], y=train_data['revenue'], scatter_kws={'alpha':0.5}, color='green')

correlation = np.corrcoef(cast_size[:3000], train_data['revenue'])[0,1]

ax.set_title("Correlation {:.2f}".format(correlation))

plt.show()
all_data['cast_size'] = cast_size

all_data['cast_male_count'] = cast_male_count

all_data['cast_female_count'] = cast_female_count
crew_size = all_data['crew'].apply(lambda y: {} if pd.isnull(y) else sorted(map(lambda x: x['name'], eval(y)))).apply(lambda x: len(x))

crew_size
# Crew gender

crew_gender = all_data['crew'].apply(lambda y: {} if pd.isnull(y) else sorted(map(lambda x: x['gender'], eval(y))))

crew_gender
# Count each gender

crew_female_count = crew_gender.apply(lambda x: (pd.Series(x)==1).sum())

crew_male_count = crew_gender.apply(lambda x: (pd.Series(x)==2).sum())
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

sns.set(color_codes=True, font_scale=1.5)



sns.regplot(x=crew_female_count[:3000], y=train_data['revenue'], scatter_kws={'alpha':0.5}, color='pink', ax=axes[0])

correlation = np.corrcoef(crew_female_count[:3000], train_data['revenue'])[0,1]

axes[0].set_xlabel("female counts")

axes[0].set_title("Correlation {:.2f}".format(correlation))



sns.regplot(x=crew_male_count[:3000], y=train_data['revenue'], scatter_kws={'alpha':0.5}, color='blue', ax=axes[1])

correlation = np.corrcoef(crew_male_count[:3000], train_data['revenue'])[0,1]

axes[1].set_xlabel("male counts")

axes[1].set_title("Correlation {:.2f}".format(correlation))



plt.tight_layout()

plt.show()
sns.set(font_scale=1.5)

fig, ax = plt.subplots(figsize=(10, 5))

sns.regplot(x=crew_size[:3000], y=train_data['revenue'], scatter_kws={'alpha':0.5}, color='orange')

correlation = np.corrcoef(crew_size[:3000], train_data['revenue'])[0,1]

ax.set_title("Correlation {:.2f}".format(correlation))

plt.show()
all_data['crew_size'] = crew_size

all_data['crew_male_count'] = crew_male_count

all_data['crew_female_count'] = crew_female_count
all_data.loc[all_data['release_date'].isnull(), 'release_date']
all_data.loc[all_data['release_date'].isnull(), 'release_date'] = '3/20/2001'
all_data.loc[all_data['release_date'].isnull(), 'release_date']
# Drop the unnecessary features

all_data = all_data.drop(labels=['belongs_to_collection','collection_name','genres','production_companies','production_countries','spoken_languages','Keywords','original_language','cast','crew'], axis=1)
all_data = all_data.drop(labels=['overview','poster_path','title'], axis=1)
all_data = all_data.drop(labels=['imdb_id','original_title'], axis=1)
all_data.columns[all_data.isnull().any()]
all_data.columns.values
data_date = all_data.release_date.apply(lambda x: x.split('/'))

data_month = data_date.apply(lambda x: x[0]).apply(int)

data_day = data_date.apply(lambda x: x[1]).apply(int)

data_year = data_date.apply(lambda x: x[2]).apply(int)
data_month.name = 'month'

data_day.name = 'day'

data_year.name = 'year'
data_year.loc[data_year==2001] = 1
data_year = data_year.apply(lambda x: x+2000 if x<=20 else x+1900)
sns.set(font_scale=1)

fig, ax = plt.subplots(figsize=(16, 8))

sns.countplot(data_year[:3000])

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,14))

sns.set(color_codes=True, font_scale=1.3)



sns.regplot(data_year[:3000], all_data['revenue'][:3000], scatter_kws={'alpha': 0.5}, ax=axes[0])

correlation = np.corrcoef(data_year[:3000], train_data['revenue'][:3000])[0,1]

axes[0].set_title("Correlation {:.2f}".format(correlation))



sns.set(font_scale=1)

sns.boxplot(data_year[:3000], all_data['revenue'][:3000], ax=axes[1])

plt.xticks(rotation=90)



plt.tight_layout()

plt.show()
# Include data_year in our dataset

all_data = pd.concat([all_data, data_year], axis=1, sort=False)

# all_data['year'] = data_year
# all_data.groupby("year")["revenue"].aggregate('mean').plot()
temp_data = pd.concat([all_data, data_month], axis=1, sort=False)

month_median = temp_data.groupby('month')['revenue'].median()

month_mean = temp_data.groupby('month')['revenue'].mean()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

sns.set(color_codes=True, font_scale=1.3)



sns.countplot(data_month[:3000], order=[1,2,3,4,5,6,7,8,9,10,11,12], ax=axes[0])

ax02 = axes[0].twinx()

ax02.grid(False)

g = sns.lineplot(month_mean.index.values-1, month_mean.values, color='#D0233C', label='mean')

plt.setp(g.get_legend().get_texts(), fontsize='16')



sns.set(font_scale=1)

sns.boxplot(data_month[:3000], all_data['revenue'][:3000], ax=axes[1])

ax12 = axes[1].twinx()

ax12.grid(False)

g = sns.lineplot(month_median.index.values-1, month_median.values, color='#C053AC', label='median')

plt.setp(g.get_legend().get_texts(), fontsize='16')



plt.tight_layout()

plt.show()
data_month_transformed = pd.get_dummies(data_month)
data_month_transformed.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
all_data = pd.concat([all_data, data_month_transformed], axis=1)
data_date = pd.concat([data_month, data_day, data_year], axis=1).apply(lambda x: "/".join([str(x.month), str(x.day), str(x.year)]), axis=1)

data_weekday = data_date.apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').weekday())

data_weekday = data_weekday.replace({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

sns.set(color_codes=True, font_scale=1.4)



sns.countplot(data_weekday[:3000], ax=axes[0])

sns.boxplot(data_weekday[:3000], all_data['revenue'][:3000], ax=axes[1])

# plt.xticks(rotation=0, fontsize=16)



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

sns.set(color_codes=True, font_scale=1)



sns.countplot(data_day[:3000], ax=axes[0])

sns.boxplot(data_day[:3000], all_data['revenue'][:3000], ax=axes[1])

# plt.xticks(rotation=0, fontsize=16)



plt.tight_layout()

plt.show()
data_weekday.name = 'weekday'

temp_data = pd.concat([train_data['revenue'], data_weekday[:3000]], axis=1, sort=False)

weekday_median = temp_data.groupby('weekday')['revenue'].median()

weekday_mean = temp_data.groupby('weekday')['revenue'].mean()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

sns.set(color_codes=True, font_scale=1.4)



g1 = sns.barplot(weekday_median.index.values, weekday_median.values, order=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], ax=axes[0], label='median')

g2 = sns.barplot(weekday_mean.index.values, weekday_mean.values, order=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], ax=axes[1], label='mean')

axes[0].set_title("Median")

axes[1].set_title("Mean")



plt.tight_layout()

plt.show()
all_data = pd.concat([all_data, pd.get_dummies(data_weekday)], axis=1)
all_data = all_data.drop(labels=['release_date'], axis=1)
all_data.select_dtypes('object').head()
all_data = pd.concat([all_data, pd.get_dummies(all_data.select_dtypes('object'), columns=['homepage','status','tagline','in_collection','in_English'], drop_first=True)], axis=1)
all_data = all_data.drop(labels=['homepage','status','tagline','in_collection','in_English'], axis=1)

all_data.columns.values
original_numerical_features = ['budget', 'popularity', 'runtime']
# Fill in movie runtime

train_data.loc[train_data['title'] == 'The Worst Christmas of My Life', 'runtime'] = 87.0

train_data.loc[train_data['title'] == 'А поутру они проснулись', 'runtime'] = 90.0

train_data.loc[train_data['title'] == '¿Quién mató a Bambi?', 'runtime'] = 86.0

train_data.loc[train_data['title'] == 'La peggior settimana della mia vita', 'runtime'] = 93.0

train_data.loc[train_data['title'] == 'Cry, Onion!', 'runtime'] = 92.0

train_data.loc[train_data['title'] == 'All at Once', 'runtime'] = 93.0

train_data.loc[train_data['title'] == 'Missing', 'runtime'] = 86.0

train_data.loc[train_data['title'] == 'Mechenosets', 'runtime'] = 108.0

train_data.loc[train_data['title'] == 'Hooked on the Game 2. The Next Level', 'runtime'] = 86.0

train_data.loc[train_data['title'] == 'My Old Classmate', 'runtime'] = 98.0

train_data.loc[train_data['title'] == 'Revelation', 'runtime'] = 111.0

train_data.loc[train_data['title'] == 'Tutto tutto niente niente', 'runtime'] = 96.0

train_data.loc[train_data['title'] == 'Королёв', 'runtime'] = 130.0

train_data.loc[train_data['title'] == 'Happy Weekend', 'runtime'] = 100.0 # missing, use mode



test_data.loc[test_data['title'] == 'Shikshanachya Aaicha Gho', 'runtime'] = 103.0

test_data.loc[test_data['title'] == 'Street Knight', 'runtime'] = 91.0

test_data.loc[test_data['title'] == 'Plus one', 'runtime'] = 98.0

test_data.loc[test_data['title'] == 'Glukhar v kino', 'runtime'] = 86.0

test_data.loc[test_data['title'] == 'Frau Müller muss weg!', 'runtime'] = 83.0

test_data.loc[test_data['title'] == 'Shabd', 'runtime'] = 140.0

test_data.loc[test_data['title'] == 'The Last Breath', 'runtime'] = 104.0

test_data.loc[test_data['title'] == 'Chaahat Ek Nasha...', 'runtime'] = 140.0

test_data.loc[test_data['title'] == 'El truco del manco', 'runtime'] = 100.0 # The runtime is missing from IMDB!

test_data.loc[test_data['title'] == 'La caliente niña Julietta', 'runtime'] = 93.0

test_data.loc[test_data['title'] == 'Pancho, el perro millonario', 'runtime'] = 91.0

test_data.loc[test_data['title'] == 'Nunca en horas de clase', 'runtime'] = 100.0

test_data.loc[test_data['title'] == 'Miesten välisiä keskusteluja', 'runtime'] = 90.0
# Clearn training set: I used the additional information from this kernel:

# https://www.kaggle.com/kamalchhirang/eda-feature-engineering-lgb-xgb-cat/notebook#Feature-Engineering-&-Prediction

train_data.loc[train_data['id'] == 16,'revenue'] = 192864          # Skinning

train_data.loc[train_data['id'] == 90,'budget'] = 30000000         # Sommersby          

train_data.loc[train_data['id'] == 118,'budget'] = 60000000        # Wild Hogs

train_data.loc[train_data['id'] == 149,'budget'] = 18000000        # Beethoven

train_data.loc[train_data['id'] == 313,'revenue'] = 12000000       # The Cookout 

train_data.loc[train_data['id'] == 451,'revenue'] = 12000000       # Chasing Liberty

train_data.loc[train_data['id'] == 464,'budget'] = 20000000        # Parenthood

train_data.loc[train_data['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II

train_data.loc[train_data['id'] == 513,'budget'] = 930000          # From Prada to Nada

train_data.loc[train_data['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol

train_data.loc[train_data['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip

train_data.loc[train_data['id'] == 850,'budget'] = 90000000        # Modern Times

train_data.loc[train_data['id'] == 1007,'budget'] = 2              # Zyzzyx Road 

train_data.loc[train_data['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman

train_data.loc[train_data['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   

train_data.loc[train_data['id'] == 1359,'budget'] = 10000000       # Stir Crazy 

train_data.loc[train_data['id'] == 1542,'budget'] = 1              # All at Once

train_data.loc[train_data['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II

train_data.loc[train_data['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp

train_data.loc[train_data['id'] == 1714,'budget'] = 46000000       # The Recruit

train_data.loc[train_data['id'] == 1721,'budget'] = 17500000       # Cocoon

train_data.loc[train_data['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed

train_data.loc[train_data['id'] == 1885,'budget'] = 12             # In the Cut

train_data.loc[train_data['id'] == 2091,'budget'] = 10             # Deadfall

train_data.loc[train_data['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget

train_data.loc[train_data['id'] == 2491,'budget'] = 6              # Never Talk to Strangers

train_data.loc[train_data['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus

train_data.loc[train_data['id'] == 2612,'budget'] = 15000000       # Field of Dreams

train_data.loc[train_data['id'] == 2696,'budget'] = 10000000       # Nurse 3-D

train_data.loc[train_data['id'] == 2801,'budget'] = 10000000       # Fracture

train_data.loc[train_data['id'] == 335,'budget'] = 2 

train_data.loc[train_data['id'] == 348,'budget'] = 12

train_data.loc[train_data['id'] == 470,'budget'] = 13000000 

train_data.loc[train_data['id'] == 513,'budget'] = 1100000

train_data.loc[train_data['id'] == 640,'budget'] = 6 

train_data.loc[train_data['id'] == 696,'budget'] = 1

train_data.loc[train_data['id'] == 797,'budget'] = 8000000 

train_data.loc[train_data['id'] == 850,'budget'] = 1500000

train_data.loc[train_data['id'] == 1199,'budget'] = 5 

train_data.loc[train_data['id'] == 1282,'budget'] = 9               # Death at a Funeral

train_data.loc[train_data['id'] == 1347,'budget'] = 1

train_data.loc[train_data['id'] == 1755,'budget'] = 2

train_data.loc[train_data['id'] == 1801,'budget'] = 5

train_data.loc[train_data['id'] == 1918,'budget'] = 592 

train_data.loc[train_data['id'] == 2033,'budget'] = 4

train_data.loc[train_data['id'] == 2118,'budget'] = 344 

train_data.loc[train_data['id'] == 2252,'budget'] = 130

train_data.loc[train_data['id'] == 2256,'budget'] = 1 

train_data.loc[train_data['id'] == 2696,'budget'] = 10000000
# Clean test set

test_data.loc[test_data['id'] == 6733,'budget'] = 5000000

test_data.loc[test_data['id'] == 3889,'budget'] = 15000000

test_data.loc[test_data['id'] == 6683,'budget'] = 50000000

test_data.loc[test_data['id'] == 5704,'budget'] = 4300000

test_data.loc[test_data['id'] == 6109,'budget'] = 281756

test_data.loc[test_data['id'] == 7242,'budget'] = 10000000

test_data.loc[test_data['id'] == 7021,'budget'] = 17540562       #  Two Is a Family

test_data.loc[test_data['id'] == 5591,'budget'] = 4000000        # The Orphanage

test_data.loc[test_data['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee

test_data.loc[test_data['id'] == 3033,'budget'] = 250 

test_data.loc[test_data['id'] == 3051,'budget'] = 50

test_data.loc[test_data['id'] == 3084,'budget'] = 337

test_data.loc[test_data['id'] == 3224,'budget'] = 4  

test_data.loc[test_data['id'] == 3594,'budget'] = 25  

test_data.loc[test_data['id'] == 3619,'budget'] = 500  

test_data.loc[test_data['id'] == 3831,'budget'] = 3  

test_data.loc[test_data['id'] == 3935,'budget'] = 500  

test_data.loc[test_data['id'] == 4049,'budget'] = 995946 

test_data.loc[test_data['id'] == 4424,'budget'] = 3  

test_data.loc[test_data['id'] == 4460,'budget'] = 8  

test_data.loc[test_data['id'] == 4555,'budget'] = 1200000 

test_data.loc[test_data['id'] == 4624,'budget'] = 30 

test_data.loc[test_data['id'] == 4645,'budget'] = 500 

test_data.loc[test_data['id'] == 4709,'budget'] = 450 

test_data.loc[test_data['id'] == 4839,'budget'] = 7

test_data.loc[test_data['id'] == 3125,'budget'] = 25 

test_data.loc[test_data['id'] == 3142,'budget'] = 1

test_data.loc[test_data['id'] == 3201,'budget'] = 450

test_data.loc[test_data['id'] == 3222,'budget'] = 6

test_data.loc[test_data['id'] == 3545,'budget'] = 38

test_data.loc[test_data['id'] == 3670,'budget'] = 18

test_data.loc[test_data['id'] == 3792,'budget'] = 19

test_data.loc[test_data['id'] == 3881,'budget'] = 7

test_data.loc[test_data['id'] == 3969,'budget'] = 400

test_data.loc[test_data['id'] == 4196,'budget'] = 6

test_data.loc[test_data['id'] == 4221,'budget'] = 11

test_data.loc[test_data['id'] == 4222,'budget'] = 500

test_data.loc[test_data['id'] == 4285,'budget'] = 11

test_data.loc[test_data['id'] == 4319,'budget'] = 1

test_data.loc[test_data['id'] == 4639,'budget'] = 10

test_data.loc[test_data['id'] == 4719,'budget'] = 45

test_data.loc[test_data['id'] == 4822,'budget'] = 22

test_data.loc[test_data['id'] == 4829,'budget'] = 20

test_data.loc[test_data['id'] == 4969,'budget'] = 20

test_data.loc[test_data['id'] == 5021,'budget'] = 40 

test_data.loc[test_data['id'] == 5035,'budget'] = 1 

test_data.loc[test_data['id'] == 5063,'budget'] = 14 

test_data.loc[test_data['id'] == 5119,'budget'] = 2 

test_data.loc[test_data['id'] == 5214,'budget'] = 30 

test_data.loc[test_data['id'] == 5221,'budget'] = 50 

test_data.loc[test_data['id'] == 4903,'budget'] = 15

test_data.loc[test_data['id'] == 4983,'budget'] = 3

test_data.loc[test_data['id'] == 5102,'budget'] = 28

test_data.loc[test_data['id'] == 5217,'budget'] = 75

test_data.loc[test_data['id'] == 5224,'budget'] = 3 

test_data.loc[test_data['id'] == 5469,'budget'] = 20 

test_data.loc[test_data['id'] == 5840,'budget'] = 1 

test_data.loc[test_data['id'] == 5960,'budget'] = 30

test_data.loc[test_data['id'] == 6506,'budget'] = 11 

test_data.loc[test_data['id'] == 6553,'budget'] = 280

test_data.loc[test_data['id'] == 6561,'budget'] = 7

test_data.loc[test_data['id'] == 6582,'budget'] = 218

test_data.loc[test_data['id'] == 6638,'budget'] = 5

test_data.loc[test_data['id'] == 6749,'budget'] = 8 

test_data.loc[test_data['id'] == 6759,'budget'] = 50 

test_data.loc[test_data['id'] == 6856,'budget'] = 10

test_data.loc[test_data['id'] == 6858,'budget'] =  100

test_data.loc[test_data['id'] == 6876,'budget'] =  250

test_data.loc[test_data['id'] == 6972,'budget'] = 1

test_data.loc[test_data['id'] == 7079,'budget'] = 8000000

test_data.loc[test_data['id'] == 7150,'budget'] = 118

test_data.loc[test_data['id'] == 6506,'budget'] = 118

test_data.loc[test_data['id'] == 7225,'budget'] = 6

test_data.loc[test_data['id'] == 7231,'budget'] = 85

test_data.loc[test_data['id'] == 5222,'budget'] = 5

test_data.loc[test_data['id'] == 5322,'budget'] = 90

test_data.loc[test_data['id'] == 5350,'budget'] = 70

test_data.loc[test_data['id'] == 5378,'budget'] = 10

test_data.loc[test_data['id'] == 5545,'budget'] = 80

test_data.loc[test_data['id'] == 5810,'budget'] = 8

test_data.loc[test_data['id'] == 5926,'budget'] = 300

test_data.loc[test_data['id'] == 5927,'budget'] = 4

test_data.loc[test_data['id'] == 5986,'budget'] = 1

test_data.loc[test_data['id'] == 6053,'budget'] = 20

test_data.loc[test_data['id'] == 6104,'budget'] = 1

test_data.loc[test_data['id'] == 6130,'budget'] = 30

test_data.loc[test_data['id'] == 6301,'budget'] = 150

test_data.loc[test_data['id'] == 6276,'budget'] = 100

test_data.loc[test_data['id'] == 6473,'budget'] = 100

test_data.loc[test_data['id'] == 6842,'budget'] = 30
# Combine the cleaned data into all_data

all_data[['runtime','budget','revenue']][:3000] = train_data[['runtime','budget','revenue']]

all_data[['runtime','budget']][3000:] = test_data[['runtime','budget']]
fig, ax = plt.subplots(figsize=(7,7))



sns.regplot(x=all_data['runtime'][:3000], y=all_data['revenue'][:3000], color='g', scatter_kws={'alpha': 0.5})

correlation = np.corrcoef(all_data['runtime'][:3000], all_data['revenue'][:3000])[0,1]

ax.set_title("Correlation {:.2f}".format(correlation))



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

colors = sns.color_palette(n_colors=2)

sns.set(color_codes=True, font_scale=1.4)



g = sns.distplot(train_data["revenue"], ax = axes[0])

axes[0].legend(["skewness: {:.2f}".format(train_data["revenue"].skew())])

plt.setp(g.get_legend().get_texts(), fontsize='16')



g = sns.distplot(np.log1p(train_data["revenue"]), ax = axes[1])

axes[1].legend(["skewness: {:.2f}".format(np.log1p(train_data["revenue"].skew()))])

plt.setp(g.get_legend().get_texts(), fontsize='16')



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

colors = sns.color_palette(n_colors=2)

sns.set(color_codes=True, font_scale=1.4)



g = sns.distplot(train_data["runtime"], ax = axes[0])

axes[0].legend(["skewness: {:.2f}".format(train_data["runtime"].skew())])

plt.setp(g.get_legend().get_texts(), fontsize='16')



g = sns.distplot(np.log1p(train_data["runtime"]), ax = axes[1])

axes[1].legend(["skewness: {:.2f}".format(np.log1p(train_data["runtime"].skew()))])

plt.setp(g.get_legend().get_texts(), fontsize='16')



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

colors = sns.color_palette(n_colors=2)

sns.set(color_codes=True, font_scale=1.4)



g = sns.distplot(train_data["budget"], ax = axes[0])

axes[0].legend(["skewness: {:.2f}".format(train_data["budget"].skew())])

plt.setp(g.get_legend().get_texts(), fontsize='16')



g = sns.distplot(np.log1p(train_data["budget"]), ax = axes[1])

axes[1].legend(["skewness: {:.2f}".format(np.log1p(train_data["budget"].skew()))])

plt.setp(g.get_legend().get_texts(), fontsize='16')



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

colors = sns.color_palette(n_colors=2)

sns.set(color_codes=True, font_scale=1.4)



g = sns.distplot(train_data["popularity"], ax = axes[0])

axes[0].legend(["skewness: {:.2f}".format(train_data["popularity"].skew())])

plt.setp(g.get_legend().get_texts(), fontsize='16')



g = sns.distplot(np.log1p(train_data["popularity"]), ax = axes[1])

axes[1].legend(["skewness: {:.2f}".format(np.log1p(train_data["popularity"].skew()))])

plt.setp(g.get_legend().get_texts(), fontsize='16')



plt.tight_layout()

plt.show()
# Apply log+1 transformations

all_data['revenue'][:3000] = np.log1p(all_data["revenue"][:3000])

all_data['runtime'] = np.log1p(all_data["runtime"])

all_data['budget'] = np.log1p(all_data["budget"])

all_data['popularity'] = np.log1p(all_data["popularity"])
(all_data['budget'] == 0).sum()
budget_train_data = all_data.loc[all_data['budget'] != 0]

budget_test_data = all_data.loc[all_data['budget'] == 0]
budget_train_X = budget_train_data.drop(labels=['budget', 'revenue'], axis=1)

budget_train_Y = budget_train_data['budget']

budget_test_X = budget_test_data.drop(labels=['budget', 'revenue'], axis=1)
lr = LinearRegression()

lr.fit(budget_train_X, budget_train_Y)
budget_Ypred = lr.predict(budget_test_X)
# Fill in the zero values with our prediction

all_data.loc[all_data['budget'] == 0, 'budget'] = budget_Ypred
Y = all_data['revenue']

final_data = all_data.drop(labels=['revenue'], axis=1)
ss = StandardScaler()

rs = RobustScaler()

# final_data = ss.fit_transform(final_data)

final_data = rs.fit_transform(final_data)

# final_data = final_data.values
final_data.shape
Ytrain = Y[:train_size]

Xtrain = final_data[:train_size]

Xtest = final_data[train_size:]