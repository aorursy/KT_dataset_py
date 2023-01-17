import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns



from pandas.api.types import is_numeric_dtype

from IPython.display import display



from collections import OrderedDict



pd.options.display.max_rows = None

pd.options.display.max_columns = None

plt.style.use('ggplot')



%matplotlib inline



data = pd.read_csv('../input/responses.csv', dtype={col_name: 'category' for col_name in [

    'Smoking', 'Alcohol', 'Punctuality', 'Lying', 'Internet usage', 'Gender', 'Left - right handed',

    'Education', 'Only child', 'Village - town', 'House - block of flats'

]})



data['BMI'] = data['Weight']/(data['Height'] * 0.01)**2



data.head(2)
nulls_by_obs = data.isnull().sum(axis=1)

nulls_by_obs = nulls_by_obs[nulls_by_obs > 0].sort_values(ascending=False)



missed_values = OrderedDict()



for i in nulls_by_obs.index:

    missed_values[i] = data.columns[data.loc[i].isnull()].tolist()

# missed_values



nulls_by_vars = data.isnull().sum().sort_values(ascending=False)

_ = nulls_by_vars.plot(kind='bar', figsize=(25, 5))
RdGr_cmap = sns.diverging_palette(10, 133, n=5, as_cmap=True)
music = data.loc[:, 'Music':'Opera'].dropna()



melted = pd.melt(music)

music_two_way_table = pd.crosstab(melted['variable'], melted['value'])

music_percentage = music_two_way_table.divide(music_two_way_table.sum(axis=1), axis=0)

_ = music_percentage.plot.barh(stacked=True, figsize=(10,10), colormap=RdGr_cmap)

_ = _.set_title('Music', fontsize=20)
movies = data.loc[:, 'Movies':'Action'].dropna()



melted = pd.melt(movies)

movie_two_way_table = pd.crosstab(melted['variable'], melted['value'])

movie_percentage = movie_two_way_table.divide(movie_two_way_table.sum(axis=1), axis=0)

_ = movie_percentage.plot.barh(stacked=True, figsize=(10,7), colormap=RdGr_cmap)

_ = _.set_title('Movies', fontsize=20)
hobbies = data.loc[:, 'History': 'Pets']



melted = pd.melt(hobbies)

hobby_two_way_table = pd.crosstab(melted['variable'], melted['value'])

hobby_percentage = hobby_two_way_table.divide(hobby_two_way_table.sum(axis=1), axis=0)

_ = hobby_percentage.plot.barh(stacked=True, figsize=(10,15), colormap=RdGr_cmap)

_ = _.set_title('Hobbies', fontsize=20)
phobias = data.loc[:, 'Flying': 'Fear of public speaking']



melted = pd.melt(phobias)

phobia_two_way_table = pd.crosstab(melted['variable'], melted['value'])

phobia_percentage = phobia_two_way_table.divide(phobia_two_way_table.sum(axis=1), axis=0)

_ = phobia_percentage.plot.barh(stacked=True, figsize=(10,5), colormap=RdGr_cmap)

_ = _.set_title('Phobias', fontsize=20)
spending = data.loc[:, 'Finances': 'Spending on healthy eating']



melted = pd.melt(spending)

spending_two_way_table = pd.crosstab(melted['variable'], melted['value'])

spending_percentage = spending_two_way_table.divide(spending_two_way_table.sum(axis=1), axis=0)

_ = spending_percentage.plot.barh(stacked=True, figsize=(10,5), colormap=RdGr_cmap)

_ = _.set_title('Spending preferances', fontsize=20)
fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(10, 7))



_ = sns.countplot(data=data, y='Smoking', ax=ax[0]).set_ylabel('Smoking', fontsize=15)

_ = sns.countplot(data=data, y='Alcohol', ax=ax[1]).set_ylabel('Alcohol', fontsize=15)

_ = sns.countplot(data=data, y='Healthy eating', ax=ax[2]).set_ylabel('Healthy lifestyle', fontsize=15)
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 5))



_ = sns.distplot(

    data['Age'].dropna(),

    bins=np.arange(data['Age'].min() - 0.5, data['Age'].max() + 0.5, 1),

    ax=ax[0])

_ = sns.boxplot(data['Age'].dropna())
plt.figure(figsize=(15, 1))

_ = sns.boxplot(data['Weight'].dropna())



plt.figure(figsize=(15, 1))

_ = sns.boxplot(data['Height'].dropna())



plt.figure(figsize=(15, 1))

_ = sns.boxplot(data['BMI'].dropna()).set_xlim((10, 40))
data['underweight'] = (data['BMI'] < 18.5).astype(np.int)

data['normal weight'] = ((18.5 <= data['BMI']) & (data['BMI'] <= 25)).astype(np.int)

data['overweight'] = (data['BMI'] > 25).astype(np.int)
_ = data[['underweight', 'normal weight', 'overweight']].sum().divide(len(data)).plot.bar()
fig = plt.figure(figsize=(15, 7))

gs = gridspec.GridSpec(2, 2, width_ratios=[4,1])



_ = sns.countplot(

    data['Number of siblings'].dropna(),

    ax=plt.subplot(gs[0,0]))

_ = sns.boxplot(

    data['Number of siblings'].dropna(),

    ax=plt.subplot(gs[1,0]))

_ = sns.countplot(x='Only child', data=data, ax=plt.subplot(gs[:, 1]))
i_thought_that_i_am_only_child = np.sum(data["Only child"].dropna() == "yes")

i_do_not_have_any_siblings = np.sum(data["Number of siblings"].dropna() == 0)



print(f'How many observations do not have any siblings: {i_do_not_have_any_siblings}')

print(f'How many observations are only child: {i_thought_that_i_am_only_child}')

print(f'How many observations live in different families with their siblings: {i_thought_that_i_am_only_child - i_do_not_have_any_siblings}')
_ = sns.countplot(y='Gender', data=data)
_ = sns.countplot(y='Left - right handed', data=data)
_ = sns.countplot(y='Education', data=data)
f, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 5))



_ = sns.countplot(y='Village - town', data=data, ax=ax[0])

_ = sns.countplot(y='House - block of flats', data=data, ax=ax[1])
start, end = np.argwhere(data.columns.isin(['Daily events', 'Internet usage'])).flatten()

catcols = np.argwhere(data.columns.isin(['Punctuality', 'Lying', 'Internet usage'])).flatten()



cols_with_categorical_answers = data.iloc[:, catcols]

cols_with_grades = data.iloc[:, list(set(range(start, end + 1)) - set(catcols))]
melted = pd.melt(cols_with_grades)

two_way_table = pd.crosstab(melted['variable'], melted['value'])

percentage = two_way_table.divide(two_way_table.sum(axis=1), axis=0)

_ = percentage.plot.barh(stacked=True, figsize=(10,20), colormap=RdGr_cmap)

_ = _.set_title('Personality traits, views on life and opinions', fontsize=20)
fig, ax = plt.subplots(nrows=len(catcols), sharex=True, figsize=(10, 7))

for i, col_i in enumerate(catcols):

    _ = sns.countplot(y=data.columns[col_i], data=data, ax=ax[i])
fig, axis = plt.subplots(ncols=2, sharey=True, figsize=(10,5))

fig.suptitle('Phobias', fontsize=21)



for i, gender in enumerate(data['Gender'].cat.categories):

    phobias = data.query(f'Gender == "{gender}"').loc[:, 'Flying': 'Fear of public speaking']



    melted = pd.melt(phobias)

    phobia_two_way_table = pd.crosstab(melted['variable'], melted['value'])

    phobia_percentage = phobia_two_way_table.divide(phobia_two_way_table.sum(axis=1), axis=0)

    _ = phobia_percentage.plot.barh(stacked=True, colormap=RdGr_cmap, ax=axis[i])

    _ = _.set_title(gender, fontsize=20)
fig, axis = plt.subplots(ncols=2, sharey=True, figsize=(10,5))

fig.suptitle('Money spending', fontsize=21)



for i, gender in enumerate(data['Gender'].cat.categories):

    spending = data.query(f'Gender == "{gender}"').loc[:, 'Finances': 'Spending on healthy eating']

    

    melted = pd.melt(spending)

    spending_two_way_table = pd.crosstab(melted['variable'], melted['value'])

    spending_percentage = spending_two_way_table.divide(spending_two_way_table.sum(axis=1), axis=0)

    _ = spending_percentage.plot.barh(stacked=True, colormap=RdGr_cmap, ax=axis[i])

    _ = _.set_title(gender, fontsize=20)
fig, axis = plt.subplots(ncols=2, sharey=True, figsize=(10,10))

fig.suptitle('Hobbies', fontsize=21)



for i, gender in enumerate(data['Gender'].cat.categories):

    hobbies = data.query(f'Gender == "{gender}"').loc[:, 'History': 'Pets']



    melted = pd.melt(hobbies)

    hobby_two_way_table = pd.crosstab(melted['variable'], melted['value'])

    hobby_percentage = hobby_two_way_table.divide(hobby_two_way_table.sum(axis=1), axis=0)

    _ = hobby_percentage.plot.barh(stacked=True, colormap=RdGr_cmap, ax=axis[i])

    _ = _.set_title(gender, fontsize=20)
two_way_table = pd.crosstab(data['Gender'], data['Punctuality'])

percentage = two_way_table.divide(two_way_table.sum(axis=1), axis=0)

_ = percentage.plot.barh(figsize=(7,2))
two_way_table = pd.crosstab(data['Gender'], data['Lying'])

percentage = two_way_table.divide(two_way_table.sum(axis=1), axis=0)

_ = percentage.plot.barh(figsize=(7,2))
two_way_table = pd.crosstab(data['Gender'], data['Alcohol'])

percentage = two_way_table.divide(two_way_table.sum(axis=1), axis=0)

_ = percentage.plot.barh(figsize=(7,2))
two_way_table = pd.crosstab(data['Gender'], data['Education'])

percentage = two_way_table.divide(two_way_table.sum(axis=1), axis=0)

_ = percentage.plot.barh(figsize=(7,3))
two_way_table = pd.crosstab(data['Gender'], data['Left - right handed'])

percentage = two_way_table.divide(two_way_table.sum(axis=1), axis=0)

_ = percentage.plot.barh(figsize=(7,2))
f = plt.figure(figsize=(10,7))



ax_male = sns.regplot(

    data=data.query('Gender == "male"'), x='Height', y='Weight', fit_reg=True,

    color='b', marker='+')



ax_female = sns.regplot(

    data=data.query('Gender == "female"'), x='Height', y='Weight', fit_reg=True,

    color='r', marker='o')