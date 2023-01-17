# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams



from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):

        display(df)
data = pd.read_csv("../input/zomato.csv")

display_all(data.head()) 
display_all(data.describe(include='all'))
del data['url']

del data['address']

del data['phone']

del data['location']

del data['book_table']



data.rename(columns={'approx_cost(for two people)':'cost_for_two','listed_in(city)':'locality', 'listed_in(type)':'category'},inplace=True)
((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)



data.rate = data.rate.replace("NEW",np.nan)

data.dropna(how='any',inplace=True)

((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)
# Convert rate and cost_for_two to float

data.rate = data.rate.astype(str)

data.rate = data.rate.apply(lambda x:x.replace('/5',''))

data.rate = data.rate.astype(float)



data.cost_for_two = data.cost_for_two.astype(str)

data.cost_for_two = data.cost_for_two.apply(lambda x:x.replace(',',''))

data.cost_for_two = data.cost_for_two.astype(float)



data.loc[data['locality'].str.contains('Koramangala'), 'locality'] = 'Koramangala'

data.loc[data['locality'].str.contains('Kalyan Nagar'), 'locality'] = 'Kammanahalli'





# Some restaurants on brigade road are wrongly marked as category=Pubs and bars, correcting the category to desserts.

# There may be many such records in the dataset but for now lets clean up only these

data.loc[(~(data['rest_type'].str.contains('Pub|Bar')) & (data['category'].str.contains('Pub')) & (data['locality']=='Brigade Road'))

          , 'category'] = 'Desserts'



# standardize the data in rest_type

vals_to_replace = {'Bar, Casual Dining':'Casual Dining, Bar'

                   ,'Cafe, Casual Dining':'Casual Dining, Cafe'

                   ,'Pub, Casual Dining':'Casual Dining, Bar'

                   ,'Casual Dining, Pub':'Casual Dining, Bar'

                   ,'Pub':'Bar'

                   ,'Pub, Bar':'Bar'

                   ,'Bar, Pub':'Bar'

                   ,'Microbrewery, Bar':'Microbrewery'

                   ,'Microbrewery, Pub':'Microbrewery'

                   ,'Pub, Microbrewery':'Microbrewery'

                   ,'Microbrewery, Casual Dining':'Casual Dining, Microbrewery'

                   ,'Lounge, Casual Dining':'Casual Dining, Lounge'

                   ,'Sweet Shop, Quick Bites':'Quick Bites, Sweet Shop'

                   ,'Cafe, Quick Bites':'Quick Bites, Cafe'

                   ,'Beverage Shop, Quick Bites':'Quick Bites, Beverage Shop'

                   ,'Bakery, Quick Bites':'Quick Bites, Bakery'

                   ,'Desert Parlor, Quick Bites':'Quick Bites, Desert Parlor'

                   ,'Food Court, Quick Bites':'Quick Bites, Food Court'

                   ,'Bar, Quick Bites':'Quick Bites, Bar'

                   ,'Cafe, Lounge':'Lounge, Cafe'

                   ,'Cafe, Bakery':'Bakery, Cafe'

                   ,'Cafe, Dessert Parlor':'Dessert Parlor, Cafe'

                   ,'Cafe, Beverage Shop':'Beverage Shop, Cafe'

                   ,'Bar, Lounge':'Lounge, Bar'

                   ,'Fine Dining, Lounge':'Lounge, Fine Dining'

                   ,'Microbrewery, Lounge':'Lounge, Microbrewery'}



data['rest_type'] = data['rest_type'].replace(vals_to_replace)

data.rest_type.sort_values().unique()
data_unique = data.drop_duplicates(subset=['locality', 'name'], keep='first')



data_unique.reset_index(inplace=True)

os.makedirs('tmp', exist_ok=True)

data_unique.to_feather('tmp/zomato-unique')
data_unique = pd.read_feather('tmp/zomato-unique')

data_unique.head()
fig, axes = plt.subplots(3,1,sharex=True, figsize=(20,14))

data_unique['cost_for_two'].plot.density(ax=axes[0], color = "#3399cc")

sns.boxplot(ax=axes[1], x=data_unique['cost_for_two'], color = "#3399cc")

data_unique['cost_for_two'].plot.hist(ax=axes[2], bins=50, color = "#3399cc")



data_unique['cost_for_two'].describe()
fig, ax = plt.subplots(figsize=(15,7))

ax = sns.countplot(x="locality", data=data_unique, order=data_unique['locality'].value_counts().index, color = "#3399cc")

ax.set_title("Count Per Locality")

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.show()
data_unique['rate'].plot.hist(bins=50, color = "#3399cc")
data_select = data[

                   (data['cost_for_two'] >= 400) & (data['cost_for_two'] <= 800)

#                           & (data_top_all['category'].str.contains('Buffet'))

#                           & (data_top_all['rest_type'].str.contains('Food Truck'))

#                           & (data_top_all['cuisines'].str.contains('North Indian'))

#                           & (data_top_all['online_order'].str.contains('Yes'))

                 ]



data_top = data_select.sort_values(['locality', 'votes', 'rate', ],ascending=False)

data_top.drop_duplicates(subset=['locality', 'name'], keep='first', inplace=True)

data_top10 = data_top.groupby('locality').head(10)

data_top10[["locality", "name", "rest_type", "category", "cost_for_two", "votes", "rate"]]
pd.concat(g for _, g in data_top10[["name", "locality", "cost_for_two", "votes", "rate"]].groupby("name") if len(g) > 1)
pd.crosstab(data_top10.name, data_top10.locality, margins=True)
data_bot = data_select.sort_values(['locality', 'votes', 'rate', ],ascending=[False,False,True])

data_bot = data_bot[data_bot['rate'] < 2.5]

data_bot.drop_duplicates(subset=['locality', 'name'], keep='first', inplace=True)

data_bot10 = data_bot.groupby('locality').head(10)

data_bot10[["locality", "name","rest_type", "category", "cost_for_two", "votes", "rate"]]

pd.concat(g for _, g in data_bot10[["name", "locality", "votes", "rate"]].groupby("name") if len(g) > 1)
pd.merge(data_top10, data_bot10, on='name')
data_top_all = data_select.sort_values(['votes', 'rate', ],ascending=False)             

data_top_all.drop_duplicates(subset=['name'], keep='first', inplace=True)

data_top_all[["name", "cuisines", "locality", "cost_for_two", "votes", "rate"]].head(10)

data_bot_all = data_select.sort_values(['votes', 'rate', ],ascending=[False,True])

data_bot_all = data_bot_all[data_bot_all['rate'] < 2.5]

data_bot_all.drop_duplicates(subset=['name'], keep='first', inplace=True)

data_bot_all[["name", "cuisines", "locality", "cost_for_two", "votes", "rate"]].head(10) 

# Cost for two vs locality

fig, ax = plt.subplots(figsize=(15,5))

ax = sns.barplot(ax=ax, x='locality', y='cost_for_two', data=data_unique, color = "#3399cc")

ax.set_title("Cost_for_two Per Locality")

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.show()
# Cost for two vs restaurant type

fig, ax = plt.subplots(figsize=(20,10))

ax = sns.barplot(ax=ax, x='rest_type', y='cost_for_two', data=data_unique, color = "#3399cc")

ax.set_title("Cost_for_two Per Restaurant Type")

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.show()
data_unique.pivot_table('cost_for_two', columns=['category'], index='locality' ).plot(kind='bar',figsize=(20,10))
fig, ax = plt.subplots(figsize=(15,5))

ax = sns.barplot(ax=ax, x='rate', y='cost_for_two', data=data_unique, color = "#3399cc")

ax.set_title("Cost_for_two vs rating")

plt.show()
fig, ax = plt.subplots(figsize=(10,5))

ax = sns.barplot(ax=ax, x='online_order', y='cost_for_two', data=data_unique, color = "#3399cc")

ax.set_title("Cost_for_two vs online_order (Yes/No)")

plt.show()
data_unique = pd.read_feather('tmp/zomato-unique')

display_all(data_unique.head())
# Select columns and set the rows in random order before feeding to the model



data_unique = data.drop_duplicates(subset=['locality', 'name'], keep='first')

data_unique = data_unique[['locality', 'rest_type', 'category', 'rate', 'votes', 'online_order', 'cost_for_two']]

data_unique = data_unique.sample(frac=1).reset_index(drop=True)
sns.pairplot(data_unique, diag_kind='kde', plot_kws={'alpha':0.2})
#1 - one hot encode string data

data_unique = pd.get_dummies(data_unique)

display_all(data_unique.head(10))
X = data_unique.drop(['cost_for_two'], axis=1)

y = data_unique.cost_for_two
# Split into train and test datasets

def split_vals(a,n): return a[:n].copy(), a[n:].copy()



n_valid = 2500

n_trn = len(data_unique)-n_valid

X_train, X_valid = split_vals(X,n_trn)

y_train, y_valid = split_vals(y,n_trn)



X_train.shape, y_train.shape, X_valid.shape, y_valid.shape

# Random Forest with default number of estimators

m = RandomForestRegressor(random_state=0, n_jobs=-1)

m.fit(X_train, y_train)

res = [m.score(X_train,y_train), m.score(X_valid,y_valid)]

print(res)
preds = np.stack([tree.predict(X_valid) for tree in m.estimators_])

preds.shape, preds[:,0], np.mean(preds[:,0]), y_valid
m = RandomForestRegressor(random_state=0, n_estimators=1, max_depth=5, n_jobs=-1)

m.fit(X_train, y_train)

res = [m.score(X_train,y_train), m.score(X_valid,y_valid)]

print(res)
from sklearn.datasets import *

from sklearn import tree

from sklearn.tree import export_graphviz

from IPython.display import Image  

import graphviz

import pydot





tree = m.estimators_[0]

# Export the image to a dot file

export_graphviz(tree, out_file = 'tree.dot', feature_names = X_train.columns, rounded = True, precision = 1)

# Use dot file to create a graph

(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Show graph

Image(graph.create_png())

# Create PDF

#graph.write_pdf("iris.pdf")

# Create PNG

#graph.write_png("iris.png")
m = RandomForestRegressor(random_state=0, n_estimators=1, n_jobs=-1)

m.fit(X_train, y_train)

res = [m.score(X_train,y_train), m.score(X_valid,y_valid)]

print(res)
# Try different numbers of n_estimators 

estimators = np.arange(10, 200, 10)

scores = []

for n in estimators:

    m.set_params(n_estimators=n)

    m.fit(X_train, y_train)

    scores.append(m.score(X_valid, y_valid))

plt.title("Effect of n_estimators")

plt.xlabel("n_estimator")

plt.ylabel("score")

plt.plot(estimators, scores)
# Random Forest with 150 estimators

m = RandomForestRegressor(random_state=0, n_estimators=150,n_jobs=-1)

m.fit(X_train, y_train)

res = [m.score(X_train,y_train), m.score(X_valid,y_valid)]

print(res)
# Calculate feature importances

importances = m.feature_importances_



# Sort feature importances in descending order

indices = np.argsort(importances)[::-1]



# Rearrange feature names so they match the sorted feature importances

names = [X_train.columns[i] for i in indices]



plt.figure(figsize=(15,15))

plt.title("Feature Importance")

plt.barh(range(X.shape[1]),importances[indices], color = "#3399cc")

plt.yticks(range(X.shape[1]), names)



plt.show()
names