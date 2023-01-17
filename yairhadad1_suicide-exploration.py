import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

dataset = pd.read_csv('../input/who_suicide_statistics.csv')

# Change country into category

dataset['country'].astype('category')



# Change sex to isMale and isFemale and remove sex column

dataset['isMale'] = dataset['sex'] == 'male'

dataset['isFemale'] = dataset['sex'] == 'female'

dataset.drop('sex', axis = 1, inplace = True)



# Drop rows with no suicides numbers or population. (nan)

dataset.dropna(subset=['suicides_no'], inplace = True)

dataset.dropna(subset=['population'], inplace = True)



dataset['age'].astype('category', inplace = True)



dataset['population_suicide_ratio'] = dataset['suicides_no']/dataset['population']* 100



age_int = {'5-14 years':0,

            '15-24 years':1,

            '25-34 years':2,

            '35-54 years':3,

            '55-74 years':4,

            '75+ years':5}



dataset['age_int_type'] = dataset['age'].map(age_int)



dataset.head()
sns.heatmap(dataset.corr(),linewidths=.5,cmap="YlGnBu")

plt.show()
minAgeGroupby = dataset.groupby('country')['suicides_no'].sum().sort_values(ascending = False)

minAgeGroupby.head(10).plot.bar(title = "Country With must suicides count", rot=100,figsize=(10, 5))

plt.ylabel('suicides count')

plt.show()

minAgeGroupby = dataset.groupby('country')['population_suicide_ratio'].mean().sort_values(ascending = False)

minAgeGroupby.head(10).plot.bar(title = "Country With must suicides ratio in population", rot=100,figsize=(10, 5))

plt.ylabel('population suicide ratio mean')

plt.show()
maxSuicides = dataset.loc[dataset['suicides_no'].idxmax()]

print("The most suicides was in",maxSuicides.country)

print("In year",maxSuicides.year)

print("Percent of population was suicides:",maxSuicides.population_suicide_ratio*100)

print()



print("Probably because of the black tuesday in russian, When the ruble collapsed dramatically against the dollar")

print("for more info look at: http://www.pravdareport.com/business/122425-black_tuesday")
yearGroupby = dataset.groupby('year')

yearGroupby['suicides_no'].sum().plot(title = "Suicides by year",grid=True)

plt.show()



yearGroupby['population_suicide_ratio'].mean().plot(title = "Suicides population ratio by year",grid=True)

plt.show()



# year with the most suicides in the wolrd

suicidesMaxYear = yearGroupby['suicides_no'].sum().idxmax()

suicidesMaxYearCount = yearGroupby['suicides_no'].sum().max()

print("year with the most suicides in the world",suicidesMaxYear,"with:",suicidesMaxYearCount,"suicides")

suicidesMinYear = yearGroupby['suicides_no'].sum().idxmin()

suicidesMinYearCount = yearGroupby['suicides_no'].sum().min()

print("year with the less suicides in the world",suicidesMinYear,"with:",suicidesMinYearCount,"suicides")



dataset['year'].plot.box()

plt.show()
dic = {0:'5-14 years',

      1:'15-24 years',

      2:'25-34 years',

      3:'35-54 years',

      4:'55-74 years',

      5:'75+ years'}



suicide = dataset.groupby('age_int_type')[['suicides_no']]

plt.figure(figsize=(10,5))

suicide.sum().plot(color = 'blue',figsize=(10, 5), grid = True)

sns.barplot(x=suicide.sum().index.map(dic.get),y=suicide.sum().suicides_no, alpha=0.7)

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2)

fig.tight_layout()



dataset.groupby('isMale').sum()['suicides_no'].plot.bar(ax=axes[0],figsize=(13, 3),rot=100,title = "Male suicides")

dataset.groupby('isFemale').sum()['suicides_no'].plot.bar(ax=axes[1],figsize=(13, 3),rot=100,title = "Female suicides")

plt.show()