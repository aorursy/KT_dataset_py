# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#reading values from given dataset

master = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv', encoding="windows-1252")

# Any results you write to the current directory are saved as output.
master.head()#examining the data
country_list = master.country.unique()#looking for country names

#here we'll examine populations per country

pop = list()

for i in country_list:

    x = master[master.country == i]

    country_pop = sum(x.population)

    pop.append(country_pop)



#making a data frame for plotting

data2= pd.DataFrame({'country_list': country_list,'population':pop})

new_index = (data2['population'].sort_values(ascending=False)).index.values

sorted_data2 = data2.reindex(new_index)



#examining the population amounts of country

plt.figure(figsize=(15,10))

sns.barplot(x = sorted_data2.country_list, y = sorted_data2.population)

plt.xticks(rotation= 90)

plt.xlabel('Countries')

plt.ylabel('Population of given country')

plt.title('Population per country')

plt.show()

#we can observe that Russia has more sum population, that means it has more increase in Russia
#we examine suicide rates per 100k for every country.

country_list = master.country.unique()

sui_pop = list()

for i in country_list:

    x = master[master.country == i]

    country_pop = sum(x["suicides/100k pop"])

    sui_pop.append(country_pop)





data = pd.DataFrame({'country_list': country_list,'suicide_rates':sui_pop})

new_index = (data['suicide_rates'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)





plt.figure(figsize=(15,10))

sns.barplot(x = sorted_data.country_list, y = sorted_data.suicide_rates)

plt.xticks(rotation= 90)

plt.xlabel('Countries')

plt.ylabel('Suicide rates per 100k')

plt.title('Suicide rates over 100k per country')

plt.show()

#we can observe that Russia has the biggest suicide rates
#we examine the suicide rates in terms of sexes

sex_list = list(master.sex.unique())

sui_pop = list()

for i in sex_list:

    x = master[master.sex == i]

    country_pop = sum(x["suicides/100k pop"])

    sui_pop.append(country_pop)





data3 = pd.DataFrame({'sex': sex_list,'suicide_rates':sui_pop})

new_index = (data3['suicide_rates'].sort_values(ascending=False)).index.values

sorted_data3 = data3.reindex(new_index)





plt.figure(figsize=(15,10))

sns.barplot(x = sorted_data3.suicide_rates, y = sorted_data3.sex)

plt.xticks(rotation= 90)

plt.xlabel('Sexes')

plt.ylabel('Suicide rates per 100k')

plt.title('Suicide rates of sexes')

plt.show()

#We observe the sex shares of suicide rates, it is obvious that men suicide way more than women
#here we observe suicide rates and suicide amounts

sui_amo = list()

for i in country_list:

    x = master[master.country == i]

    country_pop = sum(x.suicides_no)

    sui_amo.append(country_pop)



data4 = pd.DataFrame({'country_list': country_list, 'suicide_amount': anan})

new_index = (data4.suicide_amount.sort_values(ascending=False)).index.values

sorted_data4 = data4.reindex(new_index)



sorted_data.suicide_rates = sorted_data.suicide_rates/max(sorted_data.suicide_rates)

sorted_data4.suicide_amount = sorted_data4.suicide_amount/max(sorted_data4.suicide_amount)





f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='country_list',y='suicide_rates',data=sorted_data,color='red',alpha = 0.8)

sns.pointplot(x='country_list', y='suicide_amount',data=sorted_data4,color='lime',alpha = 0.8)

plt.text(40,0.6,'suicide rates',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'suicide amounts',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('Countries',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('Suicide rates/100k  VS  suicide numbers',fontsize = 20,color='blue')

plt.xticks(rotation= 90)

plt.grid()

plt.show()

#suicide rates are directly proportional with suicide amounts.
#practicing joint plot by comparing the previous ones,

# Visualization of suicide rates vs suicide amounts of each countries with different style of seaborn code

# joint kernel density

# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables

# Show the joint distribution using kernel density estimation 

data = pd.concat([sorted_data,sorted_data4.suicide_amount],axis=1)

g = sns.jointplot(data.suicide_amount, data.suicide_rates, kind="kde", size=7)

plt.savefig('graph.png')

plt.show()
g = sns.jointplot("suicide_amount", "suicide_rates", data=data,size=5, ratio=3, color="green")
#Suicide rates according to age intervals

labels = master.age.value_counts().index

colors = ['pink','blue', 'red', 'green', 'yellow', 'orange']

explode = [0,0,0,0,0,0]

sizes = master.age.value_counts().values



# visual

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Killed People According to thie ages',color = 'blue',fontsize = 15)

plt.show()
# Visualization of suicide rates vs suicide amount of each country with different style of seaborn code

# lmplot 

# Show the results of a linear regression within each dataset

sns.lmplot(x="suicide_rates", y="suicide_amount", data=data)

plt.show()

#it shows a clear direct proportionality
# Visualization of suicide rates vs suicide amount of each state with different style of seaborn code

# cubehelix plot

sns.kdeplot(data.suicide_rates, data.suicide_amount, shade=False, cut=1)

plt.show()

#it is highly concentrated on a lower level
# Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette

pal = sns.cubehelix_palette(2, rot=1, dark=.3)

sns.violinplot(data=data, palette=pal, inner="points")

plt.show()
f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data.corr(), annot=True, linewidths=5, fmt= '.1f')

plt.show() 
#we can take imoressive results from there

plt.subplots(figsize=(25, 25))

sns.boxplot(x="country", y="suicides/100k pop", hue="sex", data=master, palette="PRGn")

plt.xticks(rotation = 90)

plt.show()
sns.swarmplot(x="country", y="suicides/100k pop",hue="sex", data=master.head(2000))

plt.xticks(rotation = 90)

plt.show()
f = plt.subplots(figsize = (15,15))

plt.title("population",color = 'red',fontsize=17)

sns.countplot(master.population.head(200))

plt.xticks(rotation = 90)

plt.show()

#this graph is made for just to show how a count plot can be done, these data are very constant, count plots would be useless.
