# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#importing all important libraries

%matplotlib inline

import pandas as pd

from matplotlib import pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

import numpy as np
#Reading in data and looking at the number of rows and columns using shape 

cities = pd.read_csv('../input/cities_r2.csv')

print (cities.shape)
cities.describe()
#Grouping data

def f(x):

     return pd.Series(dict(number_of_city =x['name_of_city'].count(), 

                           pop_total= x['population_total'].sum(), pop_male=x['population_male'].sum(),

                           pop_female = x['population_female'].sum()))
#Grouping data by state name and population

state = cities.groupby(['state_name']).apply(f)
#Highest population by state

state_bar = state[['pop_male','pop_female']]

sorted_df= state_bar.sort_values(by=['pop_male'], ascending=False)

sorted_df.plot(kind='bar',stacked=True, figsize=(15,8))

plt.xlabel("States")

plt.ylabel('Population in millions')

plt.title('Population of Men/Wemen by States')
def f(x):

     return pd.Series(dict( pop_total= x['population_total'].sum(), pop_male=x['population_male'].sum(),

                           pop_female = x['population_female'].sum()))
# Top 10 cities by population

state_city = cities.groupby(['name_of_city']).apply(f)

top10_cities = state_city.sort_values(by=['pop_total'], ascending = False).head(10)
#top cities by populationtop ith male and female breakup

topcities_bypopulation = top10_cities[['pop_male','pop_female']]

topcities_bypopulation.plot(kind='bar',stacked=True, figsize=(15,8))

plt.xlabel("States")

plt.ylabel('Population in Millions')

plt.title('Top 10 cities of India by Population')
def f(x):

     return pd.Series(dict( pop_total_child= x['0-6_population_total'].sum(), 

                           pop_male_child=x['0-6_population_male'].sum(),

                           pop_female_child = x['0-6_population_female'].sum()))
#top cities by child population

top_city_bychild = cities.groupby(['name_of_city']).apply(f)

top_city_bychild = top_city_bychild.sort_values(by=['pop_total_child'], ascending = False).head(10)
#top cities by child population

topcities_by_child_population = top_city_bychild[['pop_male_child','pop_female_child']]

topcities_by_child_population.plot(kind='bar',stacked=True, figsize=(15,8))

plt.xlabel("States")

plt.ylabel('Population in Millions')

plt.title('Top 10 cities of India by Child Population')
def f(x):

     return pd.Series(dict( sex_ratio= x['sex_ratio'].mean(),

                           child_sex_ratio = x['child_sex_ratio'].mean()))
## top 10 state by sex ratio

sexratio = cities.groupby(['state_name']).apply(f)

sexratio_bystate = sexratio.sort_values(by=['sex_ratio'], ascending = False).head(10)

sexratio_bystate['sex_ratio']
# top 10 state by child sex ratio

sexratio_bystate_child = sexratio.sort_values(by=['child_sex_ratio'], ascending = False)

sexratio_bystate_child['child_sex_ratio'].head(10)
# Aggregating literacy rate by States

literacy_rate = cities[["state_name","effective_literacy_rate_total","effective_literacy_rate_male",

                      "effective_literacy_rate_female"]].groupby("state_name").agg({"effective_literacy_rate_total":np.average,

                                                                                    "effective_literacy_rate_male":np.average,

                                                                                    "effective_literacy_rate_female":np.average})

sorted_literacy_rate = literacy_rate.sort_values(by=['effective_literacy_rate_total'], ascending=False)
#plotting aggregate literacy rate for overall population along with male and female

sorted_literacy_rate[['effective_literacy_rate_male','effective_literacy_rate_female']].plot(kind='bar', 

                                                                                            figsize=(20,9),

                                                                                            alpha = 1)

sorted_literacy_rate['effective_literacy_rate_total'].plot(kind='line',color = 'orange',linewidth=2.0,use_index = True)

plt.xticks(range(len(sorted_literacy_rate)+1),rotation = 90)

plt.xlabel("States")

plt.ylabel('Literacy Rate as Aggregate')

plt.title('Top states as per Literacy')
state_graduate  = cities[["state_name","total_graduates","male_graduates","female_graduates"]].groupby("state_name").agg({"total_graduates":np.sum,

                                                                                                                        "male_graduates":np.sum,

                                                                                                                        "female_graduates":np.sum})



sort_graduates = state_graduate.sort_values(by=['total_graduates'], ascending=False)
#plotting aggregate literacy rate for overall population along with male and female

sort_graduates.plot(kind='bar', figsize=(20,9),alpha = 1)

plt.xticks(range(len(sort_graduates)+1),rotation = 90)

plt.xlabel("States")

plt.ylabel('#number of Grduates in different states')

plt.title('Top states as per number of graduates')

#Maharshtra and UP has more graduates as evident as it has more population than any other states in India
state_grouped  = cities[['state_name','literates_total','sex_ratio','total_graduates','male_graduates','female_graduates','effective_literacy_rate_female']].groupby("state_name")

state_grouped_agg = state_grouped.agg({"literates_total":np.sum,"sex_ratio":np.sum,"total_graduates":np.sum, 

                                      'male_graduates':np.sum, 'female_graduates':np.sum,'effective_literacy_rate_female':np.sum})
state_grouped_agg['literates_total'].plot(kind='bar',color=('blue'), figsize=(10,8))

state_grouped_agg['total_graduates'].plot(kind='line',color = ('black'),linewidth=2.0,use_index = True)

state_grouped_agg['female_graduates'].plot(kind='line',color = ('orange'),linewidth=2.0,use_index = True)

plt.xticks(rotation = 90)

plt.xlabel("States")

plt.ylabel('Number of Literates in different states')

plt.title('Understanding female graduates amongst all literates and total graduates')
state_grouped_agg['total_graduates'].plot(kind='bar',color=('blue'), figsize=(10,8))

state_grouped_agg['female_graduates'].plot(kind='line',color = ('orange'),linewidth=2.0,use_index = True)

plt.xticks(rotation = 90)

plt.xlabel("States")

plt.ylabel('Number of total Grduates in different states')

plt.title('Female graduates in total graduates')