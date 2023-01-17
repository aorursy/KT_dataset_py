import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns

master = pd.read_csv('../input/master.csv')

master.take(np.random.permutation(len(master))[:10])
master.isnull().sum()
for i in master.columns:

    if master[i].dtypes == object and len(master[i].unique()) < 1000:

        print(i)

        print(master[i].unique())
# Check how many data are there for each year

master.groupby('year').year.count()
fig = plt.figure(figsize=(15,8))

ax = fig.add_subplot(1, 1, 1)

ax.plot(master.groupby(['year']).suicides_no.sum(),'ko--')

tick = ax.set_xticks(range(1985,2018,2))

ax.set_title('Figure 1: The total count of suicides  from 1985 to 2016')

ax.set_xlabel('year')

ax.set_ylabel('suicide number')
# check the total number of country

len(master.country.unique())
def count_country(group):

    return len(group.country.unique())

country_no = master.groupby('year').apply(count_country)

country_no.name = 'number_of_country'

country_no
# I want to know the occurances of each 101 countries from 1985 - 2016. 

def count_year(group):

    return len(group.year.unique())

country_o = master.groupby('country').apply(count_year).sort_values(ascending = False)

country_o.name = 'occurances of country'

country_o
# Find the number of country that appear less that 20 times over the 32 years

country_o[country_o < 20]

# There were 24 countries appear less than 20 times over the 32 years. 

# About 25% of country are missing more than 10 yr of data.
coun_year = master.groupby('country-year').population.sum()

full_country_year_list = []

for year in range(1985,2017):

    for country in master.country.unique():

        full_country_year_list.append(country + str(year))

        

full_country_year_df = pd.DataFrame(coun_year.reindex(full_country_year_list))



full_country_year_df['year'] = [i[-4:] for i in full_country_year_df.index]

full_country_year_df['country'] = [i[:-4] for i in full_country_year_df.index]



country_year_pivot = pd.pivot_table(full_country_year_df,values = 'population',columns = 'country',index='year',aggfunc = 'count')



country_year_pivot
fig,axes = plt.subplots(figsize=(25,8))

sns.heatmap(country_year_pivot,ax=axes)
yr_grouped = master.groupby('year')

annual_pop = yr_grouped.population.sum()

annual_suicides = yr_grouped.suicides_no.sum()

annual_sui_rate = annual_suicides/annual_pop*100000

country_df = pd.concat([country_no,annual_suicides,annual_pop],axis = 1)

country_df['annual_sui_rate'] = country_df.suicides_no/country_df.population*100000

country_df.head(10)
sns.heatmap(country_df.corr(),annot = True)
fig,axes = plt.subplots(1,3,figsize=(18,4))

axes[0].scatter(annual_pop,country_no)

axes[1].scatter(annual_suicides,country_no)

axes[2].scatter(annual_sui_rate,country_no)

plt.subplots_adjust(wspace=0.1, hspace=1)

axes[0].set_title('Annual Population and Number of Country')

axes[0].set_xlabel('Total annual population')

axes[0].set_ylabel('Nummber of country')

axes[1].set_title('Annual Count of Suicides and Number of Country')

axes[1].set_xlabel('Total annual count of suicides')

axes[2].set_title('Annual Suicide Rate and Number of Country')

axes[2].set_xlabel('Annual suicide rate')
country_df[country_df.number_of_country == country_no.min()]
# I drop data from 2016

# drop 'HDI for year' columns for too many na value

# drop 'country-year' and 'suicides/100k pop' columns for I don't use them

nmaster = master[~ (master.year == 2016)].drop(['HDI for year','country-year','suicides/100k pop'],axis = 1)

nmaster.head(10)
ncountry_o = nmaster.groupby('country').apply(count_year).sort_values(ascending = False)

ncountry_o[country_o >= 30] 

# what's Sweden doing over there? With a value of 29? Whyyyyy???
# List of region that I selected, their data were collected continuely from 1985 to 2015

ncountry_o[country_o >= 31].index
pmaster = nmaster[nmaster.country.isin(ncountry_o[country_o >= 31].index)]

fig,axes = plt.subplots(figsize = (15,8))

to_sui = pmaster.groupby(['year']).suicides_no.sum()

to_pop = pmaster.groupby(['year']).population.sum()

sui_rate = to_sui/(to_pop/100000)

axes.bar(range(1985,2016),to_pop)

axes1 =axes.twinx()

axes1.plot(range(1985,2016),sui_rate,'ko--')

axes1.set_yticks(range(8,13,1))

axes.set_title('Total population and suicide rate from 1985 to 2015')

axes.set_xlabel('year')

axes.set_ylabel('total populaiton')

axes1.set_ylabel('suicides rate')
fig,axes = plt.subplots(3,1,figsize = (13,10))

to_pop.plot(kind='bar', ax=axes[0],title = 'population',color = 'silver')

to_sui.plot(kind = 'bar',ax = axes[1],title='count of suicides',color='silver')

axes[2].plot(sui_rate,'ko--')

axes[2].set_title('suicide rate')

plt.subplots_adjust(hspace=0.6)
country_sui_no = pmaster.groupby(['country']).suicides_no.sum()

country_sui_no_sort = country_sui_no.sort_values()

country_pop = pmaster.groupby(['country']).population.sum()

country_sui_rate = country_sui_no/country_pop*100000

fig,axes = plt.subplots(1,2,figsize = (13,5))

country_sui_no_sort.plot(kind = 'barh',ax = axes[0],title = 'Total count of suicides for countries')

country_sui_rate.sort_values().plot(kind = 'barh',ax = axes[1],title = 'Suicides rate for countries')
# find the 5 highest count of suicides

top_count = country_sui_no_sort.nlargest(5)

top_count
# find the 5 highest suicides rate

top_rate = country_sui_rate.nlargest(5)

top_rate
top_count[top_count.index.isin(top_rate.index)].index
age_group = pd.pivot_table(pmaster,index=['year'],values='suicides_no',columns=['age'],aggfunc=np.sum)

sui_age=age_group[['5-14 years','15-24 years', '25-34 years', '35-54 years','55-74 years', '75+ years']]

sui_age.plot(kind = 'bar',figsize=(10,6),stacked=True,title='How each age groups contribute to the count of suicides')
fig,axes = plt.subplots(1,3,figsize = (16,5))

sui_age.plot(ax=axes[0],title = 'count of suicides for different age groups')

pop_group = pd.pivot_table(pmaster,index=['year'],values='population',columns=['age'],aggfunc=np.sum)

pop_age = pop_group[['5-14 years','15-24 years', '25-34 years', '35-54 years','55-74 years', '75+ years']]

pop_age.plot(ax=axes[1],title = 'population for different age groups')

rate_age = sui_age/pop_age*100000

rate_age.plot(ax=axes[2],title = 'suicide rate for different age groups')
# prepare a nested donut chart for age and sex (part i)

group1=pmaster.groupby(['age','sex'],as_index = False).suicides_no.sum()

group1['percent']=round(group1.suicides_no/group1.suicides_no.sum()*100,2)

group1['sex and percent'] = group1.sex + ' ' + group1.percent.apply(str) +'%'

group1
# prepare a nested donut chart for age and sex (part ii)

group2 = group_by_age=pmaster.groupby('age').suicides_no.sum()

group2 = group2.to_frame()

group2['percent'] = round(group2.suicides_no/group2.suicides_no.sum()*100,2)

group2['age and percent'] = group2.index + ' ' + group2.percent.apply(str) +'%'

group2
# draw the nested donut chart

fig, ax = plt.subplots()

ax.axis('equal')

width = 1

cm = plt.get_cmap("tab20c")

cout = cm(np.arange(6)*4)

pie, _ = ax.pie(group2.suicides_no, radius=4, labels=group2['age and percent'], colors=cout)

plt.setp( pie, width=width, edgecolor='white')

cin = cm(np.array([1,2,5,6,9,10,13,14,17,18,21,22]))

pie2, _ = ax.pie(group1.suicides_no, radius=4-width, labels=group1['sex and percent'],labeldistance=0.8,colors=cin)

plt.setp( pie2, width=width, edgecolor='white')

ax.set_title('Nested donut chart for age and sex')
sex_ratio = pd.pivot_table(pmaster,values = 'suicides_no',index = ['age'],columns = ['sex'],aggfunc=np.sum)

sex_ratio['ratio'] = sex_ratio.male/sex_ratio.female

sex_ratio.take([3,0,1,2,4,5])
ratio_chart = sex_ratio.take([3,0,1,2,4,5]).ratio.plot(kind = 'bar',color = 'silver',figsize = (8,4),title = 'Sex ratio of each age group on count of suicides')

ratio_chart.set(ylabel = 'Sex ratio') 

ratio_chart.tick_params(axis='x', rotation=0)