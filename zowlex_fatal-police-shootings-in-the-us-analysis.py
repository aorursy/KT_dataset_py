# import all packages and set plots to be embedded inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

import warnings



warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',100)

%matplotlib inline



uni_color = sb.color_palette()[0]
#defining functions



def pie_plot(df, cat_var):

    """

    plots a cat_var from given df pie plot with ordered values, 90° start angle and counterclock direction

    """

    sorted_counts = df[cat_var].value_counts()

    plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90,

            counterclock = False);

    plt.axis('square')

    plt.show()
#data cleaning was made in a separate notebook 'data cleaning.ipynb'



path = '../input/fatal-police-shootings-clean'



median_house_income = pd.read_csv(f'{path}/median_house_income_clean.csv')

percentage_below_poverty_level = pd.read_csv(f'{path}/percentage_below_poverty_level_clean.csv')

percent_over25_comp_highschool = pd.read_csv(f'{path}/percent_over25_comp_highschool_clean.csv')

share_by_race = pd.read_csv(f'{path}/share_by_race_clean.csv')

police_killings = pd.read_csv(f'{path}/police_killings_clean.csv', parse_dates=['date'])
print(police_killings.shape)

print(police_killings.dtypes)

police_killings.head()
print(median_house_income.shape)

print(median_house_income.dtypes)

median_house_income.head()
print(percentage_below_poverty_level.shape)

print(percentage_below_poverty_level.dtypes)

percentage_below_poverty_level.head()
print(percent_over25_comp_highschool.shape)

print(percent_over25_comp_highschool.dtypes)

percent_over25_comp_highschool.head()
print(share_by_race.shape)

print(share_by_race.dtypes)

share_by_race.head()
plt.hist(police_killings.age)

plt.xlabel('age')

plt.ylabel('count')

plt.show()
plt.hist(police_killings.age, bins=40)

plt.xlabel('age')

plt.ylabel('count')

plt.show()
plt.hist(median_house_income.median_income)

plt.xlabel('median outcome')

plt.ylabel('count')

plt.show()
log_data = np.log10(median_house_income.median_income)

log_bin_edges = np.arange(0, log_data.max()+0.1, 0.1)

plt.hist(log_data, bins = log_bin_edges)

plt.xlabel('log(median_house_income)')

plt.show()
log_data = np.log10(median_house_income.median_income)

log_bin_edges = np.arange(4, log_data.max()+0.1, 0.1) #the plot is zoomed to start from 10**4

plt.hist(log_data, bins = log_bin_edges)

plt.xlabel('log(median_house_income)')

plt.show()
plt.figure(figsize=(15,5))



dates = police_killings.set_index('date').groupby(pd.Grouper(freq='M'))['id'].count()

sb.lineplot(data=dates)

plt.ylabel('# of deaths per month')

plt.xticks(rotation=90);
#body_cam

sb.countplot(data=police_killings, x='body_camera', color=uni_color);
#gender

sb.countplot(data=police_killings, x='gender', color=uni_color);
sb.countplot(data=police_killings, x='race', color=uni_color, order=police_killings.race.value_counts().index);
(share_by_race.mean()/100).reset_index(name='proportions')
us_pop_2015 = 320000000.7

prop = share_by_race.mean()/100 * us_pop_2015

killings_per_race_count = police_killings.race.value_counts()[:-1] #There's no data for other races, that's why we exclude it from the count



killings_per_race_count.loc['W']/=prop.loc['share_white']

killings_per_race_count.loc['B']/=prop.loc['share_black']

killings_per_race_count.loc['N']/=prop.loc['share_native_american']

killings_per_race_count.loc['H']/=prop.loc['share_hispanic']

killings_per_race_count.loc['A']/=prop.loc['share_asian']
sb.barplot(x=killings_per_race_count.index, y=killings_per_race_count.values, color=uni_color);
pie_plot(police_killings, 'manner_of_death')
pie_plot(police_killings, 'signs_of_mental_illness')
most_used = police_killings.armed.value_counts()>50

sb.countplot(data=police_killings.loc[police_killings.armed.isin(most_used[most_used].index.tolist())],

             y='armed', color=uni_color);
plt.figure(figsize=(9,5))



plt.subplot(121)

ax1 = sb.countplot(data=police_killings, x='threat_level', color=uni_color)

ax1.title.set_text('threat_level')



plt.subplot(122)

ax1 = sb.countplot(data=police_killings, x='flee', color=uni_color)

ax1.title.set_text('flee')
plt.figure(figsize=(7,10))

sb.countplot(data=police_killings, y='state', color=uni_color, order=police_killings.state.value_counts().index);
#age vs gender

sb.boxplot(data = police_killings, x = 'gender', y = 'age', color = uni_color);
#age vs. race

sb.pointplot(data = police_killings, x = 'race', y = 'age', color = uni_color, linestyles='')

plt.ylabel('avg age');
#age vs. signs_of_mental_illness

sb.violinplot(data = police_killings, x = 'signs_of_mental_illness', y = 'age', color = uni_color, inner='quartile');
#median_house_income/percentage_below_poverty_level

plt.figure(figsize=(15,4))



state_income = median_house_income.loc[median_house_income.geographic_area.isin(['CA','RI'])]

state_poverty_lvl = percentage_below_poverty_level.loc[percentage_below_poverty_level.geographic_area.isin(['CA','RI'])]

state_comp = percent_over25_comp_highschool.loc[percent_over25_comp_highschool.geographic_area.isin(['CA','RI'])]



plt.subplot(131)

sb.pointplot(data=state_income,x='geographic_area', y='median_income',color = uni_color, linestyles='', ci='sd')



plt.subplot(132)

sb.pointplot(data=state_poverty_lvl,x='geographic_area', y='poverty_rate',color = uni_color, linestyles='')



plt.subplot(133)

sb.pointplot(data=state_comp,x='geographic_area', y='percent_completed_hs',color = uni_color, linestyles='');

data = share_by_race.loc[share_by_race.geographic_area.isin(['CA','RI'])].groupby('geographic_area').mean()

data.plot(kind='bar',figsize=(15,4));
plt.figure(figsize=(15,8))



cat_means = police_killings.groupby(['gender', 'race']).mean()['age']

cat_means = cat_means.reset_index(name = 'age_avg')

cat_means = cat_means.pivot(index = 'gender', columns = 'race',

                            values = 'age_avg')

sb.heatmap(cat_means,annot=True, fmt = '.3f',

           cbar_kws = {'label' : 'mean(age)'})

plt.title('');
#https://stackoverflow.com/questions/45122416/one-horizontal-colorbar-for-seaborn-heatmaps-subplots-and-annot-issue-with-xtick

plt.figure(figsize=(20,15))



cat_means = police_killings.loc[police_killings.armed.isin(most_used[most_used].index.tolist())].groupby(['state', 'armed']).count()['id']

cat_means = cat_means.reset_index(name = 'count')

cat_means = cat_means.pivot(index = 'armed', columns = 'state',

                            values = 'count')

sb.heatmap(cat_means,cbar_kws={'orientation': 'horizontal', 'label' : 'weapons count', "shrink": .80},annot=True,cmap=sb.cm.rocket_r, square=True);
police_killings['year-month'] = police_killings.date.apply(lambda x: x.strftime('%b-%Y')) 
#https://stackoverflow.com/questions/25146121/extracting-just-month-and-year-separately-from-pandas-datetime-column



plt.figure(figsize=(15,8))



dates1 = police_killings.groupby(['race','year-month'])['id'].agg('count').reset_index().rename(columns={'id':'count'}).sort_values(by='year-month')



custom_dict = {x:i for i,x in enumerate(police_killings.sort_values(by='date')['year-month'].unique())}



df = dates1[dates1['race'] == 'B']

sb.lineplot(data=df.iloc[df['year-month'].map(custom_dict).argsort()] , x='year-month', y='count', sort=False)



df = dates1[dates1['race'] == 'W']

sb.lineplot(data=df.iloc[df['year-month'].map(custom_dict).argsort()], x='year-month', y='count', sort=False)



df = dates1[dates1['race'] == 'N']

sb.lineplot(data=df.iloc[df['year-month'].map(custom_dict).argsort()], x='year-month', y='count', sort=False)



df = dates1[dates1['race'] == 'H']

sb.lineplot(data=df.iloc[df['year-month'].map(custom_dict).argsort()], x='year-month', y='count', sort=False)



df = dates1[dates1['race'] == 'A']

sb.lineplot(data=df.iloc[df['year-month'].map(custom_dict).argsort()], x='year-month', y='count', sort=False)



df = dates1[dates1['race'] == 'O']

sb.lineplot(data=df.iloc[df['year-month'].map(custom_dict).argsort()], x='year-month', y='count', sort=False)







plt.xticks(rotation=90)

plt.ylabel('kills per month')

plt.legend(['B','W','N','H','A','O']);