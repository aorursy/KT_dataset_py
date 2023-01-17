import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

from scipy import stats

%matplotlib inline

plt.style.use('seaborn-darkgrid')
## open csv in sublime save as utf8 encoded



df = pd.read_csv("../input/us-mass-shootings-last-50-years/Mass Shootings Dataset Ver 5.csv", encoding = "ISO-8859-1")
df.info()
pd.options.display.max_columns = 50

df.head(5)
## rename all cols to lower case, and replace ' ' with '_'



cols = df.columns.tolist()

cols_rn = [col.lower().replace(' ', '_') for col in cols]

cols = zip(cols, cols_rn)

mydict = {}

for col in cols:

    mydict[col[0]] = col[1]

df.rename(columns=mydict, inplace=True)
## change data to date time and create a 'year' feature

## rename columns



df['date'] = pd.to_datetime(df['date'])

df['year'] = df.loc[:, 'date'].apply(lambda x: x.year)

df.rename(columns={'s#':'incident_ct', 

                   'total_victims': 'total_casualties',

                   'employeed_(y/n)':'employed'

                  }, inplace=True)

df.age.fillna('0', inplace=True)
## 5 year grouping feature



df['year_5'] = df['year'].apply(lambda x: x - 1965)

df['year_5'] = df['year_5'].apply(lambda x: (x // 5))

df['year_5'] = df['year_5'].apply(lambda x: (x * 5) + 1965)
## clean age feature

## bad entries have multiple ages for multiple shooters, took last in list for oldest



def age_clean(x):

    y = re.search(r'(\d+),', x)

    if y:

        return int(x.split(',')[-1])

    else:

        return int(x)

df['age'] = df['age'].apply(age_clean)
## 10 year age group feature, some Nan's and some bad entries



def age_group(x):

    if x == 0:

        return 0

    else :

        return x // 10 * 10

df['age_10'] = df['age'].apply(age_group)
## condense race descriptions into more concise list



df['race'].fillna('unknown', inplace=True)

df['race'] = df['race'].apply(lambda x: x.lower())

df.loc[df['race'] == 'some other race', 'race'] = 'other'

df.loc[df['race'] == 'black american or african american', 'race'] = 'black'

df.loc[df['race'] == 'white american or european american', 'race'] = 'white'

df.loc[df['race'] == 'asian american', 'race'] = 'asian'

df.loc[df['race'] == 'black american or african american/unknown', 'race'] = 'black'

df.loc[df['race'] == 'white american or european american/some other race', 'race'] = 'white'

df.loc[df['race'] == 'asian american/some other race', 'race'] = 'asian'
df['gender'].unique().tolist()
## clean gender feature



df.loc[df['gender'] == 'M', 'gender'] = 'Male'

df.loc[df['gender'] == 'M/F', 'gender'] = 'Male/Female'
## create df_year_5



df_year_5 = df.groupby('year_5', as_index=False).agg({'fatalities': 'sum', 'injured': 'sum', 

                                                    'total_casualties': 'sum', 'incident_ct': 'count'})

df_year_5['casualties_per_incident'] = df_year_5.loc[:, ['total_casualties', 'incident_ct']].apply(

                                                    lambda x: x[0] / x[1], axis=1)
## charts by year for total casualties, number of incidents, and average casualties per incident



dims = (10,17)

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=dims)

plt.setp(axes, xticks=np.arange(min(df_year_5['year_5']), max(df_year_5['year_5']) + 5, 5))







axes[0].plot(df_year_5['year_5'], df_year_5['total_casualties'])

axes[0].set_ylabel('Total Casualties', fontsize=14)

axes[0].set_title('Toatl Casualties by 5 Year Increments', fontsize=20)



axes[1].plot(df_year_5['year_5'], df_year_5['incident_ct']);

axes[1].set_ylabel('Total Count Incidents', fontsize = 14)

axes[1].set_title('Total Incident Count in 5 Year Increments', fontsize = 20);





axes[2].plot(df_year_5['year_5'], df_year_5['casualties_per_incident']);

axes[2].set_ylabel('Average Casualties per Incident', fontsize = 14)

axes[2].set_title('Average Casualties per Incident in 5 Year Increments', fontsize = 20);
## look at deaths as a percent of casualties



df_year_5['death_pct'] = df_year_5[['fatalities', 'total_casualties']].apply(lambda x: x[0] / x[1], axis=1)
## chart to show if there is any pattern of increases in fatalities over time



dims = (11, 6)

fig, ax = plt.subplots(figsize=dims)



plt.plot(df_year_5['year_5'], df_year_5['death_pct']);

ax.set_xlabel('5 Year Increments', fontsize = 14)

ax.set_ylabel('Deaths as a Percent of Casualties', fontsize = 14)

ax.set_title('Deaths as a Percent of Casualties in 5 Year Increments', fontsize = 20);

plt.xticks(np.arange(min(df_year_5['year_5']), max(df_year_5['year_5']) + 5, 5));
## set up table to show age graphic charts

## age, gender, race



df.head()

df_demo = df.groupby(['year_5', 'age_10', 'gender', 'race']).agg({

    'fatalities':sum,

    'total_casualties':sum,

    'incident_ct':'count'  

}).sort_index()

df_demo.reset_index(inplace=True)

df_demo['casualties_per_incident'] = df_demo.loc[:, ['total_casualties', 'incident_ct']].apply(

                                                    lambda x: x[0] / x[1], axis=1)
dims = (11, 6)

fix, ax = plt.subplots(figsize=dims)





sns.barplot(x='age_10', y='total_casualties', data=df_demo[df_demo['age_10'] != 0], estimator=sum, ci=None);

ax.set_xlabel('Age 10 Year Increments', fontsize = 14)

ax.set_ylabel('Toatl Casualties', fontsize = 14)

ax.set_title('Total Casualties by Age Group', fontsize = 20);
dims = (11, 6)

fix, ax = plt.subplots(figsize=dims)



sns.barplot(x='age_10', y='incident_ct', data=df_demo[df_demo['age_10'] != 0], estimator=sum, ci=None);

ax.set_xlabel('Age 10 Year Increments', fontsize = 14)

ax.set_ylabel('Total Count Incidents', fontsize = 14)

ax.set_title('Total Count Incidents by Age Group', fontsize = 20);
## excluded the 2017 shooting with ~600 casualties because it is an outlier and was

## mudying the data for 60 bc there are only 3 incidents for that age group.



dims = (11, 6)

fix, ax = plt.subplots(figsize=dims)



sns.barplot(x='age_10', 

            y='casualties_per_incident', 

            data=df_demo[(df_demo['age_10'] != 0) & (df_demo['total_casualties'] < 200)], 

            estimator=np.mean, 

            ci=None)

ax.set_xlabel('Age 10 Year Increments', fontsize = 14)

ax.set_ylabel('Casualties Per Incident', fontsize = 14)

ax.set_title('Average Casualties Per Incident by Age Group', fontsize = 20);
dims = (11, 6)

fix, ax = plt.subplots(figsize=dims)



sns.barplot(x='year_5', y='incident_ct', hue='age_10',

            data=df_demo[df_demo['age_10'] != 0], 

            estimator=sum, ci=None);

ax.set_xlabel('5 Year Increments', fontsize = 14)

ax.set_ylabel('Incident Count', fontsize = 14)

ax.set_title('Incident Count by Year by Age Group', fontsize = 20);
dims = (11, 6)

fix, ax = plt.subplots(figsize=dims)



sns.barplot(x='year_5', y='age',

            data=df[df['age_10'] != 0], 

            estimator=np.mean, ci=None);

ax.set_xlabel('5 Year Increments', fontsize = 14)

ax.set_ylabel('Average Age of Shooter', fontsize = 14)

ax.set_title('Average Age of Shooter by Year', fontsize = 20);
## Check what which year the unknown ages come from



def age_test(x):

    if x == 0:

        return 'n/a'

    else:

        return 'age'

df_demo_test = df[['age_10', 'year_5']].copy()

df_demo_test['no_age'] = df_demo_test['age_10'].apply(age_test)
dims = (11, 6)

fix, ax = plt.subplots(figsize=dims)



sns.countplot(x='year_5', hue='no_age',

            data=df_demo_test);

ax.set_xlabel('5 Year Increments', fontsize = 14)

ax.set_ylabel('Incident Count', fontsize = 14)

ax.set_title('Incident Count for 20-29 y.o. by Year', fontsize = 20);
df.loc[df['year_5'] <= 1970, ['year', 'year_5', 'age', 'fatalities', 'total_casualties', 'incident_ct']]
dims = (11, 6)

fix, ax = plt.subplots(figsize=dims)



sns.barplot(x='race', y='incident_ct', data=df_demo, estimator=sum, ci=None);

ax.set_xlabel('Race', fontsize = 14)

ax.set_ylabel('Total Incidents', fontsize = 14)

ax.set_title('Total Incidents by Race', fontsize = 20);

for tick in ax.get_xticklabels():

    tick.set_rotation(45)

    tick.set_fontsize(12)
dims = (11, 6)

fix, ax = plt.subplots(figsize=dims)



sns.barplot(x='race', y='casualties_per_incident', 

            data=df_demo[(df_demo['total_casualties'] < 500)], 

            estimator=np.mean, ci=None);

ax.set_xlabel('Race', fontsize = 14)

ax.set_ylabel('Casulaties per Incident', fontsize = 14)

ax.set_title('Total Casualties per Incident by Race', fontsize = 20);

for tick in ax.get_xticklabels():

    tick.set_rotation(45)

    tick.set_fontsize(12)
dims = (11, 6)

fix, ax = plt.subplots(figsize=dims)



sns.barplot(x='gender', y='incident_ct', data=df_demo, estimator=sum, ci=None);

ax.set_xlabel('Gender', fontsize = 14)

ax.set_ylabel('Total Incidents', fontsize = 14)

ax.set_title('Total Incidents by Gender', fontsize = 20);