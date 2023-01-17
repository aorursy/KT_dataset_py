import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



pd.options.mode.chained_assignment = None  # We'll find this warning annoying later.



data = pd.read_csv('../input/marathon_results_2016.csv')





def convert_to_seconds(time):

    seconds = time.str.split(':').map(lambda x: int(x[-1]) + int(x[-2]) * 60 + int(x[-3]) * 3600)

    return seconds



data.head()
print(data.shape)

print(data['M/F'].value_counts())
Male = data[data['M/F'] == 'M']

Male['Male Time'] = Male['Official Time']

Male = Male[['Gender', 'Male Time']]

Female = data[data['M/F'] == 'F']

Female['Female Time'] = Female['Official Time']

Female = Female[['Gender', 'Female Time']]

genders = pd.merge(Male, Female, on='Gender')
genders['Male Time'] = convert_to_seconds(genders['Male Time'])

genders['Female Time'] = convert_to_seconds(genders['Female Time'])
genders.head()
plt.plot(genders['Male Time'], genders['Female Time'], 'r-')

plt.xlabel('Male Time in Seconds')

plt.ylabel('Female Time in Seconds')
genders['Diff'] = genders['Male Time'] - genders['Female Time']
plt.plot(genders['Gender'], genders['Diff'], 'g--')

plt.xlabel('Place')

plt.ylabel('Men - Women Time in Seconds')
sns.distplot(genders['Diff'])
slow_men = data.loc[data['M/F'] == 'M', ['Official Time', 'Gender']]

slow_men.Gender = slow_men.Gender.rank(ascending=False)

slow_men['Male Time'] = slow_men['Official Time']

slow_men = slow_men[['Gender', 'Male Time']]

slow_women = data.loc[data['M/F'] =='F', ['Official Time', 'Gender']]

slow_women.Gender = slow_women.Gender.rank(ascending=False)

slow_women['Female Time'] = slow_women['Official Time']

slow_women = slow_women[['Gender', 'Female Time']]

slow_genders = slow_men.merge(slow_women)

# Convert the time to an integer

slow_genders['Male Time'] = convert_to_seconds(slow_genders['Male Time'])

slow_genders['Female Time'] = convert_to_seconds(slow_genders['Female Time'])

slow_genders['Diff'] = slow_genders['Male Time'] - slow_genders['Female Time']
plt.plot(slow_genders['Male Time'], slow_genders['Female Time'], 'y--')

plt.ylabel('Womens Time in Seconds')

plt.xlabel('Mens Time in Seconds')
plt.plot(slow_genders.Gender, slow_genders.Diff, 'b.')

plt.xlabel('Place')

plt.ylabel('Men - Women Time in Seconds')