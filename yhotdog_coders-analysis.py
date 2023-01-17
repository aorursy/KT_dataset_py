import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/2016-FCC-New-Coders-Survey-Data.csv')

data = data[data['Age'].notnull()]

data['JobRoleInterest'] = data['JobRoleInterest'].fillna('').map(lambda job: job.strip())
data['Age'].plot.hist(title='ages (N=%d, mean=%.2f)' % (len(data), data['Age'].mean()), bins=100)
average_ages = pd.DataFrame([[group, len(data.loc[rows]['Age']), data.loc[rows]['Age'].mean()]

                             for group, rows in data.groupby('JobRoleInterest').groups.items()],

                            columns=['job', 'N', 'average age']).sort('average age')

sns.barplot(data=average_ages, x='average age', y='job', hue='N', palette='Blues_d')
import scipy.stats

scipy.stats.probplot(average_ages['average age'], plot=plt)

plt.figure()

scipy.stats.probplot(average_ages['average age'].iloc[1:len(average_ages) - 1], plot=plt)
qa = average_ages[average_ages['job'] == 'Quality Assurance Engineer'].iloc[0]

num_of_simulations = 10000

print(1. * sum(data.sample(n=qa['N'])['Age'].mean() >= qa['average age']

               for _ in range(num_of_simulations)) / num_of_simulations)
mobile = average_ages[average_ages['job'] == 'Product Manager'].iloc[0]#Mobile Developer'].iloc[0]

num_of_simulations = 10000

print(1. * sum(data.sample(n=mobile['N'])['Age'].mean() <= mobile['average age']

               for _ in range(num_of_simulations)) / num_of_simulations)