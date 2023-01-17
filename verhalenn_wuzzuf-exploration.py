import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



wuzzuf_data = pd.read_csv('../input/Wuzzuf_Job_Posts_Sample.csv')

wuzzuf_data.head()
wuzzuf_data.describe()
sns.distplot(wuzzuf_data.views)
sns.regplot(x='salary_minimum', y='views', data=wuzzuf_data)
sns.regplot(x='salary_maximum', y='views', data=wuzzuf_data)
plt.plot(wuzzuf_data.salary_minimum, wuzzuf_data.salary_maximum, 'b.')

plt.xlabel('Minimum Salary')

plt.ylabel('Maximum Salary')
wuzzuf_data.city.value_counts()
wuzzuf_data.job_title.value_counts().head(20)
frequent_job_titles = wuzzuf_data.job_title.value_counts()

frequent_job_titles = frequent_job_titles[frequent_job_titles > 100].index

frequent_jobs = wuzzuf_data[wuzzuf_data.job_title.isin(frequent_job_titles)]
print(frequent_jobs.groupby('job_title').salary_minimum.mean())

print(frequent_jobs.groupby('job_title').salary_maximum.mean())
for job in frequent_job_titles:

    temp_data = frequent_jobs[(frequent_jobs.salary_maximum < 80000) & (frequent_jobs.job_title == job)]

    plt.scatter(temp_data.salary_maximum, temp_data.salary_minimum)



plt.legend(labels=frequent_job_titles)