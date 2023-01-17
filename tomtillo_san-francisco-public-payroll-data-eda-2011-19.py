import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pandas as pd 

import os 

import seaborn as sns

import matplotlib.pyplot as plt 
#Define Colors and palettes 



palette_gray_1 = ["#95a5a6"]

palette_gray_2 = ["#95a5a6","#15a5a6"]

palette_green = ["#15a5a6"]

palette_gray_10 = ["#95a5a6","#85a5a6" , "#75a5a6" , '#65a5a6' , '#55a5a6', "#45a5a6","#35a5a6" , "#25a5a6" , '#15a5a6' , '#05a5a6']

palette_gray_10_revrs = palette_gray_10[::-1]
df_master = pd.read_csv('/kaggle/input/san-francisco-city-payrollsalary-data-20112019/san-francisco-payroll_2011-2019.csv')
df_master.head()
sns.countplot(data= df_master, y = 'Year',palette=palette_gray_2)

plt.xlabel('Count of jobs')

plt.show();
#Full time to part-time ratio

sns.countplot(data = df_master[df_master['Status'].notnull()] , y = 'Year' ,hue ='Status')

plt.xlabel('Count of jobs')

plt.show();
# most common Job Title 

top10jobs_count = df_master['Job Title'].value_counts().sort_values(ascending=False).head(10).index

df_top10jobs_bycount = df_master[df_master['Job Title'].isin(list(top10jobs_count))]

sns.set(style="whitegrid")

sns.countplot(data = df_top10jobs_bycount ,y = 'Job Title', order= df_top10jobs_bycount['Job Title'].value_counts().index, palette=palette_gray_10_revrs)

plt.show();
df_top_salary = df_master.groupby('Job Title').median().reset_index().sort_values(by='Total Pay', ascending = False).head(10)

top10jobs_salary =  list(df_top_salary['Job Title'])

df_top10jobs_bysalary = df_master[df_master['Job Title'].isin(list(top10jobs_salary))]

sns.barplot(data = df_top_salary,  x = 'Total Pay' , y = 'Job Title' , palette=palette_gray_10_revrs)

plt.show();
#Distribution of the top salaried jobs ( ordered by median salary)



fig1=plt.figure()

fig1.set_size_inches(8, 5)

sns.boxplot(y='Total Pay',x='Job Title',data=df_top10jobs_bysalary , order = df_top_salary['Job Title'] ,width=0.6,  palette=palette_gray_10_revrs)

plt.xticks(rotation=85)

plt.show();
df_temp_1 = df_top10jobs_bycount[df_top10jobs_bycount['Job Title'] == 'Firefighter'].groupby('Year').median()

df_temp_2 = df_top10jobs_bysalary[df_top10jobs_bysalary['Job Title'] == 'Chief Investment Officer'].groupby('Year').mean()

df_temp_1.reset_index()

df_temp_2.reset_index()





sns.set(style="whitegrid")

sns.lineplot(data=df_temp_1['Total Pay'] , hue= 'Job Title' , palette=palette_gray_10)

plt.xlabel('Year')

plt.ylabel('Salary over the years - Firefigher')

plt.show();
sns.lineplot(data=df_temp_2['Total Pay'] , palette="tab10", linewidth=2.5)

plt.xlabel('Year')

plt.ylabel('Salary over the years-CIO')

plt.show();