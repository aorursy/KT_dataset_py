import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
cases= pd.read_csv('../input/malaria-dataset/estimated_numbers.csv')
cases.head()
incidence = pd.read_csv('../input/malaria-dataset/incidence_per_1000_pop_at_risk.csv')
incidence.head()
reports = pd.read_csv('../input/malaria-dataset/reported_numbers.csv')
reports.head()
cases.shape
incidence.shape
reports.shape
cases.info()
reports.info()
df1 = reports.sort_values(by = 'No. of cases', ascending=False)
df1 = df1[df1['Year']==2017][:50]
#Top 50 countries with largest number of cases in 2017
plt.figure(figsize=(18,10))
sns.barplot(df1['Country'], df1['No. of cases'])
plt.xticks(rotation=90, ha='right');
df2 = reports.sort_values(by = 'No. of cases', ascending=False)
df2 = df2[df2['Year']==2016][:50]
#Top 50 countries with largest number of cases in 2016
plt.figure(figsize=(18,10))
sns.barplot(df2['Country'], df2['No. of cases'])
plt.xticks(rotation=90, ha='right');
#Top 50 countries with largest number of deaths in 2017
df3 = df1[df1['Year']==2017][:50]
df3 = df3.sort_values(by = 'No. of deaths', ascending=False)
df3.head()
#Top 50 countries with largest number of deaths in 2017
plt.figure(figsize=(18,10))
sns.barplot(df3['Country'], df3['No. of deaths'])
plt.xticks(rotation=90, ha='right');
#Relationship between number of cases and number of deaths reported
plt.figure(figsize =(8,5))
plt.title('Relationship between number of cases and number of deaths reported', fontsize=14, fontweight='bold')
ax =sns.scatterplot(x= 'No. of deaths', y ='No. of cases',data = reports)
df4 =reports.groupby('Year').sum().loc[:, ['No. of cases', 'No. of deaths']]
plt.figure(figsize =(8,5))
ax = sns.lineplot(data=df4)
plt.xlabel('Year', fontsize=15)
plt.title('Malaria death and cases over the years', fontsize=14, fontweight='bold')
plt.show()
#Total cases reported which was region wise
df5 =reports.groupby('WHO Region').sum().loc[:, ['No. of cases', 'No. of deaths']]
plt.figure(figsize =(8,5))
ax = sns.lineplot(data=df5)
plt.xlabel('Year', fontsize=15)
plt.title('Malaria death and cases in particular WHO Region', fontsize=14, fontweight='bold')
plt.show()
df4.head()
incidence.head()
df6 =reports.groupby('Year').sum().loc[:, ['No. of cases']]
plt.figure(figsize =(8,5))
ax = sns.lineplot(data=df6)
plt.xlabel('Year', fontsize=15)
plt.title('Malaria death and cases over the years', fontsize=14, fontweight='bold')
plt.show()