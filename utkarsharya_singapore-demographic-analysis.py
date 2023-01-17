import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data1 = pd.read_csv('../input/singapore-res-data/singapore-residents-by-age-group-ethnic-group-and-sex-end-june-annual.csv',na_values=['na'])
data1.head()
data1['level_1'].unique()
data1['level_2'].unique()
plt.figure(figsize=(15,10))
sns.heatmap(data1.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data2 = pd.DataFrame(data1.groupby(['year','level_1','level_2'])['value'].mean())
data2.head()
def impute_value(cols):
    
    value = cols[0]
    year = cols[1]
    level_1 = cols[2]
    level_2 = cols[3]
    
    for i in range(len(data1)):
        if pd.isnull(value):
            a = data1['year'].iloc[i]
            b = data1['level_1'].iloc[i]
            c = data1['level_2'].iloc[i]
            
            return data2.loc[a].loc[b].loc[c]
        else:
            return value

data1['value'] = data1[['value','year','level_1','level_2']].apply(impute_value,axis=1)
data1.head()
plt.figure(figsize=(15,10))
sns.heatmap(data1.isnull(),yticklabels=False,cbar=False,cmap='viridis')
by_ethn = pd.DataFrame(data1.groupby('level_1')['value'].sum())
by_ethn.transpose()
print('Maximum population by ethnicity are of Chinese over the years from 1957 to 2018 : ',by_ethn.loc['Total Chinese'])
pop_growth = pd.DataFrame(data1.groupby(['year','level_1'])['value'].sum())
pop_growth.head()
a = pop_growth.loc[2018].loc['Total Chinese']
b = pop_growth.loc[1957].loc['Total Chinese']
avg_growth = (a-b)/(2018-1957)
print('The average Chinese population growth over the years : ',avg_growth)
a = pop_growth.loc[2018].loc['Total Chinese']
b = pop_growth.loc[2018].loc['Total Residents']
print('Portion of total population constituted by the Chinese : ',100*(a/b))
by_age = pd.DataFrame(data1.groupby('level_2')['value'].sum())
by_age.transpose()
print('The largest age group of population over the years 1957 to 2018 is 80-84 years and their total population over the years :',by_age.loc['80 - 84 Years'])
pop_age = pd.DataFrame(data1.groupby(['year','level_2'])['value'].sum())
pop_age.head()
a = pop_age.loc[2018].loc['80 - 84 Years']
b = pop_age.loc[1957].loc['80 - 84 Years']
avg_growth = (a-b)/(2018-1957)
print('The average population growth of 80 - 84 Years age group is : ',avg_growth)
pop_age.loc[2018].transpose()
a = pop_age.loc[2018].loc['65 Years & Over']
b = pop_age.loc[2018]['value'].sum()
c = pop_age.loc[1957].loc['65 Years & Over']
print('The age group-65-70 years becomes largest populated in 2018 with population :',a)
print('The population constituted by 65-70 year age group is : ',100*(a/b))
print('Their population growth over the years : ',(a-c)/(2018-1957))
a = pop_age.loc[2018]
b = pop_age.loc[1957]
df1 = []
for i in range(len(pop_age.loc[2018])):
    c = (a['value'].iloc[i] - b['value'].iloc[i])/(2018-1957)
    df1.append(c)

df1 = pd.DataFrame({'Growth Rate (person per year)':df1})
df2 = list(data1['level_2'].unique())
df2 = pd.DataFrame({'Age Group':df2})
df_age = df2.join(df1)
df_age
print('Highest growth rate of population by age group is of 70 - 74 Years age group with growth rate of : 33888.06')
print('Lowest growth rate or depreciating growth rate is of 0 - 4 Years age group with depreciating rate of : -5193.37 ')
print('The group which almost stagnated with little depreciating rate is of 45 - 49 Years with rate of : -1247.93')
a = pop_growth.loc[2018]
b = pop_growth.loc[1957]
df1 = []
for i in range(len(pop_growth.loc[2018])):
    c = (a['value'].iloc[i] - b['value'].iloc[i])/(2018-1957)
    df1.append(c)

df1 = pd.DataFrame({'Growth Rate (person per year)':df1})
df2 = list(data1['level_1'].unique())
df2 = pd.DataFrame({'Ethnic Group':df2})
df_ethn = df2.join(df1)
df_ethn
print('Highest growth rate of population by ethnic group is Malays with growth rate of : 46527.50')
print('Lowest growth rate is of Chinese ethnic group with growth rate of :  3762.91')
print('The group which almost stagnated with little growth rate is of chinese ethnic group with rate of : 3762.91')
print('Highest growth rate of population by gender is of females with growth rate of : 1775.31')
print('Lowest growth rate by gender are of males with growth rate of : 800.29 ')
plt.figure(figsize=(30,20))
sns.barplot('Ethnic Group','Growth Rate (person per year)',data=df_ethn)
plt.figure(figsize=(30,20))
sns.barplot('Age Group','Growth Rate (person per year)',data=df_age)