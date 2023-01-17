# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv')

df.head(10)
df.info()
df['low_bound_sal'] = df['Salary Estimate'].apply(lambda x : (x.split()[0]).split('-')[0] if x != '-1' else x)

df['upp_bound_sal'] = df['Salary Estimate'].apply(lambda x : (x.split()[0]).split('-')[1] if x != '-1' else x)



df['upp_bound_sal'] = df['upp_bound_sal'].apply(lambda x : x[1:-1] if x != '-1' else x)

df['low_bound_sal'] = df['low_bound_sal'].apply(lambda x : x[1:-1] if x != '-1' else x)



df['upp_bound_sal'] = df['upp_bound_sal'].apply(pd.to_numeric)

df['low_bound_sal'] = df['low_bound_sal'].apply(pd.to_numeric)



df['median_sal'] = (df['upp_bound_sal'] + df['low_bound_sal'])/2 * 1000
sns.set(font_scale=1.2)

sns.catplot(y='Sector', x='median_sal', kind='box' ,data=df, height=7, aspect=2, \

           order=df.groupby('Sector')['median_sal'].median().sort_values(ascending=False).index)

plt.title('Distribution of median salaries by Sector',fontsize='20')

plt.xticks(rotation=90)
sns.set(font_scale=1.2)

sns.catplot(y='Industry', x='median_sal', kind='box', data=df, height=15, aspect=1, order=df.groupby('Industry')['median_sal'].median().sort_values(ascending=False).index)

plt.title('Distribution of median salaries by Industry', fontsize='20')

plt.xticks(rotation=90)
plt.figure(figsize=(12, 7))

sns.set(font_scale=1.2)

sns.countplot(y='Sector', data=df, order=df['Sector'].value_counts().index)

plt.title('Counts of job postings by Sector', fontsize='18')

plt.xticks(rotation=90)
plt.figure(figsize=(7, 18))

sns.set(font_scale=1.2)

sns.countplot(y='Industry', data=df, order=df['Industry'].value_counts().index)

plt.title('Counts of job postings by Industry', fontsize='16')

plt.xticks(rotation=90)
count = df['Location'].nunique()



print(f'There are {count} unique cities')
sns.set(font_scale=1.2)

sns.catplot(y='Location', x='median_sal', kind='box', data=df, height=10, aspect=1, order=df.groupby('Location')['median_sal'].median().sort_values(ascending=False).iloc[:25].index)



plt.title('Top 25 median salaries by Location', fontsize='18')

plt.xticks(rotation=90)
sns.set(font_scale=1.2)

sns.catplot(y='Location', x='median_sal', kind='box', data=df, height=10, aspect=1, order=df.groupby('Location')['median_sal'].median().sort_values(ascending=False).iloc[-25:].index)



plt.title('Bottom 25 median salaries by Location', fontsize='18')

plt.xticks(rotation=90)
# create the state column

df['state'] = df['Location'].apply(lambda x : x.split(',')[-1])
count = df['state'].nunique()

print(f'There are {count} states')
sns.set(font_scale=1.2)

sns.catplot(y='state', x='median_sal', kind='box', data=df, height=10, aspect=1, order=df.groupby('state')['median_sal'].median().sort_values(ascending=False).index)



plt.title('Salaries by State', fontsize='18')

plt.xticks(rotation=90)
df['Age'] = df['Founded'].apply(lambda x : (2020 - x) if x != -1 else x)
sns.set(font_scale=1.2)

sns.set_palette("Paired")

plt.figure(figsize=(15, 7))

sns.scatterplot(x='Age', y='median_sal', data=df[df['Age'] != 1], hue='state')



plt.title('Salaries by Company Age', fontsize='18')

plt.xticks(rotation=90)
# remove unknown and -1 company size

df_filtered = df.loc[(df['Size'] != '-1') & (df['Size'] != 'Unknown')]



sns.set(font_scale=1.2)

sns.catplot(y='Size', x='median_sal', kind='box', data=df_filtered, height=10, aspect=1,\

           order=df_filtered.groupby('Size')['median_sal'].median().sort_values(ascending=False).index)



plt.title('Salaries by Company Size', fontsize='18')

plt.xticks(rotation=90)
def extract_seniority(t):

    t  = t.lower()

    if 'senior' in t:

        return 'Senior'

    elif 'manager' in t:

        return 'Manager'

    elif 'lead' in t:

        return 'Lead'

    elif 'principal' in t:

        return 'Principal'

    else:

        return 'Everyone Else'

    



df['Level'] = df['Job Title'].apply(extract_seniority)



sns.set(font_scale=1.2)

sns.catplot(y='Level', x='median_sal', kind='box', data=df, height=10, aspect=1, \

            order=df.groupby('Level')['median_sal'].median().sort_values(ascending=False).index)



plt.title('Salaries by Seniority Levels', fontsize='18')

plt.xticks(rotation=90)
df[df['Job Title'].str.contains('principal', case=False)].tail(20)