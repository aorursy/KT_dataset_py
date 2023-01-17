# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/survey.csv')
df.head()
df.info()
df['Age'] = pd.to_numeric(df['Age'],errors='coerce')

def age_process(age):

    if age>=0 and age<=100:

        return age

    else:

        return np.nan

df['Age'] = df['Age'].apply(age_process)
fig,ax = plt.subplots(figsize=(8,6))

sns.distplot(df['Age'].dropna(),ax=ax,kde=False,color='#ffa726')

plt.title('Age Distribution')

plt.ylabel('Freq')
country_count = Counter(df['Country'].dropna().tolist()).most_common(10)

country_idx = [country[0] for country in country_count]

country_val = [country[1] for country in country_count]

fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x = country_idx,y=country_val ,ax =ax)

plt.title('Top ten country')

plt.xlabel('Country')

plt.ylabel('Count')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)
df['Timestamp'] = pd.to_datetime(df['Timestamp'],format='%Y-%m-%d')

df['Year'] = df['Timestamp'].apply(lambda x:x.year)
sns.countplot(df['treatment'])

plt.title('Treatement Distribution')
df['Age_Group'] = pd.cut(df['Age'].dropna(),

                         [0,18,25,35,45,99],

                         labels=['<18','18-24','25-34','35-44','45+'])
fig,ax = plt.subplots(figsize=(8,6))

sns.countplot(data=df,x = 'Age_Group',hue= 'family_history',ax=ax)

plt.title('Age vs family_history')
fig,ax =plt.subplots(figsize=(8,6))

sns.countplot(data = df,x = 'Age_Group', hue='treatment')

plt.title('Age Group vs Treatment')
fig,ax  =plt.subplots(figsize=(8,6))

sns.countplot(df['work_interfere'].dropna(),ax=ax)

plt.title('Work interfere Distribution')

plt.ylabel('Count')
fig,ax = plt.subplots(figsize=(8,6))

total = df['no_employees'].dropna().shape[0] * 1.0

employee_count  = Counter(df['no_employees'].dropna().tolist())

for key,val in employee_count.items():

    employee_count[key] = employee_count[key] / total

employee_group = np.asarray(list(employee_count.keys()))

employee_val = np.asarray(list(employee_count.values()))

sns.barplot(x = employee_group , y = employee_val)

plt.title('employee group ratio')

plt.ylabel('ratio')

plt.xlabel('employee group')
fig,ax = plt.subplots(figsize=(8,6))

sns.countplot(data = df,x = 'no_employees', hue ='tech_company',ax=ax )

ticks = plt.setp(ax.get_xticklabels(),rotation=45)

plt.title('no_employee vs tech_company')
fig,ax = plt.subplots(figsize=(8,6))

sns.countplot(data = df,x = 'no_employees', hue ='remote_work',ax=ax )

ticks = plt.setp(ax.get_xticklabels(),rotation=45)

plt.title('no_employee vs remote_work')