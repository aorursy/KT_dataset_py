# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import geopandas as gpd

import seaborn as sns

sns.set()





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Sp_missions_df=pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')
Sp_missions_df.head()
Sp_missions_df.info()
Sp_missions_df=Sp_missions_df.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
Sp_missions_df['Year']=pd.to_datetime(Sp_missions_df['Datum']).apply(lambda x : x.year)

Sp_missions_df['Month']=pd.to_datetime(Sp_missions_df['Datum']).apply(lambda x : x.month)

Sp_missions_df['Day']=pd.to_datetime(Sp_missions_df['Datum']).apply(lambda x : x.day)
Sp_missions_df['Country']=Sp_missions_df['Location'].apply(lambda x : x.strip().split(",")[-1])
Sp_missions_df.info()
fig, ax= plt.subplots(figsize =(12,8))

splot=sns.countplot(data=Sp_missions_df,x='Status Mission')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.title('Status of mission from 1957 onwards')

plt.xlabel("Mission status")

plt.show()
Details=Sp_missions_df.groupby('Status Mission')['Detail'].count()

List_Mission_status=['Failure','Partial Failure','Prelaunch Failure','Success']
fig, ax= plt.subplots(figsize =(12,20))

plt.pie(Details,labels=List_Mission_status,autopct='%1.2f%%')    

plt.show()
fig, ax= plt.subplots(figsize =(12,8))

cplot=sns.countplot(data=Sp_missions_df,x='Status Rocket')

for p in cplot.patches:

    cplot.annotate(format(p.get_height(),'.0f'),(p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('Roceket status used for mission from  1957 onwards')

plt.show()
RocketDetails=Sp_missions_df.groupby('Status Rocket')['Detail'].count()

List_Mission_status=['StatusActive','StatusRetired']

fig, ax= plt.subplots(figsize =(12,20))

plt.pie(RocketDetails,labels=List_Mission_status,autopct='%1.2f%%')    

plt.show()
fig, ax= plt.subplots(figsize =(20,20))

cplot=sns.countplot(data=Sp_missions_df,x='Status Rocket',hue='Country')

for p in cplot.patches:

    cplot.annotate(format(p.get_height(),'.0f'),(p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('Roceket status used for mission from  1957 onwards country wise')

plt.xlabel('Rocket status')

plt.show()
fig, ax= plt.subplots(figsize =(20,20))

cplot=sns.countplot(data=Sp_missions_df,y='Company Name')

for p in cplot.patches:

    cplot.annotate(format(p.get_width(),'.0f'),(p.get_x() + p.get_width()+25, p.get_y()+0.90), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.title('Number of missions by each organizaation ')

plt.xlabel("Number of missions")

plt.ylabel("Organization names")

plt.show()
fig, ax= plt.subplots(figsize =(20,20))

cplot=sns.countplot(data=Sp_missions_df,y='Year',color='#93d498')

for p in cplot.patches:

    cplot.annotate(format(p.get_width(),'.0f'),(p.get_x() + p.get_width()+3, p.get_y()+0.90), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('Space missions eah year')

plt.ylabel("Year")

plt.xlabel('Number of space missions')

plt.show()
Sp_missions_df_1971=Sp_missions_df[Sp_missions_df['Year']==1971]
Sp_missions_df_1971
fig, ax= plt.subplots(figsize =(20,20))

cplot=sns.countplot(data=Sp_missions_df_1971,y='Country')

for p in cplot.patches:

    cplot.annotate(format(p.get_width(),'.0f'),(p.get_x() + p.get_width()+2, p.get_y()+0.5), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('Nation wise space missions during 1971')

plt.ylabel('Country Name')

plt.xlabel('Number of missions')

plt.show()
countries_list = list()

frequency_list = list()

test = Sp_missions_df.groupby("Country")["Company Name"].unique()

for i in test.iteritems():

    countries_list.append(i[0])

    frequency_list.append(len(i[1]))

    

companies = pd.DataFrame(list(zip(countries_list, frequency_list)), columns =['Country', 'Company Number'])

companies = companies.sort_values("Company Number", ascending=False)

companies
fig,ax=plt.subplots(figsize=(12,12))

barplot=sns.barplot(data=companies,y='Country',x='Company Number')

plt.ylabel("Country Name")

plt.xlabel('Number of Companies or Space organizations')

plt.show()
countries_list = list()

frequency_list = list()

test = Sp_missions_df.groupby("Country")["Detail"].unique()

for i in test.iteritems():

    countries_list.append(i[0])

    frequency_list.append(len(i[1]))

    

companies_misiion_count = pd.DataFrame(list(zip(countries_list, frequency_list)), columns =['Country', 'Space Mission count'])

companies_misiion_count = companies_misiion_count.sort_values("Space Mission count", ascending=False)

companies_misiion_count
fig,ax=plt.subplots(figsize=(12,12))

barplot=sns.barplot(data=companies_misiion_count,y='Country',x='Space Mission count')

plt.title('Space missions country wise')

plt.ylabel('Country Name')

plt.xlabel('Number of space missions')

plt.show()