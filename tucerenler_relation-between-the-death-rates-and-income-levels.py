#import the libraries 

import pandas as pd

import requests

import io

import numpy as np

import os

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
#Get the list of all files and directories in current working directory

l = os.listdir(os.getcwd())



l = [i for i in l] 

l
df = pd.read_csv('../input/externalourworldindata/number-of-deaths-by-risk-factor.csv') #number of deaths

df2=pd.read_csv('../input/project2datasets/worldpopulation.csv', skiprows=3) #world population

df3 = pd.read_excel('../input/project2datasets/OGHIST-2.xls',sheet_name=2).iloc[6:-2,1:] #income levels

df.size
#Let's see how the dataframes look like

df.head()  #it shows countries deaths number by risk factor between 1990-2017
df2.head() #Second data shows world population between 1960-2019
df3.head() #income levels
#For the first and third dataframe, change the 'Entity' and 'Bank's fiscal year'  columns name to 'Country Name'

df=df.rename(columns={"Entity":"Country Name"})

df3=df3.rename(columns={"Bank's fiscal year:":"Country Name"})
#see the full summary of df

df.info()
df2.info()
df2=df2.drop(['Country Code','Indicator Name','Indicator Code','Unnamed: 64','2019'],axis=1)

df2.head()
df2_tidy = pd.melt(df2,['Country Name'], var_name="Year", value_name="Population")

df2_tidy.head()
#First,convert data types to integer

df2_tidy['Year'] = df2_tidy['Year'].astype(int)

df2_tidy['Population']=df2_tidy['Population'].astype('Int64')

df2_tidy.head()
#mask the other years

start_year=1990

end_year=2017

mask = (df2_tidy['Year'] >= start_year) & (df2_tidy['Year'] <= end_year)

df2_population = df2_tidy.loc[mask]

df2_population.head()
#get the info

df2_population.info()
#see which rows have missing values in population column

df2_miss = df2_population[df2_population.isna().any(axis=1)]

#see if the entire population column is emtpy

df2_population.isnull().sum()==df2_population.shape[0]
#change the columns year format range the calender date

cols = list(range(1987,2019))

cols.insert(0,'Country Name')

df3.columns = cols

df3.head()
df3.drop(df3[[1987,1988,1989,2018]],axis=1,inplace=True)
#Make the data tidy

df3_income = pd.melt(df3,["Country Name"], var_name="Year", value_name="Income Level")

df3_income.head()
#Merge the dataframe with population dataframe

df_merge1= pd.merge(df,df2_population, how= 'inner',on=['Year','Country Name'])

df_merge1.head()
#Merge the income dataframe

final_merge = pd.merge(df_merge1,df3_income,how='inner',on=['Year','Country Name'])

final_merge.head()
final_merge.info()
#Create a lag plot to see if the order of this data matters 

pd.plotting.lag_plot(final_merge['High cholesterol (deaths)'],c='orange')

#interpolate the missing values

final_merge['High cholesterol (deaths)']= final_merge['High cholesterol (deaths)'].interpolate()

final_merge.isnull().sum() #count the number of missing values per column
final_merge[final_merge['Population'].isnull()]
#fill the place with value in the previous respectively.

final_merge['Population']=final_merge['Population'].fillna(method='bfill') 

final_merge.head()
df_final=final_merge.copy()

cols=df_final.columns.tolist()
new_cols=cols[3:38] #choose the columns that have risk factors
#Calculate the death rate with loop

for i in new_cols:

    df_final[i]=(df_final[i]/df_final['Population'])*100000

df_final.head()    
cols = [col for col in df_final.columns if col not in ['Country Name', 'Code',

                                                           'Population','Income Level']]

for col in cols:

    df_final[col] = df_final[col].astype(int)
df_final.head()
df_anlyz = df_final.copy()

deaths = df_anlyz.iloc[:,3:-2] #display the columns which have number of deaths by risk factor.

deaths.head()
#calculate the total deaths number and create 'Total Deaths Number' columns

deaths.loc[:,'Total Death Rate'] = deaths.sum(axis=1) 

deaths.loc[:,'Income Level'] = df_anlyz['Income Level'] 

deaths.loc[:,'Year'] = df_anlyz['Year']

deaths = deaths[deaths['Income Level']!='..'] 

deaths.describe()
#create a line plot of total deaths number over years

fig = plt.figure(figsize=(10,7))

#specify the column value for the hue parameter

sns.lineplot(x='Year',y='Total Death Rate',data=deaths,hue='Income Level') 

plt.title('Total Death Rate for Income Level',fontsize=15)

plt.ylim(500,3000)
group = deaths.groupby(['Income Level']).sum() #group by income levels
group = group.reset_index()

group
labels = list(group['Income Level'])

sizes = list(group['Total Death Rate']/(1e5))

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice 



fig1, ax1 = plt.subplots(figsize=(10,7))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, colors=colors)

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.suptitle('Total Deaths Share by Income Level',color='k',fontsize=18)

plt.show()
features = list(df_anlyz.iloc[:,3:-3])

f,axs = plt.subplots(6,6,figsize=(20,10))

axes = axs.flatten().tolist()

axes = axes[:-2]

import random

random.seed(42)

colors = []

for i in range(len(axes)):

    x=random.uniform(0, 1)

    y=random.uniform(0, 1)

    z=random.uniform(0, 1)

    colors.append((x,y,z))

for i,j in enumerate(axes):

    sns.lineplot(x='Year',y=features[i],data=deaths,ax=j,label=features[i],color=colors[i])

    j.grid(True)

    j.set_ylabel('') 

axs[5,4].set_axis_off()

axs[5,5].set_axis_off()

fig.subplots_adjust(wspace=0.5,hspace=0.5)

fig.tight_layout()
final_copy=df_final.copy()

final_copy.head()
#display the columns we need

final_copy.set_index(['Country Name','Year'], inplace=True)

eth_usa=final_copy.loc[['Ethiopia','United States'],['Population','Obesity (deaths)','Income Level']]

eth_usa.head()
eth = eth_usa.reset_index()[eth_usa.reset_index()['Country Name'].str.startswith('E')]

usa = eth_usa.reset_index()[eth_usa.reset_index()['Country Name'].str.startswith('U')]

eth_usa_last=pd.merge(eth,usa,on='Year')

eth_usa_last.head()
#Calculation the ratio and see the result in a new column 

eth_usa_last['Ratio'] = (eth_usa_last['Obesity (deaths)_y'] / eth_usa_last['Obesity (deaths)_x']).astype(int)
eth_usa_last
eth_usa_last.describe()
ax = sns.barplot(x="Year", y="Ratio", data=eth_usa_last)

plt.ylabel

plt.title('Ratio of Death rates in Usa to Ethiopia (Obesity)')

plt.xticks(rotation=60)
finalcopy=df_final.copy()
finalcopy.set_index(["Country Name"], inplace=True)

usa_drugs=finalcopy.loc[['United States'],['Year' ,'Population','Drug use (deaths)']]

strt=2009

usa_drugs=usa_drugs.loc[(usa_drugs["Year"] >= strt)] #display the year 2009 and after

usa_drugs
usa_drugs["Percentage of Difference"] = usa_drugs["Drug use (deaths)"].pct_change(axis=0,fill_method='bfill')*100

usa_drugs
#The first row contains NaN values, as there is no previous row from which we can calculate the change.

#the NaN values in the dataframe has been filled using fillna method.

usa_drugs=usa_drugs.fillna(0) 
usa_drugs['Percentage of Difference'] = usa_drugs['Percentage of Difference'].astype(int)
usa_drugs
#Generate descriptive statistics

usa_drugs.describe()