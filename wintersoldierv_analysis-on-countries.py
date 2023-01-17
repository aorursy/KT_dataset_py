# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
# Importing the dataset
df=pd.read_csv('../input/countries-of-the-world/countries of the world.csv')
#viewing the data
df.head()


#shape
df.shape 
df['Country'].shape[0]
df.isnull().sum()
df.describe()
df.sample(10)
df.info()
# Listing down the columns
df.columns.values


# creating copy of the original dataset
df_copy=df.copy()
df_copy.shape

df_copy

df_copy['Pop. Density (per sq. mi.)']=df_copy['Pop. Density (per sq. mi.)'].astype('str')
df_copy['Coastline (coast/area ratio)']=df_copy['Coastline (coast/area ratio)'].astype('str')
df_copy['Net migration']=df_copy['Net migration'].astype('str')
df_copy['Infant mortality (per 1000 births)']=df_copy['Infant mortality (per 1000 births)'].astype('str')
df_copy['Literacy (%)']=df_copy['Literacy (%)'].astype('str')
df_copy['Phones (per 1000)']=df_copy['Phones (per 1000)'].astype('str')
df_copy['Arable (%)']=df_copy['Arable (%)'].astype('str')
df_copy['Crops (%)']=df_copy['Crops (%)'].astype('str')
df_copy['Other (%)']=df_copy['Other (%)'].astype('str')
df_copy['Climate']=df_copy['Climate'].astype('str')
df_copy['Birthrate']=df_copy['Birthrate'].astype('str')
df_copy['Deathrate']=df_copy['Deathrate'].astype('str')
df_copy['Agriculture']=df_copy['Agriculture'].astype('str')
df_copy['Industry']=df_copy['Industry'].astype('str')
df_copy['Service']=df_copy['Service'].astype('str')
df_copy['Pop. Density (per sq. mi.)']=df_copy['Pop. Density (per sq. mi.)'].str.replace(',','.')
df_copy['Coastline (coast/area ratio)']=df_copy['Coastline (coast/area ratio)'].str.replace(',','.')
df_copy['Net migration']=df_copy['Net migration'].str.replace(',','.')
df_copy['Infant mortality (per 1000 births)']=df_copy['Infant mortality (per 1000 births)'].str.replace(',','.')
df_copy['Literacy (%)']=df_copy['Literacy (%)'].str.replace(',','.')
df_copy['Phones (per 1000)']=df_copy['Phones (per 1000)'].str.replace(',','.')
df_copy['Arable (%)']=df_copy['Arable (%)'].str.replace(',','.')
df_copy['Crops (%)']=df_copy['Crops (%)'].str.replace(',','.')
df_copy['Other (%)']=df_copy['Other (%)'].str.replace(',','.')
df_copy['Climate']=df_copy['Climate'].str.replace(',','.')
df_copy['Birthrate']=df_copy['Birthrate'].str.replace(',','.')
df_copy['Deathrate']=df_copy['Deathrate'].str.replace(',','.')
df_copy['Agriculture']=df_copy['Agriculture'].str.replace(',','.')
df_copy['Industry']=df_copy['Industry'].str.replace(',','.')
df_copy['Service']=df_copy['Service'].str.replace(',','.')
df_copy

df_copy['Area (sq. mi.)']=df_copy['Area (sq. mi.)'].astype('float')
df_copy['Pop. Density (per sq. mi.)']=df_copy['Pop. Density (per sq. mi.)'].astype('float')
df_copy['Coastline (coast/area ratio)']=df_copy['Coastline (coast/area ratio)'].astype('float')
df_copy['Net migration']=df_copy['Net migration'].astype('float')
df_copy['Infant mortality (per 1000 births)']=df_copy['Infant mortality (per 1000 births)'].astype('float')
df_copy['Literacy (%)']=df_copy['Literacy (%)'].astype('float')
df_copy['Phones (per 1000)']=df_copy['Phones (per 1000)'].astype('float')
df_copy['Arable (%)']=df_copy['Arable (%)'].astype('float')
df_copy['Crops (%)']=df_copy['Crops (%)'].astype('float')
df_copy['Other (%)']=df_copy['Other (%)'].astype('float')
df_copy['Climate']=df_copy['Climate'].astype('float')
df_copy['Birthrate']=df_copy['Birthrate'].astype('float')
df_copy['Deathrate']=df_copy['Deathrate'].astype('float')
df_copy['Agriculture']=df_copy['Agriculture'].astype('float')
df_copy['Industry']=df_copy['Industry'].astype('float')
df_copy['Service']=df_copy['Service'].astype('float')
df_copy.info()

df_copy[df_copy['Net migration'].isnull()]
#Taking Net migration rate:
#4.9 migrant(s)/1,000 population (2020 est.) for NORTHERN AFRICA region
#according to data from the web
#The net migration rate is estimated at -21.1 per 1,000 population for OCEANIA region
df_copy['Net migration'].loc[[47,221]]=df_copy['Population'].loc[[47,221]]/1000*-21.1
df_copy['Net migration'].loc[223]=df_copy['Population'].loc[223]/1000*4.9
df_copy[df_copy['Net migration'].isnull()]
df_copy[df_copy['Infant mortality (per 1000 births)'].isnull()]
#As we can see index no.221 & 223 has so many nan values thus having insufficient data to contribute to the analysis we can drop them
df_copy.drop(df_copy.index[[221,223]],inplace=True)
df_copy[df_copy['Infant mortality (per 1000 births)'].isnull()]
#total: 12.6 deaths/1,000 live births (2018 est.)
#male: 15.3 deaths/1,000 live births
#female: 9.8 deaths/1,000 live births
#src- 'https://www.indexmundi.com/cook_islands/infant_mortality_rate.html'
df_copy['Infant mortality (per 1000 births)'].loc[47]=12.6
df_copy[df_copy['Infant mortality (per 1000 births)'].isnull()]
#issue already handled
df_copy.isnull().sum()
df_copy[df_copy['GDP ($ per capita)'].isnull()]
df_copy[df_copy['Literacy (%)'].isnull()]
mean_l=df_copy['Literacy (%)'].mean()
df_copy['Literacy (%)']=df_copy['Literacy (%)'].fillna(mean_l)
df_copy[df_copy['Literacy (%)'].isnull()]
df_copy[df_copy['Phones (per 1000)'].isnull()]
mean_l=df_copy['Phones (per 1000)'].mean()
mean_l
#if we consider 80% of the population has phone then 
df_copy['Phones (per 1000)'].loc[[52,58,140]]=df_copy['Population'].loc[[52,58,140]]*80/100000
df_copy.loc[58]
df_copy[df_copy['Phones (per 1000)'].isnull()]
df_copy.sample(10)
df_copy[df_copy['Arable (%)'].isnull()]
df_copy[df_copy['Other (%)'].isnull()]
df_copy[df_copy['Crops (%)'].isnull()]
df_copy['Crops (%)'].loc[[85,134]]=df_copy['Crops (%)'].quantile(.40)
df_copy['Arable (%)'].loc[[85,134]]=df_copy['Arable (%)'].quantile(.40)
df_copy['Other (%)'].loc[[85,134]]=100-(df_copy['Crops (%)'].loc[[85,134]]+df_copy['Arable (%)'].loc[[85,134]])
df_copy[df_copy['Arable (%)'].isnull()]
df_copy[df_copy['Crops (%)'].isnull()]
df_copy[df_copy['Other (%)'].isnull()]
df_copy[df_copy['Climate'].isnull()]
df_copy['Climate'].fillna(round(df_copy['Climate'].mean()),inplace=True)
df_copy[df_copy['Climate'].isnull()]
df_copy[df_copy['Birthrate'].isnull()]
#Serbia Birth rate. Birth rate: 8.9 births/1,000 population (2018 est.)
#as per web.

df_copy['Birthrate'].fillna(8.9,inplace=True)
#Death rate, crude (per 1000 people) in Serbia was reported at 14.8 in 2017, according to the World Bank collection of development indicators
#as per web

df_copy['Deathrate'].fillna(14.8,inplace=True)
df_copy.loc[181]
df_copy[df_copy['Birthrate'].isnull()]
df_copy[df_copy['Deathrate'].isnull()]
df_copy.sample(10)
df_copy['Agriculture']=df_copy['Agriculture']*1000
df_copy['Industry']=df_copy['Industry']*1000
df_copy['Service']=df_copy['Service']*1000
df_copy.sample(10)
df_copy[df_copy['Agriculture'].isnull()]
df_copy['Agriculture'].fillna(round(df_copy['Agriculture'].mean(),2),inplace=True)
df_copy['Industry'].fillna(round(df_copy['Industry'].quantile(.65),2),inplace=True)
df_copy['Service'].fillna(round(1000-(df_copy['Industry'].quantile(.65)+df_copy['Agriculture'].mean()),2),inplace=True)
df_copy['Agriculture(%)']=round(df_copy['Agriculture']/10,2)
df_copy['Industry(%)']=round(df_copy['Industry']/10,2)
df_copy['Service(%)']=round(df_copy['Service']/10,2)
df_copy['Literacy (%)']=round(df_copy['Literacy (%)'],2)
df_copy['Phones (per 1000)']=round(df_copy['Phones (per 1000)'],2)
df_copy['Arable (%)']=round(df_copy['Arable (%)'],2)
df_copy['Crops (%)']=round(df_copy['Crops (%)'],2)
df_copy['Other (%)']=round(df_copy['Other (%)'],2)
df_copy
df_copy[df_copy['Agriculture'].isnull()]
df_copy[df_copy['Industry'].isnull()]
df_copy[df_copy['Service'].isnull()]
df_copy.drop(columns='Climate',inplace=True)
df_copy.columns

#Agriculture,Industry,Service columns values are not clear[Accuracy]
#Problem already solved above
df_copy['Country'].nunique() #no same name is repeated
df_copy['Country'].tolist() #an extra space is added with the names at the end 
Country_name=[]

for name in df_copy['Country']:
    n=name.strip()
    Country_name.append(n)
df_copy['Country']=Country_name
df_copy['Country'].tolist() #space after name string is thus removed 
#The data contains both country and island and also inflagged countries hence the total number of countries is 225 after dropping 2 countries of less data
df_copy[df_copy.duplicated(subset=['Country'])] #No duplicated country data found
df_copy.sample(10)
df_copy.info() #No nan values
# converting population col in million(m)
df_copy.rename(columns={'Population':'Population(million)'},inplace=True)
df_copy['Population(million)']=df_copy['Population(million)']/1000000
df_copy['Population(million)']=round(df_copy['Population(million)'],2)
df_copy
df_copy.drop(columns=['Agriculture','Industry','Service'],inplace=True)
df_copy.rename(columns={'Birthrate':'Birth Rate'},inplace=True)
df_copy.rename(columns={'Deathrate':'Death Rate'},inplace=True)
df_copy
df_copy.describe()
#All the issues are already solved 
df_new=df_copy.copy()
df_new
df_copy.shape
df_copy.head()
df_copy.columns.values
df_copy.info()
df_copy.isnull().sum() #no  null value
# Let's start with the Region col

plt.figure(figsize=(30,8))
sns.countplot(df_copy['Region'])
plt.show()

#conclusions

#1.sub saharan region has most countries
#2.balitics region has least countries 
#for Population(million) column

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Population(million)'])
plt.show()
plt.figure(figsize=(20,8))
sns.boxplot(df_copy['Population(million)'])
plt.show()
df_copy[df_copy['Population(million)']>1000]
#Conclusion
#0. graph is left skewed
#1. China and India are outliers
#2. The dropping of the rows may not be a good idea because they hold real and valid data 
#3. But this can hamper mathematical calculations, hence replacing the outliers populations with std of population 

#as the data is based on real world we should not hamper it hence will create another column to keep manipulated data 
round(df_copy['Population(million)'].std(),2)
#df_copy['Population(million)_M']=df_copy['Population(million)'].copy()
#replacing with std to get better mathematical data
#but again as the data is real rinning the code below is not recommended 
#df_copy['Population(million)_M'].loc[[42,94]]=round(df_copy['Population(million)'].std(),2)
#changing the population will also hamper the population density 
#re-calculating population density for india and china 
#but again as the data is real rinning the code below is not recommended 
#df_copy['Pop. Density (per sq. mi.)_M']=df_copy['Pop. Density (per sq. mi.)'].copy()
#df_copy['Pop. Density (per sq. mi.)_M'].loc[[42,94]]=round((df_copy['Population(million)_M']*1000000)/df_copy['Area (sq. mi.)'],2)
#df_copy.loc[42]
#for Area (sq. mi.) column

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Area (sq. mi.)'])
plt.show()
plt.figure(figsize=(20,10))
sns.boxplot(df_copy['Area (sq. mi.)'])
plt.show()
df_copy[df_copy['Area (sq. mi.)']>10000000]
#Conclusions

#1. graph is left skewed
#2. Russia is outlier, it has a big area
#3. The dropping of the row may not be a good idea because it hold real and valid data 
#4. But this can hamper mathematical calculations, hence replacing the outliers Area(sq. mi.) with std of Area (sq. mi.)

# therefore will be creating another column named Area (sq. mi.)_M which will contain the manipulated data for better mathematical operations and will not hamper the original data
round(df_copy['Area (sq. mi.)'].std(),2)
#but again as the data is real rinning the code below is not recommended 

#df_copy['Area (sq. mi.)_M']=df_copy['Area (sq. mi.)'].copy()
#df_copy['Area (sq. mi.)_M'].loc[169]=round(df_copy['Area (sq. mi.)'].std(),2)
#hence recalculating the population density of Russia
#but again as the data is real rinning the code below is not recommended 

#df_copy['Pop. Density (per sq. mi.)_M'].loc[169]=round((df_copy['Population(million)_M'].loc[169]*1000000)/df_copy['Area (sq. mi.)_M'].loc[169],2)
#for Pop. Density (per sq. mi.) col

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Pop. Density (per sq. mi.)'])
plt.show()
plt.figure(figsize=(20,10))
sns.boxplot(df_copy['Pop. Density (per sq. mi.)'])
plt.show()
df_copy[df_copy['Pop. Density (per sq. mi.)']>6000]
#Conclusions 

#1. Nothing can be done
#2. Hong Kong, Macau, Monaco, Singapore has high population density
#for Coastline (coast/area ratio) column 

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Coastline (coast/area ratio)'])
plt.show()
plt.figure(figsize=(20,10))
sns.boxplot(df_copy['Coastline (coast/area ratio)'])
plt.show()
df_copy[df_copy['Coastline (coast/area ratio)']>800]
#replacing with std for better mathematical calculations 
#but again as the data is real rinning the code below is not recommended 

#df_copy['Coastline (coast/area ratio)'].loc[136]=round(df_copy['Coastline (coast/area ratio)'].std(),2)#
#Conclusion

#1. Micronesia, Fed. St. has a big coastline
#for Net migration col

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Net migration'])
plt.show()
plt.figure(figsize=(30,8))
sns.boxplot(df_copy['Net migration'])
plt.show()
df_copy[df_copy['Net migration']<-400]
#replacing with std for better mathematical calculations 
#but again as the data is real rinning the code below is not recommended 

#df_copy['Net migration'].loc[47]=round(df_copy['Net migration'].std(),2) 


#Conclusion

#1. Cook Islands has high Net Migration rate 
# for Infant mortality (per 1000 births) col

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Infant mortality (per 1000 births)'])
plt.show()
plt.figure(figsize=(30,8))
sns.boxplot(df_copy['Infant mortality (per 1000 births)'])
plt.show()
df_copy[df_copy['Infant mortality (per 1000 births)']>150]
#Conclusion 

#1.  Afghanistan, Angola has high Infant mortality (per 1000 births)
# for GDP ($ per capita) col

plt.figure(figsize=(30,8))
sns.distplot(df_copy['GDP ($ per capita)'])
plt.show()
plt.figure(figsize=(30,8))
sns.boxplot(df_copy['GDP ($ per capita)'])
plt.show()
df_copy[df_copy['GDP ($ per capita)']>40000]
#Conclusion 

#1. Luxembourg is an outlier, clearly it has a very high gdp but can hamper mathematical calcutions
# for Literacy (%) col

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Literacy (%)'])
plt.show()
plt.figure(figsize=(30,8))
sns.boxplot(df_copy['Literacy (%)'])
plt.show()
df_copy[df_copy['Literacy (%)']<40]['Country'].tolist()
#Conclusion

#1. ['Afghanistan', 'Burkina Faso', 'Guinea', 'Niger', 'Sierra Leone', 'Somalia'] - countries has a very low literary(%)
# for Phones (per 1000) col

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Phones (per 1000)'])
plt.show()
plt.figure(figsize=(30,8))
sns.boxplot(df_copy['Phones (per 1000)'])
plt.show()
df_copy[df_copy['Phones (per 1000)']>900]
#Conclusion 

#1. In Monaco the usage of phone is very high
# for Arable (%) col

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Arable (%)'])
plt.show()
plt.figure(figsize=(30,8))
sns.boxplot(df_copy['Arable (%)'])
plt.show()
df_copy[df_copy['Arable (%)']>50]
df_copy[df_copy['Arable (%)']>50]['Country'].tolist()
#Conclusion

#1. ['Bangladesh', 'Denmark', 'Hungary', 'India', 'Moldova', 'Ukraine'] - these countries has a high Arable (%)
# for Crops (%) col

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Crops (%)'])
plt.show()
plt.figure(figsize=(30,8))
sns.boxplot(df_copy['Crops (%)'])
plt.show()
df_copy[df_copy['Crops (%)']>30]
df_copy[df_copy['Crops (%)']>30]['Country'].tolist()
#Conclusion

#1. ['Kiribati', 'Marshall Islands', 'Micronesia, Fed. St.', 'Sao Tome & Principe', 'Tonga'] - these countries has a high Crops(%)
# for Other (%) col

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Other (%)'])
plt.show()
plt.figure(figsize=(30,8))
sns.boxplot(df_copy['Other (%)'])
plt.show()
df_copy[df_copy['Other (%)']<40]
#nothing to conclude as such
# for Birth Rate col

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Birth Rate'])
plt.show()
plt.figure(figsize=(30,8))
sns.boxplot(df_copy['Birth Rate'])
plt.show()
#The data is good
# for Death Rate col

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Death Rate'])
plt.show()
plt.figure(figsize=(30,8))
sns.boxplot(df_copy['Death Rate'])
plt.show()
df_copy[df_copy['Death Rate']>25]
df_copy[df_copy['Death Rate']>25]['Country'].tolist()
#Conclusion

#1. ['Botswana', 'Lesotho', 'Swaziland'] these countries has a high death rate than others
# for Agriculture(%) col

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Agriculture(%)'])
plt.show()
plt.figure(figsize=(30,8))
sns.boxplot(df_copy['Agriculture(%)'])
plt.show()
df_copy[df_copy['Agriculture(%)']>60]
df_copy[df_copy['Agriculture(%)']>60]['Country'].tolist()
#Conclusion

#1. ['Guinea-Bissau', 'Liberia', 'Somalia'] these countries has high agriculture (%)
# for Industry(%) col

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Industry(%)'])
plt.show()
plt.figure(figsize=(30,8))
sns.boxplot(df_copy['Industry(%)'])
plt.show()
df_copy[df_copy['Industry(%)']>80]
df_copy[df_copy['Industry(%)']>80]['Country'].tolist()
#Conclusion

#1. ['Equatorial Guinea', 'Qatar'] these countries has higher Industry(%)
# for Service(%) col

plt.figure(figsize=(30,8))
sns.distplot(df_copy['Service(%)'])
plt.show()
plt.figure(figsize=(30,8))
sns.boxplot(df_copy['Service(%)'])
plt.show()
df_copy[df_copy['Service(%)']<20]
df_copy[df_copy['Service(%)']<20]['Country'].tolist()
#Conclusion

#1. ['Equatorial Guinea', 'Liberia', 'Qatar'] these countries has higher service (%)


#Rich Countries 
rich=df_copy['GDP ($ per capita)'].quantile(.75)
rich
poor=df_copy['GDP ($ per capita)'].quantile(.25)
poor
plt.figure(figsize=(30,10))
x=df_copy[df_copy['GDP ($ per capita)']>rich]['Region']
sns.scatterplot(x, y=df['Country'])
plt.figure(figsize=(30,10))
x=df_copy[df_copy['GDP ($ per capita)']<poor]['Region']
sns.scatterplot(x, y=df['Country'])
#Conclusion

#1. Western Europe has most of the richest countries in the world 
#2. Sub-Saharan Africa has the most poor countries in the world
df_copy.columns.values
#About More Literate Countries
df_copy['Literacy (%)'].mean()
more_literate=df_copy[df_copy['Literacy (%)']>df_copy['Literacy (%)'].mean()][['Country','Literacy (%)','Industry(%)']]
less_literate=df_copy[df_copy['Literacy (%)']<df_copy['Literacy (%)'].mean()][['Country','Literacy (%)','Industry(%)']]
more_literate
less_literate
plt.figure(figsize=(15,6))
sns.distplot(df_copy[df_copy['Literacy (%)']>df_copy['Literacy (%)'].mean()]['Industry(%)'])
sns.distplot(df_copy[df_copy['Literacy (%)']<df_copy['Literacy (%)'].mean()]['Industry(%)'])
#Conclusion 

#1. The countries with more literacy(%) has higher industry(%)
plt.figure(figsize=(15,6))
sns.distplot(df_copy[df_copy['Literacy (%)']>df_copy['Literacy (%)'].mean()]['Agriculture(%)'])
sns.distplot(df_copy[df_copy['Literacy (%)']<df_copy['Literacy (%)'].mean()]['Agriculture(%)'])
#conclusion 

#1. Countries which are little lagging behind in literacy(%) has more Agriculture(%)
