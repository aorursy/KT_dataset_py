import numpy as np

import pandas as pd

import os
# visualization



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt



# 1.3 Class for applying multiple data transformation jobs

from sklearn.compose import ColumnTransformer as ct

# 1.4 Scale numeric data

from sklearn.preprocessing import StandardScaler as ss

# 1.5 One hot encode data--Convert to dummy

from sklearn.preprocessing import OneHotEncoder as ohe

# 1.6 for data splitting

from sklearn.model_selection import train_test_split
from sklearn import linear_model

import statsmodels.api as sm
df = pd.read_csv('../input/crime.csv', encoding='ISO-8859-1')
df.shape
df.columns
df.isnull().sum()
df.columns.values
IN_vc = df.INCIDENT_NUMBER.value_counts()

IN_vc.head(10)

IN_vc.index[0]
df.loc[df.INCIDENT_NUMBER == IN_vc.index[0]].count()

df.loc[df.INCIDENT_NUMBER == IN_vc.index[0]]

#Incident number should have been unique but it is not. Intuitive?! looking closely at the top-of-list ID?

#Multiple counts(13) of offense in single event are recorded under a single Incident Number!
IN_unq = df.INCIDENT_NUMBER.nunique()

IN_unq #290156 unique INCIDENT_NUMBERs
IN_nonunq = (df.shape[0] - df.INCIDENT_NUMBER.nunique())

IN_nonunq #37664 INCIDENT_NUMBERs which are not unique
OC_vc = df.OFFENSE_CODE.value_counts()

OC_vc.head()
len(df['OFFENSE_CODE'].unique()) #222 unique offence codes
high_OC = df.loc[df.OFFENSE_CODE == OC_vc.index[0]]

#high_OC.groupby('DISTRICT').agg('sum')

high_OC.DISTRICT.value_counts() #Distribution of cases of highest offense by district

#How do other high offence cases stack-up in these districts?
sns.countplot("DISTRICT", hue="YEAR", data = df)

#Looking at how the INCIDENT_NUMBERs are distributed across DISTRICTs by year

#Notice that years 2016 and 2017 are peak years and 

#graphs for these 2 years are mostly flat
sns.countplot("DISTRICT", hue="YEAR", data = high_OC)

#Now looking at distribution of INCIDENT_NUMBERs for highest OFFENSE_CODE 

# and their distribution across DISTRICTs by year

#Notice the following:

# (i) years 2016 and 2017 are no longer the peak years and 

# (ii) graphs show a *spike* in years 2017 for high OC
df['DISTRICT'].unique()

#12 unique districts
df['REPORTING_AREA'].unique()

RA_vc=df['REPORTING_AREA'].value_counts()

len(df['REPORTING_AREA'].unique())

#880 unique reporting areas... could be useful for grouping by geographical boundary
df['OCCURRED_ON_DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE'])
df['OCCURRED_ON_DATE'].describe
df['YEAR'].unique()
df['MONTH'].unique()
df['DAY_OF_WEEK'].unique()
df['HOUR'].unique()
df['SHOOTING'].value_counts()
sns.countplot("DISTRICT", hue="SHOOTING", data = df)

#Looking at how the SHOOTINGs are distributed across DISTRICTs

#Notice that DISTRICTS B2, C11 and B3 have the highest occurence of SHOOTING
sns.catplot(x="DISTRICT",       

            hue="YEAR",      

            col="SHOOTING", 

            data=df,

            kind="count"

            )

#the above Graph now looking at occurences of SHOOTING, year-wise across DISTRICTs

#Notice the peaks in YEAR 2017. What happened sudenly and how it went down drastically in 2018
sns.catplot(x="YEAR",       

            hue="DAY_OF_WEEK",      

            col="SHOOTING", 

            data=df,

            kind="count"

            )

#looking at occurences of SHOOTING, year-wise and day-wise

#What happens suddenly on Saturdays??

#Notice the drop in shootings on Sunday in 2018. 

#Caused the drop in occurences for 2018 as observed above?
sns.catplot(x="HOUR",       

            hue="SHOOTING",      

            col="YEAR", 

            data=df,

            kind="count"

            )

#looking at occurences of SHOOTING, hourly and year-wise

#Notice the increase in late-evenings/nights shootings in 2017.

#Also notice drop in same occurences for 2018.

#So the drop in occurences of SHOOTINGs in 2018 seem to be caused by effectively reducing the

#occurences in late hours and especially on Saturdays by law enforcement authorities.
df['SHOOTING'].fillna(0, inplace = True)



df['SHOOTING'] = df['SHOOTING'].map({

    0: 0,

    'Y':1

})
#df_shooting = df.loc[df['SHOOTING'] == 1]

df_shooting = df.loc[df['SHOOTING'] == 1]

df_shooting.shape #1055 occurences of shooting should be captured here
df_loc_shooting = df_shooting[['Lat','Long']]

df_loc_shooting = df_loc_shooting.dropna()



df_loc_shooting = df_loc_shooting.loc[(df_loc_shooting['Lat']>40) & (df_loc_shooting['Long'] < -60)]  



df_shooting_x = df_loc_shooting['Long']

df_shooting_y = df_loc_shooting['Lat']



# Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”

#sns.jointplot(df_shooting_x, df_shooting_y, kind='scatter')

sns.jointplot(df_shooting_x, df_shooting_y, kind='hex')

#sns.jointplot(df_shooting_x, df_shooting_y, kind='kde')
df[['Lat','Long']].describe()
df_loc = df[['Lat','Long']]

df_loc = df_loc.dropna()



df_loc = df_loc.loc[(df_loc['Lat']>40) & (df_loc['Long'] < -60)]  
df_loc_x = df_loc['Long']

df_loc_y = df_loc['Lat']





colors = np.random.rand(len(df_loc_x))



plt.figure(figsize=(20,20))

plt.scatter(df_loc_x, df_loc_y,c=colors, alpha=0.5)

plt.show()
#### 5.3. Overlaying Lat, Long With Shooting Data
plt.figure(figsize=(30,30))

plt.scatter(df_loc_x, df_loc_y,c=colors, alpha=0.5)

plt.scatter(df_shooting_x, df_shooting_y, c='r', marker="X", alpha=1)

plt.show()

#Overlaying the scatterplot with shooting locations in Red

#The central districts appear more prone to shootings in comparison to suburbs
#### 5.4. JointPlots of Lat, Long and Shooting Data
# Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”

sns.jointplot(df_loc_x, df_loc_y, kind='scatter')

sns.jointplot(df_shooting_x, df_shooting_y, kind='hex')

#sns.jointplot(df_loc_x, df_loc_y, kind='hex')

#sns.jointplot(df_loc_x, df_loc_y, kind='kde')
# 14.2 Extract requisite dataset

#df_pairs = df.iloc[ : , [0, 6, 8, 9, 10, 14, 15]]

df_pairs = df.iloc[ : , [0, 8, 9, 10]]



# 14.3 Plot now

#sns.pairplot(df_pairs, x_vars=['YEAR','MONTH','DAY_OF_WEEK'], y_vars='INCIDENT_NUMBER', kind = 'count')

#sns.pairplot(df_pairs, hue = 'SHOOTING', kind = 'scatter', vars=['YEAR','MONTH','DAY_OF_WEEK','Lat','Long'])

#sns.pairplot(df_pairs, hue = 'SHOOTING', kind = 'scatter')