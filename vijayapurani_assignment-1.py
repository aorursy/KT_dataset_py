#importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from functools import reduce

import glob

import warnings

warnings.filterwarnings('ignore')
#Read the GDP dataset 

df=pd.read_csv('../input/gdp-analysis-datasets/ab40c054-5031-4376-b52e-9813e776f65e.csv.csv')

df.head()
# Determining the shape of the dataframe and the number of null values

print(f'The shape of the data frame :',df.shape)
#Analysing the number of null values in each column

df.isnull().sum()
#checking the % of NANs columnwise

df.isnull().sum()*100/df.shape[0] 
#Removing the data related to Union territories and West Bengal

df.drop(['West Bengal1','Andaman & Nicobar Islands','Chandigarh','Delhi','Puducherry'],axis=1,inplace=True)
#Removing the rows: '% Growth over the previous year' and 'GSDP-CURRENT PRICES' for the year 2016-17.

df = df[df['Duration']!='2016-17']

df
#Selecting only the required financial years from 2013 to 2016

df=df[(df['Duration']=='2013-14')| (df['Duration']=='2014-15')| (df['Duration']=='2015-16')]

df
#Plotting the total average growth of states

plt.figure(figsize=(14,6), dpi=80, facecolor='w')

df.iloc[:,:-1].groupby(['Items  Description']).get_group('(% Growth over previous year)').mean().sort_values().plot.bar(title='Average GDP across states')

plt.xlabel('States',fontsize=14)

plt.ylabel('Percentage growth in GDP',fontsize=14)
Comparison=df[(df['Items  Description']=='(% Growth over previous year)')][['Tamil Nadu', 'All_India GDP']].mean()

print(Comparison)

print("My state, Tamil Nadu has a GDP that is {} times higher than national average during the given duration" .format(round(Comparison[0]/Comparison[1],2)))

 # dropping the column All India GDP since it is not required anymore 

df.drop(['All_India GDP'], axis=1,inplace=True)

#selecting total GDP of the states for the year 2015-16:

df=df[(df['Items  Description']=='GSDP - CURRENT PRICES (` in Crore)')&(df['Duration']=='2015-16')]

#changing the shape of the dataframe to plot the values

df=pd.melt(df[df.columns[2:]]).sort_values(['value'])

df.head()
#Plotting the total GDP of the states for the year 2015-16

fig, ax = plt.subplots(figsize=(16,8))

sns.barplot(x=df.dropna(axis='rows').variable,y=df.dropna(axis='rows').value) # dropping the na values and plotting 

plt.title(' Total GDP of the states for the year 2015-16',fontsize=14)

plt.xticks(rotation=90,fontsize=12)



plt.xlabel('States',fontsize=14)

plt.ylabel('GDP in Crores',fontsize=14)



# Importing datasets

df_list = []   # initialize a list for storing all the dataframes

cols = ['Item','2014-15']   # to read only the columns required for analysis

fnames=glob.glob("/kaggle/input/**/NAD-*.csv",recursive=True)

for file in fnames :  # reading only those files that start with NAD (data relevant to PArt I B)

    state = file.split('.')[0].split('NAD-')[1].split('-')[0]  # extract the name of the state from file name

    df = (pd.read_csv(file,usecols=cols))        # reading only specified columns from csv file

    df.rename(columns={'2014-15':state},inplace=True)  # renaming the year column to indicate the name of the state

    df_list.append(df)                              # storing the dataframe in a list  





df_state = reduce(lambda  left,right: pd.merge(left,right,on='Item',how='outer'), df_list)  # merging all the dataframe in the list

df_state.head()
df_state.info()
# Getting rid of '_' from state names

df_state.columns=df_state.columns.str.replace('_',' ')

df_state.columns
#Filtering out the union territories

df_state.drop(['Andaman Nicobar Islands','Chandigarh','Delhi','Puducherry'],axis=1,inplace=True)
df_state.info()
df1= df_state.loc[df_state['Item']=='Per Capita GSDP (Rs.)']

df1=pd.melt(df1[df1.columns[2:]]).sort_values(['value'])
plt.figure(figsize=(14,6), dpi=80, facecolor='w')

sns.barplot(x=df1.variable,y=df1.value)



plt.title(' GDP per capita  for the year 2014-15')

plt.xticks(rotation=90)



plt.xlabel('States')

plt.ylabel('GDP Percapita (Rs.)')
#Ratio of the highest per capita GDP to the lowest per capita GDP

print("The ratio of the highest per capita GDP to the lowest per capita GDP is {}" .format(round(df1.value.max()/df1.value.min(),2)))
df_state
# Selecting the sectors and total GDP and storing it in a new dataframe

df_2=df_state.loc[(df_state['Item']=='Primary') | (df_state['Item']=='Secondary') | (df_state['Item']=='Tertiary') | (df_state['Item']=='Gross State Domestic Product')| (df_state['Item']=='Per Capita GSDP (Rs.)')]

df_2 = df_2.set_index('Item').T

df_2.head()
# Converting the contribution of primary, secondary and tertiary sectors into percentage of total GSDP

df_2['Primary'] = round(100* df_2['Primary']/ df_2['Gross State Domestic Product'],2)

df_2['Secondary'] = round(100*df_2['Secondary']/ df_2['Gross State Domestic Product'],2)

df_2['Tertiary'] =round(100* df_2['Tertiary']/ df_2['Gross State Domestic Product'],2)

df_2=df_2.sort_values(by='Per Capita GSDP (Rs.)')  # to sort the dataframe based on GSDP

df_2.head()
# plotting the percentage contribution of each sector

fig, ax = plt.subplots(figsize=(15,8))

df_2[['Primary','Secondary', 'Tertiary']].plot.bar(ax=ax,stacked=True)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('Percentage Contribution of Primary, Secondary and Tertiary Sectors',fontsize=14)

plt.xlabel('States',fontsize=14)

plt.ylabel('Percentage',fontsize=14)
#Categorising the states into four groups based on the GDP per capita

df_3=df_state.set_index('Item').T

df_3['Category']=pd.qcut(df_3['Per Capita GSDP (Rs.)'],[0.0,0.2,0.5,0.85,1],labels=['C4','C3','C2','C1'])

print(df_3.shape)

df_3.head()
df_3=df_3[['Agriculture, forestry and fishing','Mining and quarrying','Manufacturing','Electricity, gas, water supply & other utility services', 'Construction','Trade, repair, hotels and restaurants','Transport, storage, communication & services related to broadcasting','Financial services','Real estate, ownership of dwelling & professional services','Public administration','Other services','Gross State Domestic Product','Category']]

df_3.head()
df_cat=pd.DataFrame(df_3.groupby('Category').sum())

df_cat

#Transposing the data frame

df_cat=df_cat.reset_index().T

df_cat.head()
df_cat.columns=df_cat.iloc[0]  # renaming the columns

df_cat=df_cat[1:]  # dropping the first row which contains column headers

df_cat.head()
#defining a function to determine the sub-sectors that add upto 80%

import numpy as np

from textwrap import wrap

def cum_sum(lst):

    per = lst[:-1].sort_values(ascending=False).apply(lambda x : round((100*x)/lst[-1],2))  # find the percentage contribution of sorted dataframe

    cum_per=per.cumsum()  # find the cumulative percentage

    n = np.argmin(np.abs(np.array(cum_per)-80)) # find the percentage closest to 80%

    print(cum_per.head(n+1))  # print the sectors that contributes to 805 of total

    # plotting contribution of the sub-sectors as a percentage of the GSDP of each category.  

    fig, ax1 = plt.subplots(figsize=(15,6))

    ax= per.plot.bar(ax=ax1)

    labels = [item.get_text() for item in ax.get_xticklabels()]

    labels = ["\n".join(wrap(l,15)) for l in labels]

    ax.set_ylabel('Percentage of GSDP')

    ax.set_xticklabels(labels,rotation=0)
# Sectors that contribute to 80% of GSDP in category C1

cum_sum(df_cat['C1'])
# Sectors that contribute to 80% of GSDP in category C2

cum_sum(df_cat['C2'])
# Sectors that contribute to 80% of GSDP in category C3

cum_sum(df_cat['C3'])
# Sectors that contribute to 80% of GSDP in category C4

cum_sum(df_cat['C4'])
#importing dataset of the dropout rate

drop_rate=pd.read_csv('../input/gdp-analysis-datasets/rs_session243_au570_1.1.csv')

drop_rate.head()
# removing unnecessary columns

drop_rate.drop(['Sl. No.','Primary - 2012-2013','Primary - 2014-2015','Upper Primary - 2012-2013', 'Upper Primary - 2013-2014','Secondary - 2012-2013','Secondary - 2013-2014', 'Senior Secondary - 2012-2013', 'Senior Secondary - 2013-2014','Senior Secondary - 2014-2015'],axis=1,inplace=True)

print(f"The shape is ", drop_rate.shape)

drop_rate.head()

# Renaming columns

drop_rate.rename(columns={'Level of Education - State':'State','Primary - 2014-2015.1':'Primary','Upper Primary - 2014-2015':'Upper Primary','Secondary - 2014-2015':'Secondary'},inplace=True)

# dropping rows that correapond to union territories

rows_to_drop = ['A & N Islands','Chandigarh', 'Dadra & Nagar Haveli','Daman & Diu','Delhi','Lakshadweep','Puducherry','West Bengal','All India']

drop_rate.drop(pd.Index(np.where(drop_rate['State'].isin(rows_to_drop))[0]), inplace = True)

# setting state column as index to facilitate merging

drop_rate.set_index('State',drop=True,inplace=True)

print(drop_rate.shape)

drop_rate.head()
df_3=df_state.set_index('Item').T

df_3.index.rename('State',inplace=True)

df_3.head()

# adding the Per Capita GDP column to drop out rate data frame

drop_rate['Per Capita GSDP (Rs.)']=df_3['Per Capita GSDP (Rs.)']

drop_rate.isnull().sum() # checking NaN values
drop_rate
# The NaN values in Per Capita GSDP column is due to the mismatch of row names in the two dataframes.

# So the column names of drop_rat dataframe is renamed to match that of the df_state column

drop_rate.rename({'Chhatisgarh': 'Chhattisgarh','Jammu and Kashmir' : 'Jammu Kashmir', 'Uttrakhand' : 'Uttarakhand'},axis=0,inplace=True)

# Percapita GSDP column is overwritten now

drop_rate['Per Capita GSDP (Rs.)']=df_3['Per Capita GSDP (Rs.)']

drop_rate
# To determine the correlation between Per Capita GSDP and drop out rate in Primary education

sns.scatterplot(drop_rate['Primary'],drop_rate['Per Capita GSDP (Rs.)'])

plt.title('Dropout rate in Primary education vs GSDP')
# To determine the correlation between Per Capita GSDP and drop out rate in Secondary education

sns.scatterplot(drop_rate['Upper Primary'],drop_rate['Per Capita GSDP (Rs.)'])

plt.title('Dropout rate in Upper Primary education vs GSDP')
# To determine the correlation between Per Capita GSDP and drop out rate in Secondary education

sns.scatterplot(drop_rate['Secondary'],drop_rate['Per Capita GSDP (Rs.)'])

plt.title('Dropout rate in Secondary education vs GSDP')