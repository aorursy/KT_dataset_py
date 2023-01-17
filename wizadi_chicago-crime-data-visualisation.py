# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import time
import matplotlib.pyplot as plt
from decimal import Decimal
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df1 = pd.read_csv("../input/Chicago_Crimes_2012_to_2017.csv")

# use %
df1['Date'] = pd.to_datetime(df1['Date'],format='%m/%d/%Y %I:%M:%S %p')
# df1.groupby(df1["Date"].dt.year).count().plot(kind="bar")

df1['Day'] = df1['Date'].dt.day
df1['Month'] = df1['Date'].dt.month
df1 = df1.sort_values(by='Date')
print(df1.info())
print(df1.head())
# Any results you write to the current directory are saved as output.
#dataframe creation for primary types of crimes
primary_offense_temp=df1.loc[df1.Year==2017]['Primary Type'].value_counts()
primary_offense_2017=pd.DataFrame({'Primary Type':primary_offense_temp.index,'2017':primary_offense_temp.values})

primary_offense_temp=df1.loc[df1.Year==2016]['Primary Type'].value_counts()
primary_offense_2016=pd.DataFrame({'Primary Type':primary_offense_temp.index,'2016':primary_offense_temp.values})

primary_offense_temp=df1.loc[df1.Year==2015]['Primary Type'].value_counts()
primary_offense_2015=pd.DataFrame({'Primary Type':primary_offense_temp.index,'2015':primary_offense_temp.values})

primary_offense_temp=df1.loc[df1.Year==2014]['Primary Type'].value_counts()
primary_offense_2014=pd.DataFrame({'Primary Type':primary_offense_temp.index,'2014':primary_offense_temp.values})

primary_offense_temp=df1.loc[df1.Year==2013]['Primary Type'].value_counts()
primary_offense_2013=pd.DataFrame({'Primary Type':primary_offense_temp.index,'2013':primary_offense_temp.values})

primary_offense_temp=df1.loc[df1.Year==2012]['Primary Type'].value_counts()
primary_offense_2012=pd.DataFrame({'Primary Type':primary_offense_temp.index,'2012':primary_offense_temp.values})

primary_offense = pd.merge(primary_offense_2012,primary_offense_2013,on='Primary Type',how='outer')
primary_offense = pd.merge(primary_offense,primary_offense_2014,on='Primary Type',how='outer')
primary_offense = pd.merge(primary_offense,primary_offense_2015,on='Primary Type',how='outer')
primary_offense = pd.merge(primary_offense,primary_offense_2016,on='Primary Type',how='outer')
primary_offense = pd.merge(primary_offense,primary_offense_2017,on='Primary Type',how='outer')

#make all null values post merge 0
primary_offense = primary_offense.fillna(value={'2012':0,'2013':0,'2014':0,'2015':0,'2016':0,'2017':0})

#make Primary Offense as index
primary_offense = primary_offense.set_index('Primary Type')
print(primary_offense)
# primary_offense['2012']=primary_offense_2012[1]
# primary_offense['2013']=primary_offense_2013[1]
# primary_offense['2014']=primary_offense_2014[1]
# primary_offense['2015']=primary_offense_2015[1]
# primary_offense['2016']=primary_offense_2016[1]
# primary_offense['2017']=primary_offense_2017[1]

### create a table with hierarchical indexing
month_table = df1.groupby(['Year', 'Month'])['Arrest'].count()
year_table = month_table.unstack().sum(axis=1)
# day_table = df1.groupby(['Year','Month','Day'])['Arrest'].count()

day_table = df1.resample('D',on='Date')['Arrest'].count()
#making graph for 2012
x_2012 = day_table['2012'].index
y_2012 = day_table['2012']
plt.plot(x_2012,y_2012)
plt.xlabel('Dates')
plt.ylabel('Number of Arrests')
plt.title('Datewise arrests in 2012')
plt.show()
#making graph for 2013
x_2013 = day_table['2013'].index
y_2013 = day_table['2013']
plt.plot(x_2013,y_2013)
plt.xlabel('Dates')
plt.ylabel('Number of Arrests')
plt.title('Datewise arrests in 2013')
plt.show()
#making graph for 2014
x_2014 = day_table['2014'].index
y_2014 = day_table['2014']
plt.plot(x_2014,y_2014)
plt.xlabel('Dates')
plt.ylabel('Number of Arrests')
plt.title('Datewise arrests in 2014')
plt.show()
#making graph for 2015
x_2015 = day_table['2015'].index
y_2015 = day_table['2015']
plt.plot(x_2015,y_2015)
plt.xlabel('Dates')
plt.ylabel('Number of Arrests')
plt.title('Datewise arrests in 2015')
plt.show()
#making graph for 2016
x_2016 = day_table['2016'].index
y_2016 = day_table['2016']
plt.plot(x_2016,y_2016)
plt.xlabel('Dates')
plt.ylabel('Number of Arrests')
plt.title('Datewise arrests in 2016')
plt.show()
#making graph for 2017
x_2017 = day_table['2017'].index
y_2017 = day_table['2017']
plt.plot(x_2017,y_2017)
plt.xlabel('Dates')
plt.ylabel('Number of Arrests')
plt.title('Datewise arrests in 2017')
plt.show()
#per year crimes graph
ax1=year_table.plot(kind='bar')
ax1.set(xlabel='Year', ylabel='Number of Crimes',title='Number of crimes every year(2012-2017)')
#extract the index to feed as independent variable of histogram
ax2 = month_table.unstack().plot(kind='bar')
# set labels 
ax2.set(xlabel='Year', ylabel='Number of Crimes',title='Number of crimes every month(2012-2017)')

#use this as a bar plot later use BOKEH PLEASE
ax3 = primary_offense.plot(kind='bar')
ax3.set(xlabel='Primary Type of Crime', ylabel='Number of Crimes',title='Number of crimes every year by crime type(2012-2017)')



