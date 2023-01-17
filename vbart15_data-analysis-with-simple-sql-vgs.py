#First we import the necessary libraries to analyze our data

import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
from pandasql import sqldf

pysqldf= lambda q: sqldf(q, globals())
#importing our data into the notebook

vgs_data=pd.read_csv('../input/videogamesales/vgsales.csv');

#setting the index of our dataset and checking if it imported correctly:
vgs_data.set_index('Rank',inplace=True)
vgs_data.head(10)
#These are queries that print out and are not modifiable
xbox_q='''
        select Platform, SUM(NA_Sales) as "Total NA_Sales", SUM(Global_Sales) as "Total Global_Sales"
        from vgs_data
        where Platform="X360"
        '''

ps3_q='''
        select Platform, SUM(NA_Sales) as "Total NA_Sales", SUM(Global_Sales) as "Total Global_Sales"
        from vgs_data
        where Platform="PS3"
        '''

wii_q='''
        select Platform, SUM(NA_Sales) as "Total NA_Sales", SUM(Global_Sales) as "Total Global_Sales"
        from vgs_data
        where Platform="Wii"
        '''

print(sqldf(xbox_q,locals()))
print(sqldf(ps3_q,locals()))
print(sqldf(wii_q,locals()))
#query to see top 10 selling titles across the entire dataset
top10_q='''
        select Name, Platform, Publisher, NA_Sales, Global_Sales
        from vgs_data
        order by Global_Sales DESC
        limit 10

        '''
print(sqldf(top10_q,locals()))
#These commands put our SQL queries into dataframes. We then concat all these queries into a single dataframe, this allows us to mess with the dataframe using pandas.
vgs_Xsales = pysqldf('''
                    select Platform, SUM(NA_Sales) as "Total NA_Sales", SUM(Global_Sales) as "Total Global_Sales"
                    from vgs_data
                    where Platform="X360"
                    ''')


vgs_Psales = pysqldf('''
                    select Platform, SUM(NA_Sales) as "Total NA_Sales", SUM(Global_Sales) as "Total Global_Sales"
                    from vgs_data
                    where Platform="PS3"
                    ''')

vgs_Wsales = pysqldf('''
                    select Platform, SUM(NA_Sales) as "Total NA_Sales", SUM(Global_Sales) as "Total Global_Sales"
                    from vgs_data
                    where Platform="Wii"
                    ''')

sales=[vgs_Xsales,vgs_Psales,vgs_Wsales]
vgs_sales=pd.concat(sales)
vgs_sales.head()

x1=vgs_sales["Platform"]
y1=vgs_sales["Total NA_Sales"]
z1=vgs_sales["Total Global_Sales"]

#plotting with pyplot
#------------------------------
#plt.bar(x,y)
#plt.xlabel("Platforms")
#plt.ylabel("Total NA_Sales")
#plt.show()
#------------------------------

#plotting with seaborn
#-----------------------------
sns.barplot(x=x1,y=y1)
sns.set_style('whitegrid')
plt.show()
sns.barplot(x=x1,y=z1)
sns.set_style('whitegrid')
plt.show()
Xtop5df=pysqldf('''
        select Name, Platform, Publisher,NA_Sales, Global_Sales
        from vgs_data
        where Platform='X360'
        limit 5
        ''')
Xtop5df.head()
x2=Xtop5df['Publisher']
y2=Xtop5df['NA_Sales']

sns.barplot(x=x2,y=y2,)
sns.set_style('whitegrid')
plt.show()
Ptop5df=pysqldf('''
        select Name, Platform, Publisher,NA_Sales, Global_Sales
        from vgs_data
        where Platform='PS3'
        limit 5
        ''')
Ptop5df.head()
x3=Ptop5df['Publisher']
y3=Ptop5df['NA_Sales']
sns.barplot(x=x3,y=y3,)
sns.set_style('whitegrid')
plt.show()
Wtop5df=pysqldf('''
        select Name, Platform, Publisher,NA_Sales, Global_Sales
        from vgs_data
        where Platform='Wii'
        limit 5
        ''')
Wtop5df.head()
x4=Wtop5df['Publisher']
y4=Wtop5df['NA_Sales']

sns.barplot(x=x4,y=y4,)
sns.set_style('whitegrid')
plt.show()