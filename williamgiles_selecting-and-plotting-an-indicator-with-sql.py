# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import sqlite3
# defining the connection



conn = sqlite3.connect('../input/world-development-indicators/database.sqlite')
# getting the name of the different tables



pd.read_sql(""" SELECT *

                FROM sqlite_master

                WHERE type='table';""",

           conn)
# lookig at the top 5 rows all columns.

# Want to see how many different indicators there are



pd.read_sql(""" SELECT * FROM indicators

                LIMIT 5

                ;""",

           conn)



# selecting the unique codes seems like a better idea than the name, so I do that.



pd.read_sql(""" SELECT DISTINCT IndicatorCode

                FROM indicators

                ;""",

           conn)

# This time I want to see the different names



pd.read_sql(""" SELECT DISTINCT IndicatorName

                FROM indicators

                ;""",

           conn)



# I want to search for a variable. I will search for an 'Internet' variable



pd.read_sql(""" SELECT IndicatorName FROM Indicators

                WHERE IndicatorName LIKE '%Internet%'

                ;""",

           conn)



# ok this returns a lot of indicator names. many of them are repeated. 

# I will try a DISTINCT statement as well.



pd.read_sql(""" SELECT DISTINCT IndicatorName, IndicatorCode

                FROM indicators

                WHERE IndicatorName LIKE '%Internet%'

                ;""",

           conn)



# this is a better result. I can see there are three indicators related to the internet. 

# I will retrieve the data with the internet users per 100 people
# I won't need the indicator name or code as they will all be identical. 

# I will look for the entries with for this indicator with the highest value



pd.read_sql(""" SELECT CountryName, CountryCode, Year, Value FROM indicators

                WHERE IndicatorCode = 'IT.NET.USER.P2'

                ORDER BY value desc

                LIMIT 10

                ;""",

           conn)



# We can see northern european countries doing well. 
# I want to compare some of these countires to the European Union average. 

# I will see if there is one in the table.



pd.read_sql(""" SELECT DISTINCT CountryName

                FROM indicators

                WHERE CountryName LIKE '%europe%'

                ;""",

           conn)



# There is, so I will add this in.
# I will select which countires to include. To start with I will go for the scandinavian ones. 

# I will just look at the more recent years.



pd.read_sql(""" SELECT CountryName, Year, Value FROM indicators

                WHERE IndicatorCode = 'IT.NET.USER.P2'

                AND CountryName IN ('European Union', 'Norway', 'Sweden','Denmark', 'Iceland', 'Finland', 'Greenland') 

                AND Year > 2010

                ORDER BY Value

                ;""",

           conn)



#We can see that all countires greatly outperform the EU average for internet per people in the year 2014, bar Greenland, who are last.

# I want to visualise this. I am going to save these results and visualise them using matplotlib.



north_europe_internet = pd.read_sql(""" SELECT CountryName, Year, Value FROM indicators

                WHERE IndicatorCode = 'IT.NET.USER.P2'

                AND CountryName IN ('European Union', 'Norway', 'Sweden','Denmark', 'Iceland', 'Finland', 'Greenland') 

                AND Year > 2000

                ORDER BY Value

                ;""",

           conn)
# I need to edit the table so that the countires are the columns and the years are the rows (or vice versa)

# This will allow me to make the graph easily.

# I can use the .pivot_table method to achieve this.



p_table = north_europe_internet.pivot_table('Value',['CountryName'], 'Year')

p_table



# This is great but I want the columns and rows reversed. I will use the transpose method.



p_table = p_table.T

p_table
from matplotlib import rcParams

rcParams['figure.figsize']=12,8

plt.plot(p_table['Denmark'],'red')

plt.plot(p_table['European Union'], 'blue')

plt.plot(p_table['Greenland'], 'Green')

plt.plot(p_table['Finland'], 'Grey')

plt.plot(p_table['Norway'], 'purple')

plt.plot(p_table['Sweden'], 'black')

plt.plot(p_table['Iceland'], 'brown')

plt.title('Internet per 100 people in Northern Europe')

plt.xlabel('Date')

plt.ylabel('Internet per 100 people')



# this isn't bad. Can see the gains over time. Can also see the EU average way behind.


# lets look more at the lower end. 

#I will look for under 10 people per 100 in the same time period. 



pd.read_sql(""" SELECT CountryName, Year, Value FROM indicators

                WHERE IndicatorCode = 'IT.NET.USER.P2'

                AND value < 10

                AND year > 2000

                ORDER BY value

                ;""",

           conn)



# over 1000 entries. I will lower the value a little for the sake of readability

# I will reorganise this so it is easier to read in to a pivot table. 



low_internet_users = pd.read_sql(""" SELECT CountryName, Year, Value FROM indicators

                WHERE IndicatorCode = 'IT.NET.USER.P2'

                AND value < 10

                AND year > 2000

                ORDER BY value

                ;""",

           conn)
low_internet_users = low_internet_users.pivot_table('Value',['CountryName'],'Year')

low_internet_users



# I can see a lot of null values. I will remove these
low_internet_users = low_internet_users.dropna()

low_internet_users = low_internet_users.T

low_internet_users

#dropping the null values gave me countires with complete data only



# There are too many to actually put into a graph as it will be too confusing. Instead I will select on

rcParams['figure.figsize']=20,12

plt.plot(low_internet_users)



# even making it bigger, this isn't very informative as it has too much going on. Would be better to remove some of the higher
# I will drop some of the countries that were performing higher earlier on.



low_internet_users = low_internet_users.T

low_internet_users



filt = low_internet_users[2008]>1

low_internet_users.drop(low_internet_users[filt].index)

very_low_internet_users = low_internet_users.drop(low_internet_users[filt].index)

very_low_internet_users = very_low_internet_users.T

very_low_internet_users
plt.plot(very_low_internet_users)



# we can see that there are two or three countries which started in a similar position but then greatly outperformed the others.

# a quick look at the data identifies these as Burkina Faso and Cambodia, with Malawi and the CAR doing ok as well. 