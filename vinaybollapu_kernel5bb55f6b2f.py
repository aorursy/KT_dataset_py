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
# importing all the libraries required.

import pandas as pd

import matplotlib.pyplot as plt

from pandas.plotting import lag_plot

import random
data = pd.read_csv('/kaggle/input/Accidental_Drug_Related_Deaths_2012-2018.csv')
# Finding out the total Nan value present in each column.

data.isnull().sum()
# The data frame is modified. 

# notnull() removes the Nan values present in Date column

data = data[pd.notnull(data['Date'])]
# The missing values are filled with random integers. 

data['Age'].fillna(random.randint(16,56), inplace = True)
# Making sure that Age is always an integer. 

data['Age'] = data['Age'].astype('int')
# Making sure that Age column has no null values

data['Age'].isnull().sum()
# drugNames is list which stores the drug column names of the data set. 

drugNames = list(data.columns[20:37])
# This for loop replaces the Nan values with N for all the drug columns. 

for drugColumn in drugNames:

    data[drugColumn].fillna('N', inplace=True)
data['OtherSignifican'].fillna('Not Specified', inplace = True)
# This list holds the names of columns which are to be dropped

dropColumns = list(data.columns[6:18])
# This lopp drops the columns 

for columnName in dropColumns:

    data = data.drop(columnName,axis = 1)
# Data Frame is group by Gender column

genderGroup = data.groupby('Sex')
# Groupby variable holds the data frame by grouping Sex Column. Getting the count of it will answer the above question. 

# A bar graph is plotted to show the results. 

_ = genderGroup['Sex'].count().plot(kind = 'bar')

_ = plt.xlabel('Gender')

_ = plt.ylabel('Number of Deaths')

_ = plt.title('Total number of deaths categorized by Gender')
# Declaring an empty list

totalValue = []
# This loop will pick the drug column name and iterates through dat frame. 

# The summation counter increments if the value of each cell is not N. 

for drug in drugNames:

    summation = 0

    for index , value in data.iterrows():

        if(data.at[index,drug] != 'N'):

            summation = summation + 1

    totalValue.append(summation)
# Plotting the results on a scatter plot. 

_=plt.figure(figsize=(30, 10))

_=plt.scatter(drugNames,totalValue)

_ = plt.xlabel('Type of Drug',fontsize = 20)

_ = plt.ylabel('Total Number of Deaths',fontsize = 20)

_ = plt.title('Total number of deaths categorized by type of Drug', fontsize = 30)

_ =plt.grid(True)
# Grouping the Race column

raceGroup = data.groupby('Race')
# This code displays the total number of deaths caused by drugs categorized by Race in connecticut

raceGroup['Race'].count()
# Plotting the count of number of deaths from the grouped variable. 

_= plt.Figure(figsize = (50,30))

_= raceGroup['Race'].count().plot(kind = 'bar')

_ = plt.xlabel('Race', fontsize = 15)

_ = plt.ylabel('Total number of Deaths',fontsize = 15)

_ = plt.title('Total number of deaths categorized by Race',fontsize = 15)
# Creating a new column in the data set by the name Teenager. 

# lambda expression is applied which will declare the cell as True if the Age value is <= 22

data['Teenager'] = data['Age'].apply(lambda x: x <= 22)
# Displaying the data frame after adding Teenager Column

# Observe the last column name of the data set

data.head()
# This loop iterates through the data set and checks for Teenager Column. 

# Other age group Counter is incremented if Teenager is matcher with False and vice versa.

Teenager  = 0 #Teenage Counter

NotATeenager = 0 # Other Age Group Counter

for index , value in data.iterrows():

    if(data.at[index,'Teenager'] == False):

        NotATeenager = NotATeenager + 1

    else:

        Teenager = Teenager + 1
# PLotting a pie graph

labels = 'Teenage Group(<=22)', 'Other Age Groups(>22)'

sizes = [Teenager, NotATeenager]

colors = ['lightcoral', 'lightskyblue']

explode = (0.1, 0)  # explode 1st slice





plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Percentage of Deaths by age groups')

plt.axis('equal')

plt.show()