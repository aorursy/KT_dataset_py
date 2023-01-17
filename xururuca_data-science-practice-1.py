# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/food-inspections.csv')
data.info()
data.head()
data.corr()
f, ax  = plt.subplots(figsize = (10, 10))

sns.heatmap(data.corr(), annot=True, linewidths = .5, fmt= '.1f', ax=ax)

plt.show()
data.columns 
# Lets clean column names, there is gaps between words.



data.columns = [each.split()[0] + "_" + each.split()[1] if len(each.split()) > 1 else each for each in data.columns]

data.columns
# Lets see City spread.



data.groupby('City')['Inspection_ID'].nunique() 



#City names can not be used for any analysis before cleaning, so skipping it!
#lets check the unique results count

data.Results.count()

data.groupby('Results')['Inspection_ID'].nunique()



# we can look into details now:
#Bar plot for results



data['Results'].value_counts().plot.bar()
#Pie Chart for results



data['Results'].value_counts().plot.pie() #Dont know how to fix labels colliding here
x =data.groupby('Inspection_Date').nunique()

x.tail()
# lets try to see inspection count over time series

from matplotlib.pyplot import figure



#Want to see it bigger

figure(num=None, figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')



# Dont know how to add date tags to axis. tried plt.xticks but failed



data['Inspection_Date'].value_counts().sort_index().plot.line()

plt.xlabel('Dates')

plt.ylabel('Count of inspection')

plt.title('Number of Inspections per Day')
# At which date inspectors worked a lot?

data['Inspection_Date'].value_counts().sort_index().max()

i=0

x=0

for each in data['Inspection_Date'].value_counts().sort_index():

    i = i+1

    if each == 185:

        print("Inspectors worked a lot on: "+ data['Inspection_Date'].sort_index()[i-1])
data['Risk'].value_counts()
#Lets index data in terms of Risk value

data1 = data.set_index(["Risk"])

data1.head(10)
#Lets melt latitude and longtide into single column



data_melted = pd.melt(frame=data, id_vars='DBA_Name', value_vars=['Latitude','Longitude'])

data_melted
# Lets see howmany nan value we have.

data_melted.info()

# Around 1300 value has nan
data['Facility_Type'] = data['Facility_Type'].fillna('Not Given')



plt.figure(figsize=(10,10))

sns.countplot(x='Facility_Type', data=data.head(20))

plt.show()
