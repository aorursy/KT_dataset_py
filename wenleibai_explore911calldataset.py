# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



#load and check the dateframe:

df = pd.read_csv('/kaggle/input/montcoalert/911.csv')

df.info()

df.head()
# What are the top 5 zipcodes for 911 calls?

df['zip'].value_counts().head(5)
#What are the top 5 townships (twp) for 911 calls?

df['twp'].value_counts().head(5)
#Take a look at the 'title' column, how many unique title codes are there?

df['title'].nunique()
#In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.

#For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS.

df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
#What is the most common Reason for a 911 call based off of this new column?

df['Reason'].value_counts()
#Now use seaborn to create a countplot of 911 calls by Reason.

sns.countplot(x='Reason',data=df,palette='viridis')
#Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? convert to DateTime objects

type(df['timeStamp'].iloc[0])

df['timeStamp'] = pd.to_datetime(df['timeStamp'])

df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
#Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week:  

#dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day of Week'] = df['Day of Week'].map(dmap)
#Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.

sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')

# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')

# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#You should have noticed it was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, 

#possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas..

#Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. 

#Use the head() method on this returned DataFrame.

byMonth = df.groupby('Month').count()

byMonth.head()
#Now create a simple plot off of the dataframe indicating the count of calls per month.

# Could be any column

byMonth['twp'].plot()
#Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column.

sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())
#Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method.

df['Date']=df['timeStamp'].apply(lambda t: t.date())
#Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.

df.groupby('Date').count()['twp'].plot()

plt.tight_layout()

plt.grid()
#Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call

df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()

plt.title('Traffic')

plt.tight_layout()

plt.grid()
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()

plt.title('Fire')

plt.tight_layout()

plt.grid()
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()

plt.grid()
#Now let's move on to creating heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes 

#the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an unstack method. Reference the solutions if you get stuck on this!

dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

dayHour.head()
#Now create a HeatMap using this new DataFrame.

plt.figure(figsize=(12,6))

sns.heatmap(dayHour,cmap='viridis')