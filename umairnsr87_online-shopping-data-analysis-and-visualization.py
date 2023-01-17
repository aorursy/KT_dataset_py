# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #importing data visualization libraries

import matplotlib.pyplot as mplt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
#reading the dataset file

df=pd.read_csv("../input/online_shoppers_intention.csv")
#VIewing the first 10 rows of the dataset

df.head(10)
#Viewing the last 10 rows of the dataset

df.tail(10)
#checking the NA values

df.isna().sum()
#checking the Null values in the dataset

df.isnull().sum()
#checking the information of the data

df.info()
df["Administrative"].value_counts().plot.bar(color="purple",figsize=(10,5))

mplt.title("Administrative plot")

mplt.ylabel("Number of counts or visits")

mplt.show()
#Administrative_Duration values

df["Administrative_Duration"].value_counts().head(50).plot.bar(figsize=(9,6))

mplt.grid()

mplt.title("Time spent by the user on the page")

mplt.ylabel("Time in seconds")

mplt.show()
sns.distplot(df["Informational_Duration"],kde=True,norm_hist=True)

mplt.title("Time spent by the user on the website")

mplt.grid()

sns.countplot(x="Weekend",data=df)

mplt.title("Time spent by the customers on the weekend on the website")

mplt.ylabel("Time in seconds")

mplt.show()
sns.distplot(df["PageValues"],kde=False)

mplt.ylabel("Counts of the Pagevalues")


sns.lineplot(x="ExitRates",y="Revenue",data=df)
df["BounceRates"].value_counts().plot.bar(color="green",figsize=(9,5))

mplt.title("Bounce Rates")

mplt.ylabel("Number of Bounce Rates")

mplt.show()



sns.countplot(x="VisitorType",data=df)

mplt.title("Type of visitors on the website")
sns.lineplot(x="OperatingSystems",y="TrafficType",data=df)

mplt.title("Operating System vs Traffic types")


sns.lmplot(x="ProductRelated_Duration",y="Weekend",data=df)

mplt.title("Time spent on the products on the weekends")


sns.scatterplot(x="BounceRates",y="Revenue",data=df)

mplt.title("BounceRates vs Revenue")

mplt.rcParams['figure.figsize'] = (30, 20)

sns.heatmap(df.corr(),vmin=-2,vmax=2,annot=True)



mplt.title("Heatmap for the Features")
sns.catplot(x="TrafficType",data=df,alpha=.4)

mplt.title("Type of traffic on the website")


df.SpecialDay.value_counts().plot.bar(figsize=(9,5))

mplt.title("Effect of special days on the website")



sns.countplot(x="Month",data=df)

mplt.title("Monthly visits of the users on teh website")

sns.lmplot(x="Browser",y="TrafficType",data=df)

mplt.title("Different Browser vs Traffic type")

sns.countplot(x="Browser",data=df)

mplt.title("Number of Browsers used to visit the website")
label=["2","1","3","4","8","6","7","5"]

size=[6601,2585,2555,478,79,19,7,6]

colors = ['red', 'yellow', 'green', 'pink', 'blue',"orange","purple","black"]
#pie chart for the Operating systems

mplt.rcParams["figure.figsize"]=(9,9)

mplt.title("website used by different opearting systems")

mplt.pie(size,labels=label,colors=colors,shadow=True,explode = [0.1, 0.1, 0.2, 0.3, .4,.5,.6,.7])

mplt.legend()