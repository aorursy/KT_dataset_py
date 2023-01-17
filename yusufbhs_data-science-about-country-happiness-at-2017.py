# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization tool
import seaborn as sns # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2017.csv") # reading csv data through pandas
data.info() # columns,count,data types
data.columns # columns's names
data.rename(columns={"Country": "Country", "Happiness.Rank":"HappinessRank","Happiness.Score":"HappinessScore",
                        "Health..Life.Expectancy.":"HealthLifeExpectancy","Whisker.high":"WhiskerHigh","Whisker.low":"WhiskerLow",
                        "Economy..GDP.per.Capita.":"EconomyPerCapital","Freedom":"Freedom",
                        "Generosity":"Generosity","Trust..Government.Corruption.":"TrustGovernmentCorruption",
                        "Dystopia.Residual":"DystopiaResidual"},inplace = True)
# We are using inplace=True to change column names in place.
data.columns # final form of columns's names
data.head(10) # shows first 10 rows 
data.describe() # shows values such as max,min,mean (for numeric feature)
data.corr() # relevance between columns
# correlation map
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.8, fmt= '.2f',ax=ax)
plt.show()
data.EconomyPerCapital.plot(kind="line",color="green",label="EconomyPerCapital",linewidth=1,alpha=0.8,grid=True,linestyle=':',figsize = (8,8))
data.HealthLifeExpectancy.plot(kind="line",color="blue",label="HealthLifeExpectancy",linewidth=1,alpha=0.8,grid=True,linestyle='-.',figsize = (8,8))
plt.legend(loc='upper left')
plt.title('Line Plot') 
plt.xlabel("Country")
plt.ylabel("Variables")
plt.show()
data.plot(kind="scatter",x="EconomyPerCapital",y="HealthLifeExpectancy",alpha=0.5,color="r",figsize = (6,6))
plt.title('Scatter Plot') 
plt.xlabel("Economy Per Capital")
plt.ylabel("Health Life Expectancy")
plt.show()
data.Freedom.plot(kind='hist',bins=60,grid=True,figsize=(10,10)) 
plt.title('Histogram Plot') 
plt.xlabel("Freedom")
plt.show()
NewData = data[(data.Freedom > 0.5) & (data.TrustGovernmentCorruption > 0.4)] # used for filtering
NewData