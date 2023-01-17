# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb #visualization

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/who_suicide_statistics.csv")
mydf = pd.DataFrame (data)

print(mydf)
mydf.describe()
filt1 = mydf.country == "Turkey"

filt2 = mydf.sex == "male"

filt3 = mydf.age == "15-24 years"

filt4 = mydf.sex == "female"

reqdata = mydf[filt1 & filt2]

mydata = mydf[filt1 & filt2 & filt3]

print(mydata)

summ=0

for each in mydata.suicides_no:

    summ+=each

print("Total suicide numbers of 15-24 years for male in Turkey=", summ)
import matplotlib.pyplot as plt 

countr = mydf [ mydf.country == "Turkey"]

plt.plot(countr.age, countr.suicides_no, color="red", label = "Suicide Numbers", alpha = 1.5)

plt.xlabel("age")

plt.ylabel("suicides_no")

plt.legend()

plt.show()



plt.plot(reqdata.age, reqdata.suicides_no, color="red", label = "Suicide Numbers", alpha = 1.5)

plt.legend()

plt.xlabel("age")

plt.ylabel("suicides_no")

plt.show()
req2 = mydf[filt1 & filt4]

plt.plot(req2.age, req2.suicides_no, color="red", label = "Suicide Numbers", alpha = 1.5)

plt.legend()

plt.xlabel("age")

plt.ylabel("suicides_no")

plt.show()
req2 = mydf[filt1 & filt4]

plt.scatter(req2.age, req2.suicides_no, color="red", label = "Women Suicide Numbers", alpha = 0.5)

plt.scatter(reqdata.age, reqdata.suicides_no, color="green", label = "Men Suicide Numbers", alpha = 0.2)

plt.legend()

plt.xlabel("sex")

plt.ylabel("suicides_no")

plt.show()
f, ax = plt.subplots(figsize=(10,10))

sb.heatmap(data.corr(), annot=True, linewidths=.3, ax=ax) 
mydf.plot(kind="scatter", x="year", y="suicides_no",alpha=0.5,color="red")

plt.xlabel("year") # label = name of label 

plt.ylabel("suicides number") 

plt.title("Year-Suicides Number Scatter Plot")

plt.show()

mydf[ np.logical_and(mydf["year"] > 2014, mydf ["country"] =="Turkey")].suicides_no.count()