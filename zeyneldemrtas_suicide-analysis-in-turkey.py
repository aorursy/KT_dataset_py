# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib as mpl



import os

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")

data.head()
# suicides_no        = number of suicides

# suicides/100k pop  = suicide rate of per 100 000 population

# HDI for year       = a statistical tool used to measure a country's overall achievement in its social and economic dimensions

# gdp_for_year ($)   = the total monetary or market value of all the finished goods and services produced within a country's borders in a specific time period.

# gdp_per_capita ($) = measure of a country's economic output that accounts for its number of people.

# genaration         = Silent Gen = 1925-1945 ,Boomers = 1946-1963 , Gen X = 1964-1978 ,Gen Y = 1979-1995 ,Gen Z =  1995-2010
data.info()

data.columns
data.describe()
# correlation map

f,ax = plt.subplots(figsize = (12,12))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt=".2f",ax=ax,cbar=True,linecolor="w")

plt.show()

unique_country = data["country"].unique()

print(unique_country)
# filtering data for Turkey 



data_Turkey = data[(data["country"] == "Turkey")]

data_Turkey.head()
data_Turkey.describe()
# correlation map of data_Turkey

f,ax = plt.subplots(figsize = (12,12))

sns.heatmap(data_Turkey.corr(),annot=True,linewidths=.5,fmt=".2f",ax=ax,cbar=True,linecolor="w")

plt.show()
# Scatter Plot

# suicide number for each year

data_Turkey.plot(kind="scatter", x="year", y="suicides_no",alpha = 0.5,color = "red")

plt.xlabel("Year")              # label = name of label

plt.ylabel("Suicides_no ($)")

plt.title("Suicides_no & Year Scatter Plot")   


plt.figure(figsize=(10,5))

sns.barplot(x=data_Turkey["year"], y=data_Turkey["suicides_no"])

plt.title("The Relation Between Year & Number Of Suicides Cases")

plt.xticks()
# now filtering suicide number of male

data_male = data_Turkey[(data_Turkey["sex"] == "male")]

data_male.head()
# now filtering suicide number of female

data_female = data_Turkey[(data_Turkey["sex"] == "female")]

data_female.head()
### Set figure size

plt.figure(figsize=(16,7))

###Let's plot the barplot

bar_age = sns.barplot(x = "sex", y = "suicides_no",hue = "age",data = data_Turkey)

plt.title("Suicides_no by Gender In Turkey", fontsize = 16)
# we can say that suicide number of male > suicide number of female

# most suicides at the age of 35-54 
# Data to plot using pie()



labels = "Male", "Female"

sum_male = data_male["suicides_no"].sum()

sum_female = data_female["suicides_no"].sum()

sizes = [sum_male, sum_female]

colors = ["blue", "red"]

explode = (0, 0.1)  # explode 1st slice



# Plot

mpl.rcParams["font.size"] = 16

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct="%1.1f%%", shadow=True, startangle=140)



plt.axis("equal")

plt.show()