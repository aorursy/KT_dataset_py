import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

data.head()  # see first 5 rows
# Let's create three different lists.

team = ["Fenerbahce","Galatasaray"]

team_value = ["150M","180M"]

list_label = ["team","team_value"]

list_col = [team,team_value]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# Add new columns

df["Player_Number"] = ["25","23"]

df
# Broadcasting

df["Expenses"] = 100000000 #Broadcasting entire column

df
# Plotting all data 

data1 = data.loc[:,["suicides_no","population"]]

data1.plot()

plt.show()

# So confusing
# subplots

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="population",y = "suicides_no")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "suicides_no",range= (0,500),bins = 10)
fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "suicides_no",bins = 50,range= (0,500),ax = axes[0])

data1.plot(kind = "hist",y = "suicides_no",bins = 50,range= (0,500),ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt.show()
data.describe()
time_list = ["1985-01-01","2016-12-31"]

print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
# close warning

import warnings

warnings.filterwarnings("ignore")

# In order to practice lets take head of pokemon data and add it a time list

data2 = data.head()

date_list = ["1987-01-01","1987-02-01","1987-03-01","1987-04-01","1987-05-01"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

data2 
print(data2.loc["1987-01-01"])

print(data2.loc["1987-01-01":"1987-03-01"])
# We will use data2 that we create at previous part

data2.resample("A").mean()
# Lets resample with month

data2.resample("M").mean()
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# We can interpolete from first value

data2.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()

data2.resample("M").mean().interpolate("linear")
# read data

data = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

data= data.set_index("year")

data.head()
# indexing using square brackets

data["sex"][1987]
# using column attribute and row label

data.sex[1987]
# using loc accessor

data.loc[1987,["sex"]]
# Selecting only some columns

data[["sex","age"]]
# Difference between selecting columns: series and dataframes

print(type(data["sex"]))     # series

print(type(data[["sex"]]))   # data frames
# Slicing and indexing series

data.loc[:,"sex":"suicides_no"]
data.loc[::-1,"sex":"suicides_no"]
boolean = data.suicides_no > 100

data[boolean]
first_filter = data.suicides_no > 200

second_filter = data.population > 1000000

data[first_filter & second_filter]
# Plain python functions

def div(n):

    return n/2

data.suicides_no.apply(div)
# Or we can use lambda function

data.suicides_no.apply(lambda n : n/2)
# Defining column using other columns

data["Ratio"] = data.population / data.suicides_no

data.head()
# our index name is this:

print(data.index.name)

# lets change it

data.index.name = "index_name"

data.head()
# lets read data frame one more time to start from beginning

data = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

data.head()

# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index

data1 = data.set_index(["country","sex"]) 

data1.head(100)

# data1.loc["Fire","Flying"] # howw to use indexes