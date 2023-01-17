# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting library
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/earthquakes-solar-system-objects/SolarSystemAndEarthquakes.csv')
data.head(5)
# i used Start, Stop, Step
earthquake_data = data[data.columns[3:4:1]]
earthquake_data.head(10)

#i used conditional selection and for loop
moon_cols = [col for col in data.columns if 'Moon' in col]
moon_data = data[moon_cols]
moon_data.head(10)

Venus_cols = [col for col in data.columns if 'Venus' in col]
Venus_data = data[Venus_cols]
Venus_data.head(10)

# concat function concatenates data frames
correlation_data = pd.concat([earthquake_data.iloc[::],Venus_data.iloc[::],moon_data.iloc[::]],axis=1)
correlation_data.head(10)
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(correlation_data.corr(), annot=True, linewidths=.2, fmt= '.1f',ax=ax)
plt.show()
data.dtypes.head(10)
# Line Plot
data.plot(kind='line', x='earthquake.time', y='earthquake.latitude', alpha = 0.5,color = 'green',grid = True)
plt.show()
# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='Venus.speed', y='earthquake.mag',alpha = 0.2,color = 'blue',figsize = (10,10))
plt.xlabel('Venus Speed')              # label = name of label
plt.ylabel('Earthquake Magnitude')
plt.title('Scatter Plot')            # title = title of plot
plt.show()
# Histogram
# bins = number of bar in figure
data.plot(kind='hist',  y='earthquake.mag',bins = 45, alpha = 0.7,color = 'green',figsize = (10,10))
plt.show()
# we retrieved data from a .csv file.
data = pd.read_csv('../input/earthquakes-solar-system-objects/SolarSystemAndEarthquakes.csv')
data.columns
# DataFrame is a 2-dimensional data structure with columns of potentially different types.
data_frame = data[['earthquake.time','earthquake.place']]  
print((data_frame.head(10)))

# Series is single column of a dataframe
series = data['earthquake.time']        
print((series.head(10)))

# 1 - Filtering Pandas data frame, i wanted to filter earthquakes after 2016.
# there are 36 earthquakes after 2016 in our dataset.
x = data['earthquake.time']>'2016-01-01T00:00:00.000Z'
data[x].head(10)
# 2 - Filtering pandas with logical_and
# We can see the earthquakes happen which have higher magnitude than 7 and when the moon illumination bigger than 90
data[np.logical_and(data['earthquake.mag']>7, data['MoonPhase.illumination']>90 )].head(10)

# and this is also same statement with previous. We just used '&' operator for filtering
#data[(data['earthquake.mag']>7) & (data['MoonPhase.illumination']>90)]
def ExtractTuple():
    """This function returns a tuple, defined in function."""
    tuple1 = ('Istanbul','Turkey', 10, 2018 )  
    return tuple1

city, country, month, year = ExtractTuple()
tuple2 = ExtractTuple()
print(city, country, month, year)
print(tuple2[0])
print (tuple2[1:3]) # Start, Stop

def UpdateTuple(tuple3):
    """This function updates a tuple given"""
    # tuple3[0] = "This value was updated." !!! This is not valid. Remember, tuples cannot be changed. !!!
    tupleNewValue = ('New Value',) # the comma is required for one element tuple
    tupleNew = tuple3 + tupleNewValue
    return tupleNew

print(tuple2)
tupleResult = UpdateTuple(tuple2)
print(tupleResult)

thisIsVariable = 9
def f():
    """scope example"""
    thisIsVariable = 7
    return thisIsVariable

print('This is global variable',thisIsVariable)
print('This is local variable',f())
# builtins are functions and constants that embedded in Python.
# This module provides direct access to all ‘built-in’ identifiers of Python.
import builtins

# let's look at built-in identifiers
dir(builtins)
def outer(num1):
    def inner_increment(num1):  # Hidden from outer code
        return num1 + 1
    num2 = inner_increment(num1)
    print(num1, num2)

#inner_increment(10) # try calling this
outer(10)
# b is default argument
def defaultArg(name, b='Hello!'):
    print (name,b)
defaultArg('Ensar') 

# if we want to change default
defaultArg('Ensar','Hi')

# *args can be one or more
def myArgs(*args):  
    for arg in args:  
        print (arg)           
myArgs('Ensar','Turkey','Istanbul')      

# **kwargs is a dictionary
def myKwargs(**kwargs):  
    for key, value in kwargs.items(): 
        print ("%s == %s" %(key, value)) 
myKwargs(one = 'A', two='B', three ='C')

lambFunction = lambda x,y : x + y
print(lambFunction(5,6))
thisIsTuple = ("orange", "mandarin", "lemon")
thisIsIterator = iter(thisIsTuple)

# next value
print(next(thisIsIterator))

# remaining values
print(*thisIsIterator)

# Even strings are iterable objects.
aString = "Incomprehensibilities"
iterator = iter(aString)
print(next(iterator))
print(next(iterator))
print(*iterator)
numberList = [1, 2, 3]
strList = ['one', 'two', 'three']

# Two iterables are passed
result = zip(numberList, strList)

# Converting iterator to list
resultList = list(result)
print(resultList)

un_zip = zip(*resultList)
un_list1,un_list2 = list(un_zip)
print(un_list1)
print(un_list2)
oldList = [1,2,3,4,5]
newList = [ item * 10 for item in oldList if item > 2 ]

print(newList)
# Let's return to our Earthquakes data and give an example.
# I want to show you earthquakes when happened after 2016 and greater than a threshold value.

# i will look at column names and calculate a threshold.
data.columns
threshold =sum(data['earthquake.mag']) / len(data['earthquake.mag'])

# we have classified the earthquakes by threshold.
str1 = 'High'
str2 = 'Lower'
data["magnitude_level"] = [str1 if i > threshold else str2 for i in data['earthquake.mag']]

# loc function can access a group of rows and columns by labels or a boolean array.
data.loc[:,["magnitude_level","earthquake.mag","earthquake.place","earthquake.time"]].head(10)

data = pd.read_csv('../input/earthquakes-solar-system-objects/SolarSystemAndEarthquakes.csv')

# head shows first 5 rows without parameter, otherwise up to parameter
data.head()  
# tail shows last 5 rows without parameter, otherwise up to parameter
data.tail()
# columns gives the column labels
data.columns
# Return a tuple of rows and columns of the dataframe
data.shape
# This method prints information about a DataFrame
# if we want to print a short summary, we use verbose=False. Please try yourself for 'True'.
data.info(verbose=False)
# describe() is used to view some basic statistical details 
# like percentile, mean, std etc. of a data frame or a series of numeric values.
# it ignores NaN values and non-numeric values.

data.describe()

# count -> number of entries
# mean --> average of entries
# std ---> standard deviation
# min ---> minimum value
# 25% ---> first quartile, Q1
# 50% ---> second quartile, median
# 75% ---> third quartile, Q3
# max ---> maximum value
data.boxplot(column="earthquake.mag", by="MoonPhase.dynamic",figsize=(9,9))
plt.show()
# let's achiece a new dataframe from our dataset.
# i will choose only first 5 rows.
data_new = data.head()
data_new

# value_vars -> the columns that i want to melt.
melted_data = pd.melt(frame=data_new,id_vars='earthquake.time',value_vars=['earthquake.mag','Venus.speed'])
melted_data
# i want to turn variables to columns.
# values are still values.
melted_data.pivot(index='earthquake.time',columns='variable',values='value')
data1 = data.head()
data2 = data.tail()
# axis = 0 means vertically concat...
concatVertically = pd.concat( [ data1, data2 ], axis = 0, ignore_index = True )
concatVertically
# axis = 1 horizontally concat
data1 = data['earthquake.mag'].head()
data2 = data['Venus.speed'].head()
concatHorizontally = pd.concat( [ data1, data2 ], axis = 1, ignore_index = False )
concatHorizontally
data.dtypes.head(10)
# let's convert somethings.
data['earthquake.latitude'] = data['earthquake.latitude'].astype('category')
data['MoonPhase.percent'] = data['MoonPhase.percent'].astype('str')
data.dtypes.head(10)
# For this section, i will change my dataset.
# This dataset holds data about animal bites.
data = pd.read_csv('../input/animal-bites/Health_AnimalBites.csv')
data.info()
# In the previous scope we see that, we have (9003 - 6477 = )2526 null value in GenderIDDesc column.
# dropna = False means; show me the also null values.
data["GenderIDDesc"].value_counts(dropna = False)
# According to output, there are 3832 male, 2016 female, 2526 null value.
# If we want to drop null values.
data1 = data # actually this statement not preferred.

# inplace = True means, after dropna, assign new data to same dataframe, do not generate new dataframe.
#data1["GenderIDDesc"].dropna( inplace = True ) 

# we will see that there is no NaN values in GenderIDDesc column.
data1.head(10)
# Now we will check previous statement with assert.
# assert statement returns nothing if the statement is true, and returns error if the statement is false.
assert 1 == 1
assert 1 == 2
# i expect that the below statement returns nothing, because i dropped all of the NaN values.
assert data1["GenderIDDesc"].notnull().all()
data = pd.read_csv('../input/animal-bites/Health_AnimalBites.csv')

#we can also fill NaN values with 'empty'
data["GenderIDDesc"].fillna('empty',inplace = True)

# We can check a lot ot thing with assert.
# assert data.columns[0] == 'Name'
assert data.columns[0] == 'bite_date'
# dataframe from, dictionary from, list
# it is useful to see each step.
country = ["Turkey","India","USA"]
population = [34,56,67]
list_label = ["country","population"]
list_col = [country,population]
list_col

zipped = list(zip(list_label,list_col))
zipped

dictionary = dict(zipped)
dictionary

dataFrame = pd.DataFrame(dictionary)
dataFrame
# adding new column
dataFrame["capital"] = ["Ankara","New Delhi","Washington"]
dataFrame
# broadcasting
dataFrame["new column"] = 0
dataFrame
# plotting all data
data = pd.read_csv('../input/earthquakes-solar-system-objects/SolarSystemAndEarthquakes.csv')
dataPlot = data.head(1000).loc[:,["earthquake.mag","earthquake.longitude","Venus.speed"]]
dataPlot.plot()
plt.show()
# this plot is confusing, let's look at subplots.
dataPlot.plot(subplots=True)
plt.show()
# scatter plot
dataPlot.plot(kind="scatter",x = "Venus.speed", y="earthquake.mag")
plt.show()
# histogram
dataPlot.plot(kind="hist", y= "earthquake.mag", bins=50, range=(6,10), density = True)
# cumulative and non-cumulative histogram plot
dataPlot.plot(kind="hist", y= "earthquake.mag", bins=50, range=(6,10), density = True, cumulative = True)
plt.show()
data.head(100).describe()
timestamp_object = pd.to_datetime(list(data["earthquake.time"]))
timestamp_object
print(type(data["earthquake.time"][1]))
print(type(timestamp_object))
# let's make timestamps as index
data2 = data.head(10).loc[:]
timestampObj = pd.to_datetime(list(data2["earthquake.time"].str.replace("T"," ").str.replace("Z"," ")))
data2["newTime"] = timestampObj 
data2 = data2.set_index("newTime")
data2
# now we can select data according to my new index
print(data2.loc["2016-04-20":"2016-04-15"])

# resample according to year
data2.resample('A').mean()
# resample according to hour
# Because of some of hours don't have data, we see NaN.
data2.resample('H').mean().head()
# we use interpolate to estimate values don't have.
# of course, this function is valid only numeric columns.
data2.resample('H').first().interpolate("linear").head()
# read data
data = pd.read_csv('../input/earthquakes-solar-system-objects/SolarSystemAndEarthquakes.csv')
data2 = data.head(10).loc[:]
data2.insert(0, '#', range(1, len(data2) + 1))
data2.set_index('#',inplace=True)
data2
#   '#' is our new index now.
# after this index refreshing if want to reach some data, for example first index is 1 now.
#  Indexing using square brackets
print(data2["earthquake.place"][1])
#  using column attribute
# my columns includes '.' char so i will change the name.
data2=data2.rename(columns = {'earthquake.place':'earthquake_place'})
print(data2.earthquake_place[1])
#  using loc
print(data2.loc[1,"earthquake_place"])
# selecting only spesific columns
data2[["earthquake.time","earthquake_place"]]
# the difference is more square brackets.
print(type(data2["earthquake_place"]))
print(type(data2[["earthquake_place"]]))
# slicing and indexing
# from index 1 to index 3
# from earthquake.time to earthquake_place
data2.loc[1:3,"earthquake.time":"earthquake_place"]
# reverse slicing
data2.loc[10:1:-1,"earthquake.time":"earthquake_place"]
# from something to end
data2.loc[1:10,"earthquake.mag":]
condition = data2["earthquake.mag"] >= 7
data2[condition]
# combining filters
firstFilter = data2["earthquake.mag"] >= 7
secondFilter = data2["Venus.speed"] >= 1
data2[firstFilter & secondFilter ][["earthquake.mag","Venus.speed"]]
# filtering column based
# show me Venus.speed that earthquake.mag >= 7
data2["Venus.speed"][data2["earthquake.mag"] >= 7]
def div(a):
    return a*10
data2["earthquake.mag"].apply(div)

# or use lambda function
data2["earthquake.mag"].apply(lambda a: a*10)
#definin column using other columns
data2["new_column"] = data2["earthquake.mag"]+data2["MoonPhase.total"]
data2.head()
# look at our index name and change it
print(data2.index.name)
data2.index.name = "index_name"
data2.head()
# overwrite index
# if we want to modify, we need to change all of them
data2.head()
data3 = data2.copy()
data3.index = range(100,110,1) # start, stop, step
data3
data = pd.read_csv("../input/earthquakes-solar-system-objects/SolarSystemAndEarthquakes.csv")
data.head()

# outer and inner index
data1 = data.set_index(["earthquake.mag","Venus.speed"])
data1.head(10)
data = pd.read_csv("../input/earthquakes-solar-system-objects/SolarSystemAndEarthquakes.csv")
data.head()
data.pivot(columns ="MoonPhase.dynamic",values ="earthquake.mag").head(10)
Item = ['Item0', 'Item0', 'Item1', 'Item1']
cType = ['Gold', 'Bronze', 'Gold', 'Silver']
USD = [2, 4, 6, 8]
EU = ['1', '2', '3', '4']

list_label = ["Item","cType","USD","EU"]
list_col = [Item,cType,USD,EU]
list_col

zipped = list(zip(list_label,list_col))
zipped

dictionary = dict(zipped)
dictionary

dataFrame = pd.DataFrame(dictionary)
dataFrame

dataFrame1 = dataFrame.set_index(["Item","cType"])
dataFrame1

# and then we will unstack index 0 with level = 0
#dataFrame1.unstack(level=0)

# unstack index 1 with level = 1
#dataFrame1.unstack(level=1)

# please try yourself  with comment lines
# change inner and outer level index position
dataSwap = dataFrame1.swaplevel(0,1)
dataSwap
dataFrame1
pivotted = dataFrame.pivot(index ='Item', columns='cType', values='USD')
pivotted
melted = pd.melt(dataFrame, id_vars="Item",value_vars=["cType","USD"])
melted
dataFrame
dataFrameGrouped = dataFrame.groupby('cType').mean()
dataFrameGrouped
# we can only choose one of the feature
dataFrame.groupby("Item").USD.max() 
# Or we can choose multiple features
dataFrame.groupby("Item")[["USD","EU"]].min() 
dataFrame.info()
# this is show us,
# A variable is also object, if we use groupby, we can convert it to categorical data.
# categorical data uses less memory.
dataFrame["Item"] = dataFrame["Item"].astype("category")
dataFrame.info()