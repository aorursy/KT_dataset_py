# import the basic libraries

import numpy as np 

import pandas as pd 

import seaborn as sns

from matplotlib import pyplot as plt
# Reading the stores dataset

stores = pd.read_csv("../input/stores.csv")

stores.head()
# Histogram

# method 1: use pandas Dataframe attributes

stores.TotalSales.plot(kind = "hist", edgecolor = "black", bins = 5)

# kind arg is used to specify the type of the graph

# bins = how many groups you want

plt.show()

# plt.show is used to remove the print comments
# we can also do this:

stores.TotalSales.plot.hist(bins = 5, edgecolor = "black", color = "tomato")

plt.show()
# Method 2

# using plt

plt.hist(stores.TotalSales, bins = 5, color = "skyblue", edgecolor = "black")

plt.show()
plt.hist(stores.TotalSales, bins = 5, color = "skyblue", edgecolor = "black")

plt.title("Histogram for total sales")

plt.xlabel("Total sales from stores data")

plt.ylabel("Frequency")

plt.show()
# Method 3

# Using seaborn distplot = ditribution plots

sns.distplot(stores.TotalSales, bins = 5, color = "pink")

plt.show()

# seaborn by default shows density on the y axis which you can turn off using density
stores.OperatingCost.plot(kind = "box")

plt.show()
# Method 1:

plt.scatter(x = list(range(0,32)), y = stores.TotalSales, s = 200, marker = "*")

plt.scatter(x = list(range(0,32)), y = stores.OperatingCost*10, s= 200 ,marker = "+")

plt.xlabel("Indexes")

plt.ylabel("total sales from the stores data")

plt.show()
sns.distplot(stores.TotalSales)

plt.show()
plt.scatter(x = stores.OperatingCost, y = stores.TotalSales)

plt.show()
# getting a scatter plot is not possible with the pandas plot

# Method 2: Using seaborn

# Every scatter plot has a concept of best fit line. That line virtually passes through each and every point.

# This process is called linear model plot. Y = mx+c



# Seaborn doesn't accept pandas.series as input

sns.lmplot(x = "OperatingCost", y="TotalSales", data = stores)

plt.xlabel("Operating Cost or expenses")

plt.ylabel("Income or TotalSales")

plt.title("Relation between Operating cost and Total sales")

plt.show()

# you can also apply matlab file here to get the plot

# to deactivate the line put fit_reg = False



# can we add more variables

# add a new categorical variable as color use hue and you can use palette to give custom colors

sns.lmplot(x = "OperatingCost", y="TotalSales", data = stores, hue = "Location", fit_reg = False)

plt.xlabel("Operating Cost or expenses")

plt.ylabel("Income or TotalSales")

plt.title("Relation between Operating cost and Total sales")

plt.show()
# Frequency Bar

# What is the frequency distribution of storetypes in the dataset stores?

stores.StoreType.value_counts()
f1 = stores.groupby(["StoreType"])[["StoreType"]].count().add_prefix("CountOf_")

f1
f2 = stores.groupby(["StoreType"])[["StoreType"]].count().add_prefix("CountOf_").reset_index()

f2
# method 1 usina pandas

f1.plot(kind = "bar")

plt.show()
# Method 2:

# using pyplot

# we need f2 since f1 syntax is not suitable 

plt.bar(x = f2.StoreType, height = f2.CountOf_StoreType)

plt.xlabel("Counts of StoreType")

plt.ylabel("Store Type")

plt.show()
# Method 3: Using Seaborn

sns.barplot(x = "StoreType" , y ="CountOf_StoreType", data = f2)

plt.show()
# Task 2

# w.r.t location, what is the % TotalSales in each location/city?

r1 = stores.groupby(by = ["Location"])[["TotalSales"]].sum().add_prefix("sumOf_")

r1
r2 = r1.reset_index()

r2
r1.plot(kind = "bar")

plt.show()
plt.bar(x = r2.Location, height = r2.sumOf_TotalSales)

plt.show()
sns.barplot(x = "Location", y = "sumOf_TotalSales", data = r2)

plt.show()
r1
## Slight change in the question. instead of total numbers if we want to see how much the contribution is ?

r1 = stores.groupby(by = ["Location"])[["TotalSales"]].sum().add_prefix("SumOf_")

r1

GrandTotalSales = r1.SumOf_TotalSales.sum()

GrandTotalSales

r1["PctSales"] = round(r1.SumOf_TotalSales/GrandTotalSales * 100,2)

r1.drop(columns=["SumOf_TotalSales"], inplace = True)

r1
r1.plot(kind = "bar", legend = False)

plt.ylabel("% contribution")

plt.show()
r2 = r1.reset_index()

sns.barplot(x = "Location", y = "PctSales", data = r2)

plt.show()
# Task 

# get mean of operating cost for each location and store type



r3 = stores.groupby(by = ["Location","StoreType"])[["OperatingCost"]].mean().add_prefix("MeanOf_")

r3
r4 = r3.reset_index()

r4
r3.plot(kind = "bar")

plt.show()
# a more elegant version will be stacket barplot, 

# r3 is in long format. 

# convert r3 to the wide format

# r4
# Type2

r4_wide = r4.pivot(index  = "Location", columns = "StoreType", values = "MeanOf_OperatingCost")

r4_wide
# Dodged bar

r4_wide.plot(kind = "bar")

plt.show()
# Stacked bar

r4_wide.plot(kind = "bar", stacked = True)

plt.show()
# using seaborn

# we definetely have to use seaborn

sns.barplot(x  = "Location", y = "MeanOf_OperatingCost", data = r4, hue = "StoreType")

plt.show()
# stacked bar

sns.barplot(x  = "Location", y = "MeanOf_OperatingCost", data = r4, hue = "StoreType", dodge = False)

plt.show()



# there is a bug in the seaborn hence people preferably use barplot
_temp = stores.groupby(by = ["Location"])

a = _temp[["TotalSales"]].mean().add_prefix("SumOf_")

b = _temp[["OperatingCost"]].sum().add_prefix("MeanOf_")

Result_4 = pd.concat(objs=[a,b], axis = 1)

Result_4
Result_4_a = Result_4.reset_index()

Result_4_a
Result_4.plot.bar()

plt.show()
Result_4_a_long = pd.melt(Result_4_a, id_vars = "Location", value_name = "Values",var_name = "Metrics")

Result_4_a_long
sns.barplot(x = "Location", y = "Values", data = Result_4_a_long, hue = "Metrics")

plt.show()
Result_4.plot.bar(stacked = True)

plt.show()
# pie charts

f1.plot(kind = "pie", subplots =True, legend = False, autopct = "%.2f%%")

plt.ylabel("")

plt.show()
r3.plot(kind = "pie", subplots = True, legend = False, autopct = "%.2f%%")

plt.legend = False

plt.ylabel("")

plt.show()
r4 = r3.reset_index()

r4
r4_wide = r4.pivot(index = "Location", columns = "StoreType", values = "MeanOf_OperatingCost")

r4_wide
r4_wide.plot(kind = "pie", subplots = True, legend = False, figsize = (15,8))



plt.show()
r4_wide.T.plot(kind = "pie", subplots = True, legend = False, figsize = (15,8))



plt.show()