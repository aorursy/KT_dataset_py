#### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#create dataset by yourself (BUILDING DATA FRAMES FROM SCRATCH)
#way 1
dict1 = {"kind": ["cake","donut","ice cream"], "prize": [10,5,3]}
firstdataframe = pd.DataFrame(dict1)
firstdataframe
#way 2
employe_name = ["Jonathan","Stan","Chris"]     #list 1
salary = [150,170,125]                        #list 2
list_label = ["employe name","salary"]       #columns' label
list_column = [employe_name,salary]           #list which used in below of column
zipped = list(zip(list_label,list_column))     #zipping 2 columns, and making list from them 
makedict = dict(zipped)                         #making dictionary 
dataframe2nd = pd.DataFrame(makedict)          #making data frame from dictionary 
dataframe2nd
#adding a column
dataframe2nd["departmant"] = ["accountancer","manager","editor"]
dataframe2nd["new column"] = 0 #broadcasting  - add new column and give same values for all rows
dataframe2nd
data = pd.read_csv("../input/hfi_cc_2018.csv")
data.plot() #making plot from all data - there are lots of columns, this plot is confusing. let's make it for some columns
data1 = data.loc[:,["pf_rol_criminal","pf_religion"]]     #pf=personal freedom rol=rules of law - loc[:,["pf_rol_criminal","pf_religion"]] ->: all rows
data1.plot()
data1.plot(subplots = True) #subplots=True -> set plots in different areas but in same plot 
plt.show()
#scatter plot
data1.plot(kind="scatter",x="pf_rol_criminal",y="pf_religion")
plt.show()
#way 2
data1.plot.scatter(x="pf_rol_criminal",y="pf_religion",c="grey",alpha=0.5)
plt.show()           #c->color alpha->opacity 
#histogram
data1.plot(kind = "hist",y = "pf_rol_criminal",bins = 40,range= (0,50),normed=True)
#bins = 40 -> lines' thickness, range= (0,50)->detect last number on y line- normed=True -> normalizing frequency between 0-1
import warnings as w
w.filterwarnings("ignore")
fig, axes = plt.subplots(nrows=2,ncols=1)#nrows-ncols : number of rows and columns
data1.plot(kind = "hist",y = "pf_rol_criminal",bins = 50,range= (0,20),ax = axes[0],normed=True) #axes[0] -> column wise  axes[1]-> row wise
data1.plot(kind = "hist",y = "pf_rol_criminal",bins = 50,range= (0,20),ax = axes[1],normed=True,cumulative = True) #cumulative = True -> as the name implies, previous plus next, plus next.. (it's like fibonacci numbers)
#INDEXING PANDAS TIME SERIES
times = ["2019-01-24","2004-01-08","1923-04-23"] #make a normal list, includes date
print(type(times[1]))  #elements of list are strings yet
datetimeObject = pd.to_datetime(times) #str converted to datetime object
print(type(datetimeObject))
#let's use it in a dataset
data2 = data.head() #top 5
data2             #this dataset already have years but i will use that i will add
datelist = ["2004-01-08","2004-01-23","1999-09-23","1999-01-10","2004-02-23"] #make a list includes date
datetObject = pd.to_datetime(datelist)   #making datetime object from list
print(type(datetObject)) #confirm lists' elements are date time object  
data2["dates"] = datetObject #make a new column and put elements under it
datanew = data2.set_index("dates",inplace=True) #make new column index
datanew
print(data2.loc["1998-01-19":"2000-03-16"])      # loc["a-b-c","d-e-f"]-- now i can learn how many dates between the dates i want
print(data2.loc[:"2004-02-20"])
data2.resample("M").mean() #monthly mean
data2.resample("A").mean() #yearly mean
data2.resample("A").first().interpolate("linear") #first("xD") : get the rows for the first x days
#interpolate("linear") : ignore the index and treat the values as equally spaced
data2.resample("M").mean().interpolate("linear")  