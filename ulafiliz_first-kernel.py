# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#ödev1

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as mpl

import seaborn as sns



our_data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

our_data.info()

f, ax = mpl.subplots(figsize=(18, 18))

corr_map = sns.heatmap(our_data.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

mpl.show()



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as mpl

import seaborn as sns



#what is the relation between variables?

our_data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

our_data.info()

f, ax = mpl.subplots(figsize=(18, 18))

corr_map = sns.heatmap(our_data.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

mpl.show()



our_data.head(10)

our_data.columns

our_data.shape

our_data.info



our_data.plot(kind = 'hist',x = 'writing score', y = 'math score' ,color = 'r',label = 'math score',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
#ödev2

#determining the failed, taken half of the both notes and indicated that pass note is 60

passnote = 60

our_data["situation of failure"] = ["passed" if i >=passnote else "failed" for i in (our_data["math score"]/2 + our_data["reading score"]/2)]



print(our_data["situation of failure"])



#separating failed and passed student values for statistics.

failures = []

prouds = []

zipped =zip(list(our_data["math score"]) , list(our_data["reading score"]))

for i in zipped:

    for j in our_data["situation of failure"]:

        if  j == 'failed':

            failures.append(i)

        else:

            prouds.append(i)





print("students passed", (len(prouds)/1000))

print("students failed", (len(failures)/1000))
#ödev3

#extracting the data of students from given data (not yet cleaned.)

print(our_data.gender.value_counts(dropna = True))

print(our_data["race/ethnicity"].value_counts(dropna = False))

print(our_data["parental level of education"].value_counts(dropna = False))

print(our_data["math score"].value_counts())

print(our_data["reading score"].value_counts(dropna = False))
our_data.describe() 
#usşng boxplot graphic

our_data.boxplot(column='math score',by = 'gender')
#this part is taken from data science tutorial kernel

data_new = our_data.head()    # I only take 5 rows into new data

print(data_new)

# let's melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'gender', value_vars= ['math score', 'reading score', 'writing score'])

melted

# This data will be cleared!

melted.pivot(columns ='variable', values = 'value')#if I use index here, because that it consists of string values, code makes an error.
parental = our_data["parental level of education"].head()

math =our_data["math score"].head()

reading =our_data["reading score"].head()

writing =our_data["writing score"].head()

cnctd = pd.concat([parental, math, reading, writing], axis = 1, ignore_index =0)#took these values to compare the level of education and scores



print(cnctd)



f, ax = mpl.subplots(figsize=(18, 18))

corr_map = sns.heatmap(cnctd.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

mpl.show()#using heatmap to show how much related reading, writing and math scores are.
our_data.dtypes

#our_data["situation of failure"] = our_data["situation of failure"].astype('category')

our_data.dtypes

#our_data["situation of failure"]

data1 = our_data["math score"] #we should do this because it doesn't work either way.

data1= our_data["math score"].dropna(inplace = True)



assert our_data["math score"].notnull().all()#controlling if there's any nan on the code, doesn't return anything.

our_data["math score"].fillna('female',inplace = True)#filling nan values with not entered statement
#ödev4

data = our_data.head()

data = data.loc[:,["math score","writing score","reading score"]]#plotting the locations of the exam results

data.plot(subplots = True)

mpl.show()

data.plot(kind = "scatter",x="reading score",y = "writing score")

mpl.show()#showing the linearity between reading and writing scores, it appears to be highly related and graphic is linear.



data.plot(kind = "hist",y = "math score",bins = 50,range= (0,100),normed = True)

mpl.show()

data.plot(kind = "hist",y = "math score",bins = 50,range= (0,100),normed = True, cumulative = True)

mpl.show()

data.plot(kind = "hist",y = "reading score",bins = 50,range= (0,100),normed = True)

mpl.show()

data.plot(kind = "hist",y = "reading score",bins = 50,range= (0,100),normed = True, cumulative = True)

mpl.show()

data.plot(kind = "hist",y = "writing score",bins = 50,range= (0,100),normed = True)

mpl.show()

data.plot(kind = "hist",y = "writing score",bins = 50,range= (0,100),normed = True, cumulative = True)

mpl.show()#showing the frequencies of the scores both cumulatively and normally.
our_data.describe()
list_of_time = ["1992-08-14","1992-08-15", "1992-08-16", "1992-08-17", "1992-08-18"]#assuming a list which includes the dates students got into the exam

list_of_time = pd.to_datetime(list_of_time)#transforming to datetime object using pandas library

data["exam date"] = list_of_time #adding to data 

data = data.set_index("exam date") #setting the index as exam date using set_index

print(data.loc["1992-08-16"])#showing the location of given date

print(data.loc[:"1992-08-16"])#showing the locations until given date



data.resample("M").mean()

#better to do it with month resampling.

data.resample("M").mean().interpolate("linear")#trying to use interpolate function but it gives the same results because the interval is not too wide

#ödev5

def scoring(math, reading, writing):

    avg = math /4 + reading/4 + writing/4

    return avg



our_data["average"] = scoring(our_data["math score"], our_data["reading score"], our_data["writing score"])



#better way to see the people who passed.

boolean = our_data["average"] > 70

our_data[our_data["average"] > 70]



our_data.index.name = "ind" #changing the index name to ind in order to prevent confusions.

our_data.tail()



# Overwrite index



# first copy of ouR_data to data3 then change index to prevent data lost

data3 = our_data.copy()

# lets make index start from 100. It is not remarkable change but it is just example

data3.index = range(100,1100,1)

data3.head()
# Setting index : gender is outer ethnicity is inner index

data1 = our_data.set_index(["gender","race/ethnicity"]) 

data1



# change inner and outer level index position and see it like this

df2 = data1.swaplevel(0,1)

df2

#let's start all over

our_data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

our_data.index.name = "ind"

our_data["average"] = scoring(our_data["math score"], our_data["reading score"], our_data["writing score"])

our_data.head()

data3 = our_data.copy()

pd.melt(data3,id_vars="gender",value_vars=["average","reading score"])#melting

 
our_data.groupby("gender").mean() #that's a useful work
# We can choose multiple features also

our_data.groupby("gender")[["math score","reading score", "writing score"]].min() 