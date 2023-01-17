# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#user defined function
def return_tuple():
    t = (1,2,3)
    return t #returning t tuple
a,s,d = return_tuple() #a defined as 1,b 2, c 3
print(a,s,d)
#scope (global,local,builtins)
x = 5
def func():
    x = 10       
    return x
print(x) 
print(func())
x = 5
def func1():
    return x
print(x)
print(func1())
import builtins
dir(builtins)
##nested functions are unified(func inside func) functions   example:
def f():
    e=a+2
    def f2():
        e=3
        return e
    return a
print(f())
##default and flexible arguments 
def k(a,b,c=4):
    a = b + c
    print(a)
k(1,2,3)    
#flexible *args
def k2(*a):
    for i in a:
        print(i)
k2(6,5,6,7)
print("")
k2(2)
print("")
k2()
#flexible *kwargs
def f_dict(**a):
    for key,value in a.items(): #-> provides to see dictionaries' key and value
        print(key,value)
f_dict(kind = "cake",flavour = "banana",piece = 12)
print(f_dict)
#making lambda function
def sqrt(a):
    return a**0.5 #this is common way when write funcs
print(sqrt(16))
sqrt2 = lambda a: a**0.5 # this is lambda way
print(sqrt2(16))

    
#anonymous func. -> map()
l = [1,2,3,4,5]
a = map(lambda a: a**0.5,l)  #this will apply function for all items in the list
print(list(a))
#zipping -- zip()  ->it is just like cartesian product in the maths, but a bit different
l1 = [1,2,3,4]
l2 = [5,6,7,8] 
new = list(zip(l1,l2))
print(new)
#unzipping
unzipped = zip(*new)
unlist1,unlist2=list(unzipped) #unzip returned tuple
print(type(unlist1))

##LIST COMPREHENSION
liste = [1,2,3,4,5]
liste2 = [i+2 for i in liste] 
print(liste2)
#another example
liste3 = [i+5 if i>3 else i-2 if i <= 3 else i+3 for i in liste]
print(liste3)         #else i-2 if i <= 3 -> replace elif
                     # elif i<=3:
                    #     i = i-2  
data = pd.read_csv("../input/metal_bands_2017.csv", encoding = "ISO-8859-1")
avrg = sum(data.fans)/len(data.fans)
data["admiration"] = ["high" if i > avrg else "low" for i in data.fans]
print(data.admiration,data.fans)