# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2017.csv")
# example of what we learn above

def tuple_ex():

    """ return defined t tuple"""

    t = [1,2,3]

    return t

a,b,c = tuple_ex()

print(a,b,c)
# guess print what

x = 2

def f():

    x = 3

    return x

print(x)      # x = 2 global scope

print(f())    # x = 3 local scope
# What if there is no local scope

x = 5

def f():

    y = 2 * x        # there is no local scope x

    return y

print(f())         # it uses global scope x

# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.
# How can we learn what is built in scope (reserved words)

import builtins

dir(builtins)
#nested function

def square():

    """ return square of value """

    def add():

        """ add two local variable """

        x = 2

        y = 3

        z = x + y

        return z

    return add() ** 2

print(square())    
# default arguments

def f(a, b = 1, c = 2):

    y = a + b + c

    return y

print(f(5))

# what if we want to change default arguments

print(f(5,4,3))

# flexible arguments *args

def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,4)

# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    """ print key and value of dictionary"""

    for key, value in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop

        print(key, " ", value)

f(country = 'spain', capital = 'madrid', population = 123456)
# lambda function

square = lambda x : x ** 2     # where x is name of argument

print(square(4))

tot = lambda x,y,z : x + y + z   # where x,y,z are names of arguments

print(tot(1,2,3))
number_list = [1,2,3]

y = map(lambda x : x ** 2,number_list)

print(list(y))
# iteration example

name = "ronaldo"

it = iter(name)

print(next(it))    # print next iteration

print(*it)         # print remaining iteration
# zip example

list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)
un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip) # unzip returns tuble

print(un_list1)

print(un_list2)

print(type(un_list2))
# Example of list comprehension

num1 = [1,2,3]

num2 = [i + 1 for i in num1 ]

print(num2)
# Conditionals on iterable

num1 = [5,10,15]

num2 = [i ** 2 if i == 10 else i - 5 if i < 7 else i + 5 for i in num1]

print(num2)
# Lets return 2017 csv and make one more list comprehension example

# Lets classify countries whether they have high or low family rate. Our threshold is average family rate.

threshold = sum(data.Family)/len(data.Family)

data["family_level"] = ["high" if i > threshold else "low" for i in data.Family]

data.loc[:10,["family_level","Family"]] # we will learn loc more detailed later