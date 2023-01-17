# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def func():

    "returns the defined numbers"

    t = (1,2,3)

    return t

(b,c,a) = func()

print(b,c,a)



print('=======================')



def multiply(x,y):

    "multiplication of the given numbers"

    return x*y

def addition(x,y):

    "addition of the given numbers"

    return x+y

x=8

y=10



print(multiply(x,y), "and", addition(x,y))
x = 8

def f():

    x = 4

    return x

print(x) # x = 2 global scope

print(f()) #
import builtins

dir(builtins)
def square():

    "takes the square of the number"

    def add():

        r = 2

        t = 8

        z = r+t

        return z

    return add()**2

print(square())
def f(a,b = 1, c = 2):

    y = a + b + c

    return y

print(f(5)) # we only changed the 'a' because other ones are default

print(f(5, 4, 3)) # now we changed the default ones too
def f(*args):

    for i in args:

        print(i)

f(2)

print("====================")

f(1,55,22,11)

print("^^^^^^^^^^^^^^^^^^^^")



def func(**kwargs):

    for key, value in kwargs.items():

        print(key, ":", value)

func(area = 'Besiktas', value = '150,000$')
square = lambda x: x**2 #square of x (short way of def())

print(square(2))



tot = lambda x,y,z : x+y+z

print(tot(4,2,4))
citylist=["İstanbul","London","New Delhi"]

y=map(lambda x:x+" city",citylist)

print(list(y))



print("or diffent example")



numberlist=[99,11,23,float(0.54)]

a=map(lambda o:o+12,numberlist)

print(list(a))
num1 = [8,7,5]

result = [i*5 for i in num1]

print(result)
notes = [85,49,38,98] # an easy way to make sure that who is passed or not

situation = ["Failed" if i < 50 else "Passed" for i in notes]

print(situation)
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

data
data["age"]

age_average = sum(data.age)/len(data.age)

print(age_average)

print("===========")

data["age_situation"] = ["Old" if i>=age_average else "Young" for i in data.age]

data.loc[:, ["age_situation", "age"]]