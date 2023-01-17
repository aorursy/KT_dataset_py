# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

#lists can store any type of data
#creating a list
mixed_data = [1,2,3,'A','B','C']
print("Create",mixed_data)

#appending an element
mixed_data.append('D')
print("Append",mixed_data)

#Extend list by appending all elements from the iterable
new_data = ['Z']
new_data.extend(mixed_data)
print("Extend", new_data)

#Remove first element with matching value
new_data.remove('A')
print("Removing A",new_data)

#Sorting
try:
    new_data.sort()
except:
    print("Sort should be failing due to incompatible types")

#Copying
copied_data = new_data.copy()
copied_data.remove('Z')
print("Copy",copied_data,new_data)

#Without Copy
n = new_data
n.remove('B')
print("Same content",n,new_data)

my_tuple = (1,2,'abc',[1])
print("Creating Tuple", my_tuple)

print("Accessing the first element", my_tuple[0])

#Tuples are immutable
try:
    my_tuple[0] = 2
except:
    print("TypeError Exception is thrown")

my_tuple[3].extend([2,3,4,5,6])
print("However this is possible", my_tuple)

#Create sets using curly braces
my_set = {'A','B','C'}
#Note empty sets are created using the set() method
empty_set = set()
print("My_Set: ",type(my_set),"Empty Set:",type(empty_set), "Dictionary:", type({}))

#operations
second_set = {'A'}
print('Difference: ',my_set.difference(second_set), 'Intersection:',my_set.intersection(second_set),'Union',my_set.union(second_set))

print('Difference: ',my_set - second_set, 'Intersection:',my_set & second_set,'Union',my_set | second_set)

#Useful for removing redundant elements
print(set(['A','B','C','A']))

try:
    my_set[0] = 'D'
except:
    print("Sets are immutable")

#creating dictionaries
my_dict = {1:'A',2:'B'}
empty_dict = {}
second_dict = dict(a = 1, b = 2, c = 3)
third_dict = dict([('a',1),('b',2),('c',3)])
print("[Creating a dictionary]",'my_dict',my_dict,'empty_dict',empty_dict,'second_dict',\
      second_dict,'third_dict',third_dict)

print("[Accessing elements]",'Key is an integer:',my_dict[1],'Key is a string:',third_dict['a'])

#adding an element
my_dict[3] = 'C'
print("Adding an element",my_dict)

my_dict['D'] = 4
print("Adding heterogenous elements",my_dict)

print("Accessing Keys:",my_dict.keys(),"Accessing Values:",my_dict.values())

try:
    my_dict[list(1,2)] = 2
except:
    print("Mutable type not allowed as key")



#Arrays
a = np.array([1,2,3])
print(a)

a = np.array([1.,2.,3.])
print(a)

a = np.array([[1,2],[3,4]])
print(a)

a = np.array([1,2],dtype=complex)
print(a)

#Creating an array from sub-classes
a = np.array(np.mat('1 2; 3 4'),subok=True)
print(a,type(a))
#looping through lists
my_list = [1,2,3,4,5,6,7,8,9]
for e in my_list:
    print(e)
    
#attach index
for i,e in enumerate(my_list):
    print("Index:",i,"Element",e)
    
#list comprehensions -> creating lists
print([x for x in range(10)]) #range(x) returns values within the intervall [0,x[
print("Even",[x for x in range(10) if x%2==0])
print("Odd",[x for x in range(10) if x%2==1])

#nested list comprehension
matrix = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
]
print("Nested List - Transposition",[[row[i] for row in matrix] for i in range(3)])
#looping through dictionaries
my_dict = {1:'A',2:'B',3:'C'}

#iterate over set of key value pairs
for k,v in my_dict.items():
    print("Key-Value",k,v)

#iterate over keys
for k in my_dict.keys():
    print("Key",k)
    
#dict comprehensions -> creating dictionaries
print("Dict Compr",{k:v for k,v in enumerate([1,2,3,4,5])})
print("Dict Compr",{x: x**3 for x in range(4)})
#looping through sets

my_set = {1,2,3,4,5,'A'}
for x in my_set:
    print(x)
    
my_tuple = ([1,2,3],{'A':1},1,"abc")
for x in my_tuple:
    print(x)
my_list = [1,2,3,4,5,6,7]
my_set = {1,2,3,4}
my_tuple = (1,[1,2,],1,2,4)
my_dict = {'A':1,'B':2}
print("Max",max(my_list),"Min",min(my_list),"Round",round(1.234))
print("Length","List:",len(my_list),"Set:",len(my_set),"Tuple:",len(my_tuple),"Dict:",len(my_dict))


list_one = [1,2,3]
list_two = ['A','B','C']

#Zip Makes an iterator that aggregates elements from each of the iterables.
zipped = zip(list_one,list_two)
print("Unpacked",*zipped)

my_string = """This is a multiline
string; hence writing a text across
multiple lines is possible"""

print("Remove Space",my_string.replace(" ",""))
print("Replace",my_string.replace(' ',os.linesep))
print('I can write anything here {0} {1} {2}'.format('One','Two','Three'))
print('The area is {Area}'.format(Area=100))
print('{0:d} {0:x} {0:o}'.format(10))
#template strings
from string import Template

s = Template('$Name is weak')
print(s.substitute(Name='Bruce Wayne'))



"""

Asterisk(*) used in function definition

Packing the passed arguments into a tuple

"""
def do_something(*a):
    print(a)

do_something(1,2,3,4,5)
do_something((1,2),{'A','B','C'})
do_something({1:2,2:2,3:2})
do_something([1,2,3,4,5])

def unpack(*a):
    print(*a) #unpack, as if each element was passed separately as an argument

unpack(1,2,3,4,5)
unpack((1,2),{'A','B','C'})
unpack({1:2,2:2,3:2})
unpack([1,2,3,4,5])

"""

Double Asterisk(**) used in function definition

Packing the passed arguments into a dictionary

"""
def do_something_2(**a):
    print(a)
    
do_something_2(a=2,b=3,c=4)


#argument designations must coincide with parameter names
def unpack_dict(a,b,c):
    print(a,b,c)

unpack_dict(**{'a':1,'b':2,'c':3})


import re

print("All but newline: ",re.search('.*','I am 22 years old').group(0))
print("Line End/String End: ",re.search('d$','I am 22 years old').group(0))
print("All but digits:",re.search('[^0-9]*','I am 22 years old').group(0))

p = re.compile(r'\bseminar\b')
m = p.search('This seminar is about data science and data engineering')
if m:
    print(m,m.group())
    print(p.search('I attend a seminar which is about cognition'))



func_simple = lambda x: x ** 2
print(func_simple(2))

func_simple = lambda x,y: x + y
print(func_simple(1,2),func_simple([1,2],[3,4]))

try:
    print(func_simple({1,2},{3,4}))
except:
    print("The + operator is unsupported by sets")

#return a function whose one summand is fixed
def incrementor(n):
    return lambda x: x + n

func = incrementor(100)
print(func(42),func(100))

"""
Some built-in functions (map,filter,reduce) expect an explicit function in order to
apply its operations

map: Return an iterator that applies function to every item of iterable, yielding the results.
filter: Construct an iterator from those elements of iterable for which function returns true.
reduce (needs to be imported): Reduces the sequence to a single value

"""

my_list = [1,2,3]
my_list_2 = [1,2,3]
#map
print(list(map(lambda x: x+2,my_list)))
print(list(map(lambda x,y: x+y,my_list,my_list_2)))
print(list(map(lambda x,y: x==y,my_list,my_list_2)))
print(list(map(lambda x,y: 'First: {0} Second: {1}'.format(x,y),my_list,my_list_2)))

#filter
my_list = [1,2,3,4,5,6,7,8,9,10]
print("Even:",list(filter(lambda x: x%2==0,my_list)))
print("Odd:",list(filter(lambda x: x%2,my_list)))

#reduce
from functools import reduce
#expression applied from left to right
print(reduce(lambda x,y: x+y,[1,2,3,4]))
print(reduce(lambda x,y: x*y,[1,2,3,4]))
print(reduce(lambda x,y: x-y,[1,2,3,4]))
print(reduce(lambda x,y: x/y,[1,2,3,4]))



"""
Scope Test
https://docs.python.org/3/tutorial/classes.html
"""
def scope_test():
    def do_local():
        spam = "local spam"

    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"

    def do_global():
        global spam
        spam = "global spam" #changes the global binding

    spam = "test spam"
    do_local()
    print("After local assignment:", spam)
    do_nonlocal()
    print("After nonlocal assignment:", spam)
    do_global()
    print("After global assignment:", spam)

scope_test()
print("In global scope:", spam)

class MyClass:
    i = 10 #class variable
    t = [] #mutable object
    
    def __init__(self,x):
        self.x = x #instance variable
        
    def unexpected_behaviour(self,n):
        self.t.append(n)

x = MyClass(1)
y = MyClass(2)

print("Class Variable",x.i,y.i)
print("Instance Variable",x.x,y.x)

#changing instance variable
MyClass.i = 20
print(x.i,y.i)

#be aware of the following example where a mutable object is shared among all instances
x.unexpected_behaviour(22)
y.unexpected_behaviour(66)
print(x.t)
"""
Inheritance

For C++ programmers: all methods in Python are effectively virtual.
"""

class A:
    def __init__(self):
        self.do_first()
    
    def do_first(self):
        self.do_second()
        
    def do_second(self):
        self.x = 10
        
    def __str__(self):
        return str(self.x)

class B(A):
    def __init__(self):
        super().__init__()
        
    def do_second(self):
        self.x = -99
a = B()
print(a)
"""
Creating Data frames
"""

# 6x4 array with random numbers by using numpy functions
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df)

df_nan = pd.DataFrame(np.random.randn(5,5))
df_nan = df_nan.reindex([1,2,3,4,5,6])
print(df_nan)

# creating dataframe by using a dictionary
df2 = pd.DataFrame({ 'A' : 1.,\
                    'B' : pd.Timestamp('20130102'),\
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),\
                    'D' : np.array([3] * 4,dtype='int32'),\
                    'E' : pd.Categorical(["test","train","test","train"]),\
                    'F' : 'foo' })
print(df2)

#creating dataframe by reading from csv file
df3 = pd.read_csv("../input/stack_network_nodes.csv")
df3.head()

"""
Accessing Data
"""
df3

#access by column name
df3['name']
#since column name is spacefree direct access is possible
df3.name
#print row information
df3.loc[0] #0th row
#df3.loc[0:10] #from 0th row until 10th row
#df3.loc[:] #all rows
"""
Viewiwng Data

To view a small sample of a Series or DataFrame object, use the head() and tail() methods. 
The default number of elements to display is five, but you may pass a custom number.
"""

df.head()

df.tail()
df.index
df.columns
df.values #To get the actual data inside a data structure, one need only access the values property:
df.info
df.describe() #describe() shows a quick statistic summary of your data
df.T #transposing your data
#check whether not NaN
df_nan.notna()
#check if there are NaN values
df_nan.isna()
#filling missing values
print(df_nan)
df_nan.fillna(0)
