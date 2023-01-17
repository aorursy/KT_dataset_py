# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

print('Hello world')
"""
Variables and Data Types
Function
Loops
Data Structures
oops
"""
#Integer
"""
x=10
print(x)
y=10.5
print(y)
z=10j
print(z)
print(type(z))
"""
#strings
"""
a='Praveen'
print(a)
print(len(a))
count=0
for i in a:
    print(i)
    count=count+1
print(count)

print(a[1:20])
#a[1]='k' Strings are immutable
"""


#List
"""
fruits=['apple','bannna','apple',30,40,30.6]
print(fruits)
fruits[1]='Mango'
print(fruits)
fruits.insert(2,'Cherry')
fruits.append('Grape')
print(fruits)
print(fruits.reverse())
print(fruits)
"""
#Distonary
"""
mylist={'name':'praveen',60:5678,60:456,60:456,'name':'navya'}
print(mylist)
print(mylist)
mylist['age']=45
print(mylist)
"""
"""
#tuple
c=('praveen',29,29,29)
print(c[1])
print(c.count(29))

"""
"""
#SET
s={10,10,20,30,'praveen'}
print(s)
#print(s[3])

d={'name':'praveen',60:5678,60:456,60:456,'name':'navya'}
l=list(d)
print(l)
x = y = z = "Orange"
print(z)

x = "awesome"

def myfunc():

  x = "fantastic"
  return x


def myFunction() :
  return True

if myFunction():
  print("YES!")
else:
  print("NO!")

print(myfunc())

print("Python is " + x)


x = b"Hello"
print(x)

d={'name':'praveen',60:5678,56:456,58:456,'name':'navya'}
for i,j in d.items():
    print(i,':',j)

if 'name1' in d:
    print('Yes')
    
    """

txt = """9134885104,10
9134885103,15
9134885104,1"""

list1= txt.splitlines()
'''
print(x)

for i in x:
    d={}
    l=i.split(',')
    d[l[0]]=l[1]
    '''
dic={}
list=[]
for i in list1:
    list.append(i.split(','))
for i in range(len(list)):
     if list[i][0] in dic.keys():# if key is present in the list, just append the value
         dic[list[i][0]].append(int(list[i][1]))
     else:
         dic[list[i][0]]= [] # else create a empty list as value for the key
         dic[list[i][0]].append(int(list[i][1]))

print(dic)

for v in dic.values():
    k=sum(v)


print(dic)

#ll=sum(int(item['9134885104'] for item in dic))
mic={}
lk=[]
for k,v in dic.items():
    print(k)
    for n in v:
        if n>10:
            lk.append(n*20)

        else:
            lk.append(n*10)

    m=sum(lk)
    mic[k]=m
    lk=[]

print(mic)
#print(lk)









