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
## Dropping Duplicates from List



l=list(input("Enter the elements: ").split(","))

#print(l)

#print(len(l))

#using the list comprehension converting all string into int

l=[int(i) for i in l]

s=[]

for i in l:

    if i not in s:

        s.append(i)

print(s)
## Dropping Duplicates from List



l=[1,2,23,2,1,32]

#print(l)

#print(len(l))

#using the list comprehension converting all string into int

s=[]

for i in l:

    if i not in s:

        s.append(i)

print(s)
#ascending order without sort fn.

l=list(input("Enter elements: ").split(","))

l=[int(i) for i in l]

print(l)

for i in range(0,len(l)):

    for j in range(i+1,len(l)):

        if(l[i]>l[j]):

            t=l[j]

            l[j]=l[i]

            l[i]=t

print(l)
#ascending order without sort fn.



l=[23,79,1,100,52]

print(l)

for i in range(0,len(l)):

    for j in range(i+1,len(l)):

        if(l[i]>l[j]):

            t=l[j]

            l[j]=l[i]

            l[i]=t

print(l)
#check if a string is a palindrome

def palin(str1):

    if(len(str1)==0):

        return True

    elif(len(str1)>0):

        r=str1[::-1]

        if(r==str1):

            return True

        else:

            return False

    



str1=input("Enter the string: ")

n=palin(str1)

print(n)
#check if a string is a palindrome

def palin(str1):

    if(len(str1)==0):

        return True

    elif(len(str1)>0):

        r=str1[::-1]

        if(r==str1):

            return True

        else:

            return False

    



str1="MALAYALAM"

n=palin(str1)

print(n)