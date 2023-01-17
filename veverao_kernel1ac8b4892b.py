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
def ther(n):

    if(n>0):

        for i in range(1,n+1):

            print("\t",i)

    else:

        print("Enter a valid no.")



#n=int(input("Enter the ending no.: "))

n=10

ther(n)
def even(n):

    if(n>0):

        for i in range(0,n+1):

            if(i%2==0):

                print(i)

    else:

        print("Enter a no. greater than 0")

#n=even(int(input("Enter the last no.: ")))

n=5
def odd(n):

    if(n>0):

        for i in range(0,n+1):

            if(i%2!=0):

                print(i)

    else:

        print("Enter a no. greater than 0")

#n=odd(int(input("Enter the last no.: ")))

n=5

def fib(n,o,p):

    sum=0

    while(sum<p):

            sum=n+o

            print(sum,"\t")

            n=o

            o=sum

n=0

o=1

#p=int(input("Enter the upperlimit: "))

p=10

print(n,o)

fib(n,o,p)

    
def primen(q,n):

    for j in range(q,n+1):

        k=0

        if(j>1):

            for i in range(2,j//2+1):

                 if(j%i==0):

                       k=k+1

                 

            if(k<=0):

                print(j)

              

                

        

#q=(int(input("Enter the start.: "))) 

#n=(int(input("Enter the endno.: ")))

q=2

n=10

primen(q,n)