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
df=pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")

df.head(10)

df.suicides_no

df.population

list_p=df.population[0:15]

list_s=df.suicides_no[0:15]

print(list_p)

print(list_s)
#calc per ten-thousandts of suicides no.s in terms of population for first value

def tenth():

    def thos():

        p=list_p[0]

        s=list_s[0]

        r=s/p

        return r

    return thos()*100000

print(tenth())
def f(*args):

    for i in args:

        print(i)

f(list_s[:15])
def f(**kwargs):

    for key, value in kwargs.items():

        print(key,value)

f(country= df.country[0],suicides=df.suicides_no[0])
o=df.population[0]

t=df.population[1]

future=lambda o:o+t

print("total population of 15-54 man=", future(o))
pop=df.population[:15]

h=map(lambda x:x/2 , pop)

print("half of populations=",list(h))
country=df.country[0]

it=iter(country)

print(next(it))

print(*it)
a=zip(list_p,list_s)

print(a)

alist=list(a)

print("populations and its suicides nos:",alist)
unzip=zip(*alist)

unlist_p,unlist_s=list(unzip)

print(list(unlist_p))

print(list(unlist_s))
num1=list(df.suicides_no[:15])

num2=list([i*2 for i in num1])

print("doubled suicides nos:",list(num2))



pop1=df.population[:100]

pop2=["high" if i>300000 else "low" for i in pop1]

print("high means 300k and more population, low means 10k and under population=",pop2)
treshold=(sum(df.suicides_no[:15]))/(len(df.suicides_no[:15]))

print(treshold)

df["suicide_case"]=["high" if i>treshold else "low" for i in df.suicides_no]

df.loc[:15,["suicide_case","suicides_no"]]