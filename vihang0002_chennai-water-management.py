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
from datetime import datetime

df1=pd.read_csv("/kaggle/input/chennai_reservoir_levels.csv")
df1
x=np.array(df1["Date"])

type(x[1])

for i in range(len(x)):

    df1["Date"][i]=datetime.strptime(x[i], '%d-%m-%Y')
type(x[1])
df1["TOTAL"]=df1["POONDI"]+df1["CHOLAVARAM"]+df1["REDHILLS"]+df1["CHEMBARAMBAKKAM"]
import matplotlib.pyplot as plt

x=np.array(df1["Date"])

y=np.array(df1["POONDI"])

plt.plot(x, y, label = "line 1",color="orange")

plt.xlabel('Date')



plt.ylabel('Water-Level')

plt.title('POONDI')

plt.legend()

plt.show()
x1=np.array(df1["Date"])

y1=np.array(df1["CHOLAVARAM"])

plt.plot(x1, y1, label = "line 2",color="red")

plt.xlabel('Date')

plt.ylabel('Water-Level')

plt.title('CHOLAVARAM')

plt.legend()

plt.show()
x2=np.array(df1["Date"])

y2=np.array(df1["REDHILLS"])

plt.plot(x2, y2, label = "line 3",color="green")

plt.xlabel('Date')

plt.ylabel('Water-Level')

plt.title('REDHILLS')

plt.legend()

plt.show()
x3=np.array(df1["Date"])

y3=np.array(df1["CHEMBARAMBAKKAM"])

plt.plot(x3, y3, label = "line 4",color="blue")

plt.xlabel('Date')

plt.ylabel('Water-Level')

plt.title('CHEMBARAMBAKKAM')

plt.legend()

plt.show()
x3=np.array(df1["Date"])

y3=np.array(df1["TOTAL"])

plt.plot(x3, y3, label = "Line 5",color="black")

plt.xlabel('Date')

plt.ylabel('Water-Level')

plt.title('TOTAL')

plt.legend()

plt.show()
df2=pd.read_csv("/kaggle/input/chennai_reservoir_rainfall.csv")
df2
x=np.array(df2["Date"])

for i in range(len(x)):

    df1["Date"][i]=datetime.strptime(x[i], '%d-%m-%Y')
x

DATE=[]

for i in range(16):

    DATE.append(2004+i)

DATE
#for i in range(366):

    #print(df2["POONDI"][i])

k=0



    

for i in range(len(df2["Date"])):

    string=df2["Date"][i]

    substring=str(DATE[0])

    l=string.count(substring)

    if l>0:

        k=k+1

print(k)

df3=df2[:k]

list=[]

total=[]

total1=0

total2=0

total3=0

total4=0

for i in range(len(df3["POONDI"])):

    list.append(df3["POONDI"][i])

    total1=total1+list[i]

total.append(total1)

list=[]

for i in range(len(df3["POONDI"])):

    list.append(df3["CHOLAVARAM"][i])

    total2=total2+list[i]



total.append(total2)

list=[]

for i in range(len(df3["POONDI"])):

    list.append(df3["REDHILLS"][i])

    total3=total3+list[i]



total.append(total3)

list=[]

for i in range(len(df3["POONDI"])):

    list.append(df3["CHEMBARAMBAKKAM"][i])

    total4=total4+list[i]



total.append(total4)

print(total)

x=["1","2","3","4"]



plt.bar(x,total,width = 0.8)

plt.xlabel(str(DATE[0])) 

plt.ylabel('Rainfall(mm)')

plt.legend()

plt.title('')

plt.show() 

k=0

for i in range(len(df2["Date"])):

    string=df2["Date"][i]

    substring=str(DATE[1])

    l=string.count(substring)

    if l>0:

        k=k+1

        m=366

print(k)

df3=df2[m:m+k]

list=[]

total=[]

total1=0

total2=0

total3=0

total4=0

for i in range(len(df3["POONDI"])):

    list.append(df3["POONDI"][366+i])

    total1=total1+list[i]

total.append(total1)

list=[]

for i in range(len(df3["POONDI"])):

    list.append(df3["CHOLAVARAM"][366+i])

    total2=total2+list[i]



total.append(total2)

list=[]

for i in range(len(df3["POONDI"])):

    list.append(df3["REDHILLS"][366+i])

    total3=total3+list[i]



total.append(total3)

list=[]

for i in range(len(df3["POONDI"])):

    list.append(df3["CHEMBARAMBAKKAM"][366+i])

    total4=total4+list[i]



total.append(total4)

print(total)

x=["1","2","3","4"]



plt.bar(x,total,width = 0.8)

plt.xlabel(str(DATE[1])) 

plt.ylabel('Rainfall(mm)')

plt.legend()

plt.title('')

plt.show() 

k=0

for i in range(len(df2["Date"])):

    string=df2["Date"][i]

    substring=str(DATE[2])

    l=string.count(substring)

    if l>0:

        k=k+1

        m=366+365

print(k)

df3=df2[m:m+k]

list=[]

total=[]

total1=0

total2=0

total3=0

total4=0

for i in range(len(df3["POONDI"])):

    list.append(df3["POONDI"][366+365+i])

    total1=total1+list[i]

total.append(total1)

list=[]

for i in range(len(df3["POONDI"])):

    list.append(df3["CHOLAVARAM"][366+365+i])

    total2=total2+list[i]



total.append(total2)

list=[]

for i in range(len(df3["POONDI"])):

    list.append(df3["REDHILLS"][366+365+i])

    total3=total3+list[i]



total.append(total3)

list=[]

for i in range(len(df3["POONDI"])):

    list.append(df3["CHEMBARAMBAKKAM"][366+365+i])

    total4=total4+list[i]



total.append(total4)

print(total)

x=["1","2","3","4"]



plt.bar(x,total,width = 0.8)

plt.xlabel(str(DATE[2])) 

plt.ylabel('Rainfall(mm)')

plt.legend()

plt.title('')

plt.show() 
k=0

for i in range(len(df2["Date"])):

    string=df2["Date"][i]

    substring=str(DATE[3])

    l=string.count(substring)

    if l>0:

        k=k+1

        m=366+365+365

print(k)

df3=df2[m:m+k]

list=[]

total=[]

total1=0

total2=0

total3=0

total4=0

for i in range(len(df3["POONDI"])):

    list.append(df3["POONDI"][366+365+365+i])

    total1=total1+list[i]

total.append(total1)

list=[]

for i in range(len(df3["POONDI"])):

    list.append(df3["CHOLAVARAM"][366+365+365+i])

    total2=total2+list[i]



total.append(total2)

list=[]

for i in range(len(df3["POONDI"])):

    list.append(df3["REDHILLS"][366+365+365+i])

    total3=total3+list[i]



total.append(total3)

list=[]

for i in range(len(df3["POONDI"])):

    list.append(df3["CHEMBARAMBAKKAM"][366+365+365+i])

    total4=total4+list[i]



total.append(total4)

print(total)

x=["1","2","3","4"]



plt.bar(x,total,width = 0.8)

plt.xlabel(str(DATE[1])) 

plt.ylabel('Rainfall(mm)')

plt.legend()

plt.title('')

plt.show() 