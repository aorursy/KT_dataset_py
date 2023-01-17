import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df=pd.read_csv(r'../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df.head() 
x=df['Age']

y=df['CustomerID']

plt.scatter(x,y)

plt.show() 
m=0

c=0

L=0.001

count=75

n=float(len(x))
for i in range(count):

    y_predict =m*x+c

    m=m-(L/n)*sum(x*(y_predict -y))

    c=c-(L/n)*sum(y_predict -y)

print(m,c) 
plt.scatter(x,y)

plt.plot(x,y_predict, color ='black')

plt.show() 