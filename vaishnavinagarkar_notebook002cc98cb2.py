import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df=pd.read_csv(r'../input/indian-food-101/indian_food.csv')
df.head() 
x=df['prep_time'] 

x
y=df['cook_time']

y
plt.scatter(x,y)

plt.show() 
x_mean=np.mean(x)

x_mean
y_mean=np.mean(y)

y_mean
m=sum((x-x_mean)*(y-y_mean))/sum((x-x_mean)*(x-x_mean))

m
c=y_mean-m*x_mean

c
def predict_time(x):

    y_predicted=0.09586901450902668*x+31.54732147809569

    return y_predicted 

y_predicted =predict_time(x)

y_predicted
plt.scatter(x,y) 

plt.plot(x,y_predicted, color='black')  

plt.show()  
r_square=1-sum((y-y_predicted)*(y-y_predicted))/sum((y-y_mean)*(y-y_mean))

r_square
