import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import os
os. getcwd()
df = pd.read_csv("../input/energyconsumedstuser/StudentServicesKaggle.csv")#to Have Index as Column df.reset_index()
df

df.describe()
december=df[df["Month"]== 12]
december=december.reset_index(drop=True)
december["NewIndex"]=decemberNI.index+1
december
print("Days in December: ",pd.unique(december["Day"]).size)
print("Entries in December: (31*24*60)/15=",len(december))
plt.plot(december["NewIndex"],december["UsedKW"])
plt.plot(december["NewIndex"],december["GeneratedKWS"])
plt.xlabel('December KW: Used and Generated')
plt.ylabel('KW')
plt.legend()
oneDay=december[december["Day"]==1]
plt.plot(oneDay["NewIndex"],oneDay["UsedKW"])
plt.xlabel('One Day Uses KW (Dec 1:Sunday)')

count=1
while count<=7:
    day=december[december["Day"]==count]
    count +=1
    plt.plot(day["NewIndex"],day["UsedKW"])
    plt.xlabel('One Week Uses KW')

day=1
while day<=pd.unique(december["Day"]).size:
    days=december[december["Day"]==day]
    day +=1
    plt.plot(days["NewIndex"],days["UsedKW"])
    plt.xlabel('One Month Uses KW')
print("There is a pattern where Sunday and Saturday is low and other Days are High \
      \nFirst three weeks are normal but last week is inconsistent because of holidays")
#Modeling based on Day 4 because it looks like a average of Weekdays
fourDay=december[december["Day"]==4]
plt.plot(fourDay["NewIndex"],fourDay["UsedKW"])
plt.xlabel('Day 4')
count=0
points=np.polyfit(fourDay["NewIndex"],fourDay["UsedKW"],5)
construction=np.poly1d(points)
prediction=construction(fourDay["NewIndex"])
plt.plot(fourDay["NewIndex"],fourDay["UsedKW"])
plt.plot(fourDay["NewIndex"],prediction)
print("Model Equation: \n",construction)
