import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv(r'../input/indian-food-101/indian_food.csv')

df.head()
corelation =df.corr() 
sns.heatmap(corelation,xticklabels=

corelation.columns, yticklabels=

corelation.columns, annot=True)            
sns.pairplot(df) #pairplot
sns.countplot(x='ingredients',data=df)   #bargraph_for_ingredients
sns.countplot(x='diet',data=df)  #bargraph_for_diet
sns.countplot(x='diet',hue='prep_time', data=df) 
#line_chart

sns.lineplot(x='prep_time',y='cook_time',data=df)    
#scatter_plot

sns.relplot(x='prep_time',y='cook_time',hue='state',data=df) 
df.plot.scatter(x='prep_time',y='state')  
#histogram

sns.distplot(df['prep_time'],bins=7)
#piechart

sizes=df['diet'].value_counts()

fig1, ax1=plt.subplots()

ax1.pie(sizes,labels=['vegetarian','non vegetarian'], 

autopct='%1.1f%%',shadow=True)

plt.show()