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
# Loading the dataset 

df = pd.read_csv("/kaggle/input/car-price-prediction/CarPrice_Assignment.csv")
df
df.dtypes
df.info()
# Checking null values

print(df.isnull().sum())
df.columns
# Heat Map

import matplotlib.pyplot as plt
import seaborn as sns
correlation = df.corr()
plt.figure(figsize = (12 , 12))
sns.heatmap(correlation)
# 3D Scatter Plot

import plotly.express as px

fig = px.scatter_3d(df, x='CarName', y='carheight', z='carlength', color='CarName')
fig.show()
# plot

my_circle=plt.Circle( (0,0), 0.9, color='white')
plt.pie(df['CarName'].value_counts()[:10].values, labels = df['CarName'].value_counts()[:10].index)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
import numpy as np 
import plotly 
import plotly.graph_objects as go 
import plotly.offline as pyo 
from plotly.offline import init_notebook_mode 
  
init_notebook_mode(connected=True) 
  
# generating 150 random integers 
# from 1 to 50 
x = df['CarName']
  

y = df['price']
  
# plotting scatter plot 
fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', marker=dict( 
        color=np.random.randn(20), 
        colorscale='Viridis',  
        showscale=True
    ) )) 
  
fig.show() 
import plotly.express as px

fig = px.bar(df, x='carbody', y='enginesize', color='carbody', height = 500,width = 1100, text = 'enginesize', title = "Car body VS Engine size")
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()


df1 = df.select_dtypes(exclude=[np.number])
df1
df1.columns
# labelling

from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

le = LabelEncoder()
for col in df1.columns:
    df[col]=le.fit_transform(df[col])
df.head()
df.drop('car_ID',axis=1,inplace=True)
df
X= df.iloc[:,:-1].values
y= df.iloc[:,-1].values
print(X)

print(y)
x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
linear=LinearRegression()
linear.fit(x_train,y_train)
result = linear.predict(x_test)
from sklearn import metrics
metrics.r2_score(y_test,result)
# set the background style of the plot 
sns.set_style('whitegrid') 
sns.distplot(result, kde = False, color ='red', bins = 30) 
