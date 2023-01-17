import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

init_notebook_mode(connected = True)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import os

for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
dt=data.copy() 
dt
dt.dtypes
dt.dtypes.value_counts()
dt.shape
dt.shape[0]
dt.shape[1]
dt.columns
dt.index
dt.axes
cat_ft = dt.select_dtypes(include = 'object').columns.tolist()

num_ft= dt.select_dtypes(exclude='object').columns.tolist()



print ('Categorical Feature :', cat_ft)

print ('\nNumeric Feature :' ,num_ft)

print ('\nNumber of Categorical Feature : ' , len(cat_ft))

print ('\nNumber of Numeric Feature : ' , len(num_ft))
type(dt)
dt.info()
dt.describe()
dt.describe().T
dt.describe(include=["O"])
dt.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(dt.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
corr = dt.corr()

f, ax = plt.subplots(figsize=(15, 12))

sns.heatmap(corr, linewidths=.5, vmin=0, vmax=1, square=True)

plt.show()
dt.head()
dt=dt.iloc[:,1:len(dt)]
dt.head()
dt.head(10)
dt.sample(5)
display(dt.head(3))
print(dt.head(5))
dt.tail()
dt.tail(5)
dt_rand=dt[(dt['HP']>80)&(dt['Attack']>110)]

dt_rand.head()
dt['New Column']=dt['HP']+dt['Attack']

dt.head()
dt.drop('New Column',axis=1,inplace=True)

dt.head()
dt.sort_values(by='Attack',ascending=False).head()
dt['Name'].sort_values().head()
dt['Type 1'].value_counts()
dt.Name.str.upper()
dt.Name.str.contains('Diancie')
dt.Name.str.replace('Diancie','Yusuf')
dt.iloc[[0,1]]
dt.iloc[1:5,1]
dt.index
def miss(dt):

    ttl = dt.isnull().sum().sort_values(ascending = False)

    prc = (dt.isnull().sum()/dt.isnull().count()*100).sort_values(ascending = False)

    miss_data  = pd.concat([ttl, prc], axis=1, keys=['Total', 'Percent'])

    return miss_data

    



msdf = miss(dt)

msdf.head()
dt.isnull().sum()
miss_cl = dt.columns[dt.isnull().any()].values

ttl_miss_cl = np.count_nonzero(dt.isnull().sum())

print('We have ' ,ttl_miss_cl  ,  'features with missing values: \n' , miss_cl)
plt.figure(figsize=(10,5))

sns.heatmap(dt.isnull(), yticklabels=False, cbar=False, cmap = 'winter')

plt.show()
null=dt['Type 2'].isnull()

dt[null]
ntnull=dt['Type 2'].notnull()

dt[ntnull]
gp=dt['HP'].between(60,80)

dt[gp]
len(dt['Type 2'].unique())
dt.nsmallest(5,columns="Attack")
dt.nlargest(5,columns="Speed")
dt.query('Name=="Ninjask"')
dt.Speed.plot(kind = 'line', color = 'b',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

dt.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '--')

plt.legend(loc='upper right')    

plt.xlabel('x')            

plt.ylabel('y')

plt.title('Plot')           

plt.show()
def scPlot(ttl, x_axis, y_axis, size, c_scal,x,y):

    trace = go.Scatter(x = x,y = y,mode = 'markers',marker = dict(color = y, size=size, showscale = True, colorscale = c_scal))

    lyt = go.Layout(hovermode = 'closest', title = ttl, xaxis = dict(title = x_axis), yaxis = dict(title = y_axis))

    fg = go.Figure(data = [trace], layout = lyt)

    return iplot(fg)

scPlot( 'HP vs Speed', 'HP', 'Speed', 10, 'Rainbow',dt.HP, dt['Speed'])
plt.scatter(dt.Attack,dt.Defense,color="red",alpha=0.5)

plt.show()
dt.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'green')

plt.xlabel('Attack')             

plt.ylabel('Defence')

plt.title('Attack Defense Scatter Plot')      
dt.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()