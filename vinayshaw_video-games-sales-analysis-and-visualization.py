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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('darkgrid')
data=pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
data.head()
data.shape
data.info()
print("Number of games: ", len(data))

publishers = data['Publisher'].unique()

print("Number of publishers: ", len(publishers))

platforms = data['Platform'].unique()

print("Number of platforms: ", len(platforms))

genres =data['Genre'].unique()

print("Number of genres: ", len(genres))
data.isnull().sum()
data=data.dropna()
plt.figure(figsize=(15, 10))

ax=sns.countplot(x="Genre", data=data, order = data['Genre'].value_counts().index)

plt.xticks(rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',

                    color= 'black')

plt.show()
import plotly.express as px

import plotly.graph_objs as go

import plotly.offline as pyoff

import plotly.graph_objects as go



uniq=data["Genre"].unique()



total_Genre = []

for i in uniq:

    total_Genre.append(len(data[data["Genre"]==i]))

import plotly.graph_objects as go

from plotly.subplots import make_subplots



labels = uniq



# Create subplots: use 'domain' type for Pie subplot



#fig.add_trace(go.Pie(labels=labels, values=total_Genre, name="Genre"))



fig = go.Figure(data=[go.Pie(labels=labels, values=total_Genre, hole=.3)])

fig.update_layout(

    title_text="Total Game Count in Genre",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='Genre', x=0.5, y=0.5, font_size=20, showarrow=False)])

fig.show()

plt.figure(figsize=(15,10))

ax=sns.countplot(data.Year)

plt.xticks(rotation=45)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x()+0.05, p.get_height()+1), va='bottom',

                    color= 'black')

plt.show()
y = data.groupby(['Year'])['Global_Sales'].sum()

x = y.index.astype(int)

plt.figure(figsize=(12,8))

ax=sns.barplot(y = y, x = x)

plt.xlabel(xlabel='Year', fontsize=16)

plt.xticks(fontsize=12, rotation=50)

plt.ylabel(ylabel='Millions $', fontsize=16)

plt.title(label='Game Sales in Millions $ Per Year', fontsize=20)

plt.show()
plt.figure(figsize=(15,10))

plt.xticks(rotation=45)

sns.countplot( x="Platform", data=data, order = data['Platform'].value_counts().index)

plt.show()
plt.figure(figsize=(30, 10))

sns.countplot(x="Year", data=data, hue='Genre', order=data.Year.value_counts().iloc[:5].index)

plt.xticks(size=16)

plt.xlabel(xlabel='Year', fontsize=20)

plt.ylabel(ylabel='Count', fontsize=20)

plt.show()
data_genre = data.groupby(by=['Genre'])['Global_Sales'].sum().reset_index().sort_values(by=['Global_Sales'], ascending=False)
plt.figure(figsize=(15, 10))

ax=sns.barplot(x="Genre", y="Global_Sales", data=data_genre)

plt.xticks(rotation=90)

for p in ax.patches:

    ax.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',

                    color= 'black')

plt.show()
dataYear=data["Year"].dropna().unique()

dataYear.sort()
na=[]

eu=[]

jp=[]

other=[]

glbl=[]



for i in dataYear:

    x=data[data["Year"]==i]

    

    na.append(sum(x["NA_Sales"]))

    eu.append(sum(x["EU_Sales"]))

    jp.append(sum(x["JP_Sales"]))

    other.append(sum(x["Other_Sales"]))

    glbl.append(sum(x["Global_Sales"]))

 

yearSales=pd.DataFrame({"Year":dataYear, "NA":na, "EU":eu, "JP":jp, "Other":other, "Global":glbl})

yearSales["Year"].astype("int64")



yearSales.info()
yearSales["Year"]=yearSales.Year.astype("int64")



f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x="Year",y="NA" ,data=yearSales,color='lime',alpha=0.8)

sns.pointplot(x="Year",y="EU",data=yearSales,color='gold',alpha=0.8)

sns.pointplot(x="Year",y="JP",data=yearSales,color='purple',alpha=0.8)

sns.pointplot(x="Year",y="Global",data=yearSales,color='red',alpha=0.8)

plt.text(1,670,'Annual Global Sales',color='red',fontsize = 18,style = 'italic')

plt.text(1,640,'Annual NA Sales',color='lime',fontsize = 18,style = 'italic')

plt.text(1,610,'Annual EU Sales',color='gold',fontsize = 18,style = 'italic')

plt.text(1,580,'Annual Japan Sales',color='purple',fontsize = 18,style = 'italic')

plt.xticks(rotation=90)

plt.xlabel('Years',fontsize = 20,color='blue')

plt.ylabel('Sales',fontsize = 20,color='blue')

plt.title('ANNUAL SALES',fontsize = 20,color='blue')

plt.grid()
data_platform = data.groupby(by=['Platform'])['Global_Sales'].sum().reset_index().sort_values(by=['Global_Sales'], ascending=False)
plt.figure(figsize=(15, 10))

sns.barplot(x="Platform", y="Global_Sales", data=data_platform)

plt.xticks(rotation=90)

plt.show()
top_publisher = data.groupby(by=['Publisher'])['Year'].count().sort_values(ascending=False).head(20)

top_publisher = pd.DataFrame(top_publisher).reset_index()
plt.figure(figsize=(15, 10))

ax=sns.countplot(x="Publisher", data=data, order = data.groupby(by=['Publisher'])['Year'].count().sort_values(ascending=False).iloc[:20].index)

plt.xticks(rotation=90)

for p in ax.patches:

    ax.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',

                    color= 'black')