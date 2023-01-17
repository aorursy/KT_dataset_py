# data EDA



import numpy as np 

import pandas as pd



# data visualization



import matplotlib.pyplot as plt

plt.style.use('ggplot') # ggplot style plot

import seaborn as sns

sns.set_palette('husl')



# Bokeh

from bokeh.io import show, output_notebook

from bokeh.palettes import Spectral9

from bokeh.plotting import figure

output_notebook() # if you want to have an output in notebook





import warnings 

warnings.filterwarnings('ignore') 
pd.set_option('display.max_columns',200) # display options

pd.set_option('display.float_format','{:.4f}'.format)
data=pd.read_csv('/kaggle/input/fifa19/data.csv')

data.head(3)
data.describe() # summary statistics for numeric features
data.describe(include='O') # categorical feature description
%%time

import requests

from PIL import Image # for image processing by Python, PIL package. Image class is used to read imange (Image.open()) 

from io import BytesIO



fig, ax = plt.subplots(10,10, figsize=(12,12))



for i in range(100):

    r = requests.get(data['Photo'][i]) # crawling file

    im = Image.open(BytesIO(r.content))

    ax[i//10][i%10].imshow(im) # Display an image

    ax[i//10][i%10].axis('off')

plt.show()
fig=plt.figure(figsize=(10,4))

fig.add_subplot(1,1,1)

sns.distplot(data['Age'])
pd.DataFrame(data['Age'].describe()).rename(columns={'Age':'Age Statistics'})
fig=plt.figure(figsize=(20,10)) 

fig.add_subplot(2,1,1)

sns.boxplot(x='Age', y='Potential', data=data)



fig.add_subplot(2,1,2)

sns.set(style="white", color_codes=True) # suitable theme for jointplot

sns.scatterplot(data=data, x='Age', y='Potential',hue="Overall", size='International Reputation')

plt.show()
fig=plt.figure(figsize=(20,10)) 

fig.add_subplot(2,1,1)

sns.boxplot(x='Age', y='Overall', data=data)



fig.add_subplot(2,1,2)

sns.set(style="white", color_codes=True) # suitable theme for jointplot

sns.scatterplot(data=data, x='Age', y='Overall',hue="Potential", size='International Reputation')

plt.show()
age_club=data.groupby('Club')['Age'].mean().sort_values(ascending=False).reset_index()

age_club_std=data.groupby('Club')['Age'].std().sort_values(ascending=False).reset_index()
print(age_club.head())

print(age_club.tail())
print(age_club_std.head())

print(age_club_std.tail())
age_club_stat=pd.merge(age_club,age_club_std, how='left', on=['Club']).rename(columns={'Age_x':'mean_Age','Age_y':'std_Age'})

age_club_stat['cv']=age_club_stat['mean_Age']/age_club_stat['std_Age']

age_club_stat
from bokeh.plotting import figure, output_file, show, ColumnDataSource



source = ColumnDataSource(data=dict(

    x=age_club_stat["mean_Age"],

    y=age_club_stat["std_Age"],

    Club=age_club_stat["Club"],

))



TOOLTIPS = [

    ("('mean','st')", "($x, $y)"),

     ("('club')","(@Club)")  ]







p = figure(title = "Age mean and std by club", tooltips=TOOLTIPS)

p.xaxis.axis_label = 'mean_Age'

p.yaxis.axis_label = 'std_Age'



p.circle('x','y', fill_alpha=0.2, size=7, source=source)



show(p)

data.head(3)
data['unit']=data['Wage'].str.slice(start=-1)

data.loc[data['unit']=='K','unit']=data.loc[data['unit']=='K','unit'].replace('K','1000')

data.loc[data['unit']=='0','unit']=data.loc[data['unit']=='0','unit'].replace('0','0')



data.loc[data['Wage'].str.slice(start=-1)=='K','Wage_new']=data['Wage'].str.slice(start=1, stop=-1)

data.loc[data['Wage'].str.slice(start=-1)=='0','Wage_new']=data['Wage'].str.slice(start=1, stop=2)

data['Wage_new']=data['Wage_new'].astype(float)*data['unit'].astype(int)



data['unit']=data['Value'].str.slice(start=-1)

data.loc[data['unit']=='M','unit']=data.loc[data['unit']=='M','unit'].replace('M','100000')

data.loc[data['unit']=='K','unit']=data.loc[data['unit']=='K','unit'].replace('K','1000')

data.loc[data['unit']=='0','unit']=data.loc[data['unit']=='0','unit'].replace('0','0')





data.loc[data['Value'].str.slice(start=-1)=='M','Value_new']=data['Value'].str.slice(start=1, stop=-1)

data.loc[data['Value'].str.slice(start=-1)=='K','Value_new']=data['Value'].str.slice(start=1, stop=-1)

data.loc[data['Value'].str.slice(start=-1)=='0','Value_new']=data['Value'].str.slice(start=1, stop=2)

data['Value_new']=data['Value_new'].astype(float)*data['unit'].astype(int)
sns.jointplot(data=data, x=np.log1p(data['Value_new']), y=np.log1p(data['Wage_new']), kind='reg', color='red')
fig=plt.figure(figsize=(20,10)) 

fig.add_subplot(3,1,1)

sns.distplot(data['Wage_new'])





fig.add_subplot(3,1,2)

sns.set(style="white", color_codes=True) # suitable theme for jointplot

sns.scatterplot(data=data, x='Age', y='Wage_new',hue="Value_new", size='International Reputation')





fig.add_subplot(3,1,3)

sns.distplot(data['Value_new'])



plt.show()
sns.jointplot(data=data, x=data['Age'], y=np.log1p(data['Wage_new']), kind='kde', color='red')