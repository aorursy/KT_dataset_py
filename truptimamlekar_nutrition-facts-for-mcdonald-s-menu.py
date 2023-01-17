import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

%matplotlib inline
df = pd.read_csv('../input/nutrition-facts/menu.csv')

df.head(1)
df.isnull().any()
df.shape 
df.info()
df.describe()
df.isnull().sum()
sns.set(font_scale=2)

plt.figure(figsize=(15, 10))

sns.countplot(y='Category', data=df)
df.hist (bins=10,figsize=(35,25))

plt.show ()
df.plot(kind='hist',figsize=(20,8))
sns.set(font_scale=1.5)

plt.figure(figsize=(22,8))

corr = (df.corr())

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values,cmap="YlGnBu",annot=True,linewidths=.5, fmt=".2f")

plt.title("Pearson Correlation of all Nutrient Elements")
df.corr()
f,ax=plt.subplots(2,2,figsize=(25,15))

v1= sns.violinplot(x="Category", y="Calories",data=df, palette="muted",ax=ax[0][0])

ax[0][0].set_xticklabels(ax[0][0].get_xticklabels(), rotation=45,horizontalalignment='right')

v1= sns.violinplot(x="Category", y="Carbohydrates",data=df, palette="muted",ax=ax[0][1])

ax[0][1].set_xticklabels(ax[0][1].get_xticklabels(), rotation=45,horizontalalignment='right')

v1= sns.violinplot(x="Category", y="Dietary Fiber",data=df, palette="muted",ax=ax[1][0])

ax[1][0].set_xticklabels(ax[1][0].get_xticklabels(), rotation=45,horizontalalignment='right')

v1= sns.violinplot(x="Category", y="Protein",data=df, palette="muted",ax=ax[1][1])

ax[1][1].set_xticklabels(ax[1][1].get_xticklabels(), rotation=45,horizontalalignment='right')
f,axes=plt.subplots (1,1,figsize=(15,4))

sns.distplot(df['Calories'],kde=True,hist=True,color="g")
f,axes=plt.subplots (1,1,figsize=(15,4))

df['Category'].replace([0], 'Beef & Pork', inplace=True) 

df['Category'].replace([1], 'Beverages', inplace=True) 

df['Category'].replace([2], 'Breakfast', inplace=True) 

sns.kdeplot(df.loc[(df['Category']=='Breakfast'), 'Calories'], color='b', shade=True, Label='Breakfast')

sns.kdeplot(df.loc[(df['Category']=='Coffee & Tea'), 'Calories'], color='g', shade=True, Label='Coffee & Tea')

sns.kdeplot(df.loc[(df['Category']=='Smoothies & Shakes'), 'Calories'], color='r', shade=True, Label='Smoothies & Shakes')

plt.xlabel('Calories') 

plt.ylabel('Probability Density') 
f,axes = plt.subplots(1,1,figsize=(10,5),sharex = True,sharey =True)

s=np.linspace(0,3,10)

cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)

x = df['Cholesterol (% Daily Value)'].values

y = df['Sodium (% Daily Value)'].values

sns.kdeplot(x,y,cmap=cmap,shade=True,cut = 5)
f,ax=plt.subplots(1,3,figsize=(25,5))

box1=sns.boxplot(data=df["Dietary Fiber (% Daily Value)"],ax=ax[0])

ax[0].set_xlabel('Dietary Fiber (% Daily Value)')

box1=sns.boxplot(data=df["Calcium (% Daily Value)"],ax=ax[1])

ax[1].set_xlabel('Calcium (% Daily Value)')

box1=sns.boxplot(data=df["Iron (% Daily Value)"],ax=ax[2])

ax[2].set_xlabel('Iron (% Daily Value)')
ax = sns.scatterplot(x="Carbohydrates (% Daily Value)", y="Category",color = "orange",data=df)

ax = sns.scatterplot(x="Calcium (% Daily Value)", y="Category",color = "blue",data=df)

ax = sns.scatterplot(x="Iron (% Daily Value)", y="Category",color = "green",data=df)
df.head(1)

df1=df.drop(['Sodium','Sodium (% Daily Value)','Carbohydrates','Carbohydrates (% Daily Value)','Dietary Fiber','Dietary Fiber (% Daily Value)','Iron (% Daily Value)','Calcium (% Daily Value)','Vitamin C (% Daily Value)','Vitamin A (% Daily Value)','Protein','Sugars'],axis=1)
sns.set(style="ticks", color_codes=True)

g = sns.pairplot(df1)
# Prepare Data

df1= df.groupby('Category').size()

# Make the plot with pandas

df1.plot(kind='pie', subplots=True, figsize=(8, 8))

plt.title("Pie Chart of Various Category")

plt.ylabel("")

plt.show()
from wordcloud import WordCloud 
df1=df['Item'].to_string()
# Start with one review:

text = df1

# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)

# Display the generated image:

f,ax=plt.subplots(1,1,figsize=(25,5))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()