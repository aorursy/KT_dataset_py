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

import seaborn as sns

import pylab as pl



mcd = pd.read_csv('../input/nutrition-facts/menu.csv')

#SEE IF NaN VALUES EXISTS

print(mcd.isna().any())



print(mcd.info())

print(mcd.head())

print(mcd.dtypes)

print(mcd.describe())







#FILTERS DATA

mcd[(mcd['Calories from Fat']>200) & (mcd['Calories']>100)]



 



#CORRELATION OF DATA. HEAT MAP INCLUDED

print(mcd.corr())

fig, ax = plt.subplots(figsize=(16, 16))

sns.heatmap(mcd.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()









#CATEGORY BREAKDOWN

print(mcd['Category'].value_counts())



#PIE CHART OF CATEGORIES

labels = 'Coffee & Tea', 'Breakfast', 'Smoothies & Shakes', 'Chicken & Fish', 'Beverages','Beef & Pork', 'Snacks & Sides', 'Desserts', 'Salads'

sizes = [95,42,28,27,27,15,13,7,6]

explode = (0.1, 0, 0,0,0,0,0,0,0 )



fig1 , ax1 = plt.subplots()



ax1.pie(sizes,

        explode = explode,

        labels = labels,

        autopct = '%1.1f%%',

        shadow = True,

        startangle = 100)

ax1.axis ('equal')

plt.show()



#SIDE BAR GRAPH OF CATEGORIES

sns.set(font_scale=2)

plt.figure(figsize=(15, 10))

sns.countplot(y='Category', data=mcd)











#SIDE BAR GRAPH OF TOTAL FAT FOR GRILLED VS CRISPY

mcd['isGrilled']=mcd.Item.str.contains("Grilled")

mcd['hasEggWhites']=mcd.Item.str.contains("Egg Whites")



crispy_mcd1=mcd.loc[mcd.isGrilled==True,'Item'].str.replace('Grilled','Crispy')

crispy_mcd=mcd.loc[mcd.Item.isin(crispy_mcd1),['Item','Total Fat (% Daily Value)','Calories']]

grilled_mcd=mcd.loc[mcd.isGrilled==True,['Item','Total Fat (% Daily Value)','Calories']]



mcd1=grilled_mcd.reset_index(drop=True).merge(crispy_mcd.reset_index(drop=True),left_index=True,right_index=True)

mcd1.columns=['Items-Grilled','TotalFat-Grilled','Calories-Grilled','Items-Crispy','TotalFat-Crispy','Calories-Crispy']

mcd1=mcd1.drop('Items-Crispy',axis=1)

mcd1['Item']=mcd1['Items-Grilled'].str.replace("Grilled","")

mcd1=mcd1.drop('Items-Grilled',axis=1)

mcd1.index=mcd1.Item

from pylab import rcParams

rcParams['figure.figsize'] = 8, 10

mcd1.loc[:,['TotalFat-Grilled','TotalFat-Crispy','Item']].plot(kind='barh',title="Fat grilled versus crispy")



#SIDE BAR GRAPH OF CALORIES FOR GRILLED VS CIRSPY

mcd1.loc[:,['Calories-Grilled','Calories-Crispy','Item']].plot(kind='barh',title="Fat grilled versus crispy")













#CALROIES BREAKDOWN

print(mcd.Calories.mean())

print(mcd.Calories.median())



#CALROIES DISTRIBUTION GRAPH

axes=plt.subplots (1,1,figsize=(15,4))

sns.distplot(mcd['Calories'],kde=True,hist=True,color="b")



#BOX PLOT OF CALROIES

sns.set(style="whitegrid")

ax = sns.boxplot(x=mcd["Calories"])











#SUGAR BREAKDOWN

#THE HIGHEST SUGARS

max_sug=max(mcd['Sugars'])



print(mcd[(mcd.Sugars ==max_sug)])



plt.figure(figsize=(12,5))

plt.title("Distribution Sugars")

ax = sns.distplot(mcd["Sugars"], color = 'c')





print(mcd.Sugars.mean())

print(mcd.Sugars.median())



#SIDE BAR GRAPH OF ITEM SUGARS

def plot(grouped):

    item = grouped["Item"].sum()

    item_list = item.sort_index()

    item_list = item_list[-20:]

    plt.figure(figsize=(9,10))

    graph = sns.barplot(item_list.index,item_list.values)

    labels = [aj.get_text()[-40:] for aj in graph.get_yticklabels()]

    graph.set_yticklabels(labels)

sug = mcd.groupby(mcd["Sugars"])

plot(sug)



#SCATTER PLOT OF CHOLESTEROL VS SUGARS

mcd.plot(kind='scatter',y='Cholesterol',x='Sugars',alpha=0.4,color='m')

plt.ylabel('Cholesterol')

plt.xlabel('Sugars')

plt.title('More sugar leads to higher cholesterol?')















#PRINTS A SWARM PLOT FOR EACH MEASURE ELEMENT PER CATEGORY

measures = ['Calories', 'Total Fat', 'Cholesterol','Sodium', 'Sugars', 'Carbohydrates']

for x in measures:

     dot_measure= sns.factorplot(x="Category", y=x,data=mcd, kind="swarm",size=5, aspect=2.5);

















#VITAMIN BREAKDOWN

#SIDE BAR GRAPH OF VITAMIN C 

vc = mcd.groupby(mcd["Vitamin C (% Daily Value)"])

plot(vc)



#SIDE BAR GRAPH OF VITAMIN A 

va = mcd.groupby(mcd["Vitamin A (% Daily Value)"])

plot(va)











#SIDE BAR GRAPH OF PROTEINS

protein = mcd.groupby(mcd["Protein"])

plot(protein)





#SIDE BAR GRAPH OF CHOLESTEROL

chol = mcd.groupby(mcd["Cholesterol"])

plot(chol)

#THE HIGHEST CHOLESTEROLS

max_chol = max(mcd['Cholesterol'])

print(mcd[(mcd.Cholesterol ==max_chol)])





#FAT BREAKDOWN

fat = mcd.groupby(mcd["Trans Fat"])

plot(fat)





#SIDE BAR GRAPH OF CALORIES

calories = mcd.groupby(mcd["Calories"])

plot(calories)



#CALORIES THRESHOLD

threshold=sum(mcd['Calories'])/len(mcd['Calories'])

print('threshold is', threshold)



mcd["cal_level"]=["high" if i>threshold else "low" for i in mcd['Calories']]

print(mcd.loc[:15,["cal_level","Calories","Item"]])