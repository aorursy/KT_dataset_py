# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings #disable warnings

warnings.filterwarnings('ignore') 



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
vgsales = pd.read_csv("../input/vgsales.csv")
vgsales.head()
# ** Show the counts of observations in each categorical bin using bars.

sns.countplot(vgsales.Genre)

plt.xticks(rotation=45)

plt.show()
#mean sales for each genre

genreList = list(vgsales.Genre.unique())      # create a list using unique values of Genre

eusales = []       # empty lists

nasales = []

jpsales = []

othersales = []

globalsales = []

for i in genreList:

    x = vgsales[vgsales['Genre']==i]      # find means of sales for each genre

    eusales.append(sum(x.EU_Sales)/len(x))

    jpsales.append(sum(x.JP_Sales)/len(x))

    nasales.append(sum(x.NA_Sales)/len(x))

    othersales.append(sum(x.Other_Sales)/len(x))

    globalsales.append(sum(x.Global_Sales)/len(x))

dataframe1 = pd.DataFrame({'genreList': genreList,'eusales':eusales,'nasales':nasales,'jpsales':jpsales,'othersales':othersales,'globalsales':globalsales}) # create dictionary, turn it to dataframe

new_index = (dataframe1['globalsales'].sort_values(ascending=False)).index.values # sort values ,ascending=False : sort by decreasing

final1 = dataframe1.reindex(new_index) # put sorted index in new_index
# ** Show point estimates and confidence intervals as rectangular bars.

plt.figure(figsize=(15,10))

sns.barplot(x=final1['genreList'], y=final1['globalsales'])

plt.xticks(rotation= 0)

plt.xlabel('Genres')

plt.ylabel('Global Sales')
# horizontal bar plot

plt.figure(figsize=(15,10))

sns.barplot(x="eusales",y="genreList",data=final1,label="EU Sales",color="cyan",alpha=0.6)

sns.barplot(x="jpsales",y="genreList",data=final1,label="JP Sales",color="gold",alpha=0.79)

sns.barplot(x="nasales",y="genreList",data=final1,label="NA Sales",color="pink",alpha=0.5)

sns.barplot(x="othersales",y="genreList",data=final1,label="Other Sales",color="green",alpha=0.7)

sns.barplot(x="globalsales",y="genreList",data=final1,label="Global Sales",color="lightgrey",alpha=0.4)

plt.xlabel("Sales")

plt.ylabel("Genres")

plt.legend()

plt.show()
#point plot

plt.figure(figsize=(15,10)) #create a (15,10) frame for plot 

sns.pointplot(x="genreList",y="eusales",color='magenta',alpha=0.6,data=final1) #alpha=opacity, data= dataframe that i use (final1)

sns.pointplot(x="genreList",y="nasales",color='navy',alpha=0.6,data=final1)

plt.grid()

plt.xlabel("Genres")

plt.ylabel("Values")

plt.title("Europe Sales - North America Sales")

plt.text(3,0.4,'EU Sales',color='magenta',fontsize = 17,style = 'normal') # 3,0.4 is where text put in

plt.text(3,0.38,'NA Sales',color='navy',fontsize = 17,style = 'normal')

plt.xticks(rotation=0) #rotation of column's uniques 

plt.show()
# ** Draw a plot of two variables with bivariate and univariate graphs.

sns.jointplot(final1.eusales, final1.nasales,kind="kde", height=7)

plt.show()
# ** Plot data and regression model fits across a FacetGrid.

sns.lmplot(x="eusales",y="nasales",data=final1)

plt.show()  

# the line on middle shows optimum value
# ** Fit and plot a univariate or bivariate kernel density estimate.

sns.kdeplot(final1.jpsales,final1.nasales,shade=True,cut=3)

plt.show()
# ** Draw a combination of boxplot and kernel density estimate.

plt.figure(figsize=(15,10))

pal = sns.cubehelix_palette(9, rot=-0.99, dark=0.4)  # sets colors, can find different palettes in net

sns.violinplot(data=final1,palette=pal,inner="points") # "innner=points" shows inner points, =>can change to box, quartile, stick, None

plt.show()
final1.corr() #correlation
# ** Plot rectangular data as a color-encoded matrix.

plt.figure(figsize=(15,10))

sns.heatmap(final1.corr(),annot=True,linewidths=.5,fmt=".3f") #annot=True > shows the numbers that are in middle of the rectangles

plt.show()  # linewidths=.5 > thickness of lines between rectangles, fmt=".1f" > shows one digit after point (.3f shows 3 digits) 
# ** Draw a box plot to show distributions with respect to categories.

plt.figure(figsize=(15,10))

sns.boxplot(x=vgsales["Year"],y=vgsales["Genre"],palette="PuBuGn_r")

plt.show()
# ** Draw a categorical scatterplot with non-overlapping points.

plt.figure(figsize=(15,10))

sns.swarmplot(x=vgsales["Year"],y=vgsales["Platform"],palette="PuBuGn_r")

plt.show()
# ** Plot pairwise relationships in a dataset.

sns.pairplot(final1)

plt.show()