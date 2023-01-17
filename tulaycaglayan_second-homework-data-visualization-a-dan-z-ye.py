# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import matplotlib.pyplot as plt # visualizing data
import seaborn as sns 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/BlackFriday.csv')
df.head(10)
#update NaN values with 0 for Product_Category_1 , Product_Category_2 , Product_Category_3
df["Product_Category_1"].fillna(0, inplace=True)
df["Product_Category_2"].fillna(0, inplace=True)
df["Product_Category_3"].fillna(0, inplace=True)

df.info()
# this method draws plot by using dataframe's own plot method. get counts of df[column] for df[group]
def plot(group,column,plot):
    ax=plt.figure(figsize=(6,4))
    df.groupby(group)[column].sum().plot(plot)

# this method draws plot by using sns library. get counts of df[column] for df[group]
def plotUsingSns(group,column):
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.countplot(df[group],hue=df[column])
    
# this method draws piechart for counts of df[column]
def pieChartByCounts(df, column):
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sf = df[column].value_counts() #Produces Pandas Series
    explode =()
    for i in range(len(sf.index)):
        if i == 0:
            explode += (0.1,)
        else:
            explode += (0,)
    ax1.pie(sf.values, explode=explode,labels=sf.index, autopct='%1.1f%%', shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    plt.legend()
    plt.show()

# this method draws piechart for sf.values for sf.indexes
def pieChartByValues(sf, title, legentTitle):
    
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')
    cmap = plt.get_cmap("magma_r")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    fig1, ax1 = plt.subplots(figsize=(8,5))
    
    explode =()
    
    for i in range(len(sf.values)):
        if sf.index[i] == sf.idxmax():
            explode += (0.1,)
        else:
            explode += (0,)
    ax1.pie(sf.values, explode=explode,labels=sf.index, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors, radius =1)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    plt.legend(loc='upper center',prop=fontP, bbox_to_anchor=(1.2, 1),title=legentTitle)
    plt.title(title)
    plt.show()
pieChartByCounts(df, 'Gender' )
plot('Gender','Purchase','bar')
#Filter data 
df_by_occupation_and_categories = df.groupby(['Gender','Product_Category_1']).count().reset_index('Product_Category_1')

#use filtered data to draw graphs for each index 
for i in list(df_by_occupation_and_categories.index.unique()):
    sf = pd.Series (df_by_occupation_and_categories['Purchase'][i].get_values() , index = df_by_occupation_and_categories['Product_Category_1'][i].get_values())
    pieChartByValues(sf , "Gender {0}".format(i), "Product Category")
plotUsingSns('Age','Gender')
plot('Age','Purchase','bar')
pieChartByCounts(df,'Age')
pieChartByCounts(df,'City_Category')
plotUsingSns ('City_Category', 'Stay_In_Current_City_Years')
plotUsingSns ('Marital_Status', 'Product_Category_1')
#Filter data
df_by_occupation_and_categories = df.groupby(['Occupation','Product_Category_1']).count().reset_index('Product_Category_1')

# draw on filtered data for each index value of data 
for i in range (len(df_by_occupation_and_categories.index.unique())):
    sf = pd.Series (df_by_occupation_and_categories['Gender'][i].get_values() , index = df_by_occupation_and_categories['Product_Category_1'][i].get_values())
    pieChartByValues(sf , "Occupation {0}".format(i), "Product Category")