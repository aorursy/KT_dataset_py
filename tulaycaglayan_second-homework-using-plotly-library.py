# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# plotly

from plotly.offline import init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


df = pd.read_csv('../input/BlackFriday.csv')
df.head()
df.info()
#update NaN values with 0 for Product_Category_1 , Product_Category_2 , Product_Category_3

df["Product_Category_1"].fillna(0, inplace=True)
df["Product_Category_2"].fillna(0, inplace=True)
df["Product_Category_3"].fillna(0, inplace=True)
# prepare data frame 
# for each gender , sum  Purchase 
df_Gender = df.groupby(['Gender','Product_Category_1'])['Purchase'].sum().reset_index('Gender').reset_index('Product_Category_1')

# add rank column for counts 
df_Gender['Rank'] = df_Gender.groupby('Gender')['Purchase'].rank(ascending=False).astype(int)

# Filter female and male 
df_Female = df_Gender[df_Gender['Gender'] =='F'] 
df_Male = df_Gender[df_Gender['Gender'] =='M'] 

# get only 3 columns 
df_Female = df_Female[['Product_Category_1', 'Purchase','Rank']]
df_Male = df_Male[['Product_Category_1', 'Purchase','Rank']]

#Add category names
df_Female['Product_Category_Name'] = df_Female.apply(lambda row: "PrdType_" + row['Product_Category_1'].astype(str), axis=1)
df_Male['Product_Category_Name'] = df_Male.apply(lambda row: "PrdType_" + row['Product_Category_1'].astype(str), axis=1)

df_Female.sort_values(by=['Rank'], inplace = True)
df_Male.sort_values(by=['Rank'], inplace = True)

print(df_Female )

# Draw graph 
# Creating trace1
trace1 = go.Scatter(
                    x = df_Female.Rank,
                    y = df_Female.Purchase,
                    mode = "lines",
                    name = "Female Purchase",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df_Female.Product_Category_Name)
# Creating trace2
trace2 = go.Scatter(
                    x = df_Male.Rank,
                    y = df_Male.Purchase,
                    mode = "lines+markers",
                    name = "Male Purchase",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df_Male.Product_Category_Name)
data = [trace1,trace2]
layout = dict(title = 'Male and Female vs buying Product_Category_1',
              xaxis= dict(title= 'Product Purchase Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

# creating trace1
trace1 =go.Scatter(
                    x = df_Female.Rank,
                    y = df_Female.Purchase,
                    mode = "markers",
                    name = "Female Purchase",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df_Female.Product_Category_Name)
# creating trace2
trace2 =go.Scatter(
                    x = df_Male.Rank,
                    y = df_Male.Purchase,
                    mode = "markers",
                    name = "Male Purchase",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df_Male.Product_Category_Name)

data = [trace1, trace2]
layout = dict(title = 'Male and Female vs buying Product_Category_1',
              xaxis= dict(title= 'Product Purchase Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Puchase',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)  

#second bar chart 

trace1 = {
  'x': df_Female.Product_Category_1,
  'y': df_Female.Purchase,
  'name': 'Female Purchase',
  'type': 'bar'
};
trace2 = {
  'x': df_Male.Product_Category_1,
  'y': df_Male.Purchase,
  'name': 'Male Purchase',
  'type': 'bar'
};
data = [trace1, trace2];
layout = {
  'xaxis': {'title': 'Product Category 1'},
  'barmode': 'relative',
  'title': 'Male and Female vs buying Product_Category_1'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)
# creating trace1
trace1 =go.Bar(
                    x = df_Female.Rank,
                    y = df_Female.Purchase,
                    name = "Female Purchase",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)', line=dict(color='rgb(0,0,0)',width=1.1)),
                    text= df_Female.Product_Category_Name)
# creating trace2
trace2 =go.Bar(
                    x = df_Male.Rank,
                    y = df_Male.Purchase,
                    name = "Male Purchase",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)', line=dict(color='rgb(0,0,0)',width=1.1)),
                    text= df_Male.Product_Category_Name)

data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = dict(data = data, layout = layout)
iplot(fig)
#%% second bar chart 
trace1 = {
  'x': df_Female.Product_Category_1,
  'y': df_Female.Purchase,
  'name': 'Female Purchase',
  'type': 'bar'
};
trace2 = {
  'x': df_Male.Product_Category_1,
  'y': df_Male.Purchase,
  'name': 'Male Purchase',
  'type': 'bar'
};
data = [trace1, trace2];
layout = {
  'xaxis': {'title': 'Product Category 1 '},
  'barmode': 'relative',
  'title': 'Male and Female vs buying Product_Category_1'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)

# figure
fig = {
  "data": [
    {
      "values": df_Male.Purchase[df_Male.Rank <=5],
      "labels": df_Male.Product_Category_Name[df_Male.Rank <=5],
      "domain": {"x": [0, .5]},
      "name": "Total Purchase",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],

  "layout": {
        "title":"Top 5 Product category1 purchase for men",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Total Purchase",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)


#prepare data 
# filter users who does not buy, means 0 
df_Cat1 = df[df.Product_Category_1 != 0].groupby(['User_ID','Product_Category_1']).count().reset_index('User_ID').reset_index('Product_Category_1')
df_Cat2 = df[df.Product_Category_2 != 0].groupby(['User_ID','Product_Category_1']).count().reset_index('User_ID').reset_index('Product_Category_1')
# Rename column Product_ID as Count
df_Cat1.rename(columns={'Product_ID': 'Count_Prd_Cat1'}, inplace = True)
df_Cat2.rename(columns={'Product_ID': 'Count_Prd_Cat2'}, inplace = True)

# add rank column for counts 
df_Cat1['Rank_Prd_Cat1'] = df_Cat1.groupby('Product_Category_1')['Count_Prd_Cat1'].rank(ascending=False).astype(int)

#filter columns 
df_Cat1 = df_Cat1[['User_ID','Product_Category_1','Count_Prd_Cat1', 'Rank_Prd_Cat1' ]]
df_Cat2 = df_Cat2[['User_ID','Product_Category_1','Count_Prd_Cat2']]
# merge 
df_Users = pd.merge(df_Cat1, df_Cat2, on=['User_ID','Product_Category_1'])

# Add  User_Name  column 
df_Users['User_Name'] = df_Users.apply(lambda row: "Usr_" + row['User_ID'].astype(str), axis=1)
df_Users.sort_values(by=['Product_Category_1'], inplace = True)

df_Users = df_Users[['User_ID', 'User_Name', 'Product_Category_1','Count_Prd_Cat1','Count_Prd_Cat2', 'Rank_Prd_Cat1']][df_Users.Rank_Prd_Cat1 ==1]

df_Users

#%% Show first person for each ProductCategory1 and how many they bought .
# define sizes of bubbles for ProductCategory2 to show did they also buy ProductCategory2 
# draw graph 
count_size  = [ float( each) for each in df_Users.Count_Prd_Cat2]
international_color = [float( each) for each in df_Users.User_ID]
    
fig = {
  "data": [
    {
        'y': df_Users.Count_Prd_Cat1,
        'x': df_Users.Product_Category_1,
        'mode': 'markers',
        'marker': {
            'color': international_color,
            'size': count_size,
            'showscale': True
        },
        "text" :  df_Users.User_Name
    },],

  "layout": {
        "title":"Top User For Each Product Category1 and Size for Product Category2",
        "annotations": [
            { "font": { "size": 14},
              "showarrow": False,
              "text": "Product Category 1 ",
                "x": 8,
                "y": -10 
            },
        ]
    }
}
iplot(fig)
# prepare data
df_Female = df[df['Gender'] =='F']['Product_Category_1']
df_Male = df[df['Gender'] =='M']['Product_Category_1'] 
# Counts for Product_Category_1 for Female and Male
trace1 = go.Histogram(
    x=df_Male,
    opacity=0.75,
    name = "Male",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    x=df_Female,
    opacity=0.75,
    name = "Female",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',
                   title=' Product_Category_1 counts purchased by Female and Male',
                   xaxis=dict(title='Product Category 1'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
#%% World cloud 

Product_ID = df.Product_ID
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(Product_ID))
plt.imshow(wordcloud)
plt.axis('off')
#plt.savefig('graph.png')

plt.show()
# data preparation
df_Male = df[df.Gender == 'M']
df_Female = df[df.Gender == 'F']

trace0 = go.Box(
    y=df_Male.Purchase,
    name = 'Total Purchase of Males in Black Friday',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=df_Female.Purchase,
    name = 'Total Purchase of Female in Black Friday',
    marker = dict(
        color = 'rgb(12, 128, 12)',
    )
)
data = [trace0, trace1]
iplot(data)
# import figure factory
import plotly.figure_factory as ff
# prepare data
df_Male = df[df.Gender == 'M']

df_Male = df_Male.loc[:100,["Product_Category_1","Product_Category_2", "Product_Category_3"]]
df_Male["index"] = np.arange(1,len(df_Male)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(df_Male, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)
# prepare data frame 
# for each gender , count  number of Product_Category_1 types 
df_Gender = df.groupby(['Gender','Product_Category_1']).count().reset_index('Gender').reset_index('Product_Category_1')

# add rank column for counts 
df_Gender['Rank'] = df_Gender.groupby('Gender')['Product_ID'].rank(ascending=False).astype(int)

# Filter female and male 
df_Female = df_Gender[df_Gender['Gender'] =='F'] 
df_Male = df_Gender[df_Gender['Gender'] =='M'] 

# get only 3 columns 
df_Female = df_Female[['Product_Category_1', 'Product_ID','Rank']]
df_Male = df_Male[['Product_Category_1', 'Product_ID','Rank']]

# Rename column Product_ID as Count
df_Female.rename(columns={'Product_ID': 'Count'}, inplace = True)
df_Male.rename(columns={'Product_ID': 'Count'}, inplace = True)

df_Female['Product_Category_1'] = df_Female.apply(lambda row: "PrdType_" + row['Product_Category_1'].astype(str), axis=1)
df_Female.sort_values(by=['Rank'], inplace = True)
df_Male.sort_values(by=['Rank'], inplace = True)
print(df_Female )
# Draw graph 
# Creating trace1
trace1 = go.Scatter(
                    x = df_Female.Rank,
                    y = df_Female.Count,
                    mode = "markers",
                    name = "Female Buy",
                    marker = dict(color = 'rgba(256, 0, 0, 0.8)'),
                    text= df_Female.Product_Category_1)
# Creating trace2
trace2 = go.Scatter(
                    x = df_Male.Rank,
                    y = df_Male.Count,
                    xaxis='x2',
                    yaxis='y2',
                    mode = "lines+markers",
                    name = "Male Buy",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df_Male.Product_Category_1)
data = [trace1,trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2',        
    ),
    yaxis2=dict(
        domain=[0.6, 0.95],
        anchor='x2',
    ),
    title = 'Male and Female Purchase Count for Product Category 1'

)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
#prepare data frame 
# for each gender , count  number of Product_Category_1 types 
df_Gender = df.groupby(['Gender','Product_Category_1']).count().reset_index('Gender').reset_index('Product_Category_1')

# add rank column for counts 
df_Gender['Rank'] = df_Gender.groupby('Gender')['Product_ID'].rank(ascending=False).astype(int)

# Filter female and male 
df_Female = df_Gender[df_Gender['Gender'] =='F'] 
df_Male = df_Gender[df_Gender['Gender'] =='M'] 

# get only 3 columns 
df_Female = df_Female[['Product_Category_1', 'Product_ID','Rank']]
df_Male = df_Male[['Product_Category_1', 'Product_ID','Rank']]

# Rename column Product_ID as Count
df_Female.rename(columns={'Product_ID': 'Count'}, inplace = True)
df_Male.rename(columns={'Product_ID': 'Count'}, inplace = True)

df_Female['Product_Category_1'] = df_Female.apply(lambda row: "PrdType_" + row['Product_Category_1'].astype(str), axis=1)
df_Female.sort_values(by=['Rank'], inplace = True)
df_Male.sort_values(by=['Rank'], inplace = True)

# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=df_Female.Count,
    y=df_Male.Count,
    z=df_Female.Rank,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(255,0,0)',                # set color to an array/list of desired values      
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
#%%
trace1 = go.Scatter(
    x=df_Female.Rank,
    y=df_Female.Count,
    name = "Female count"
)
trace2 = go.Scatter(
    x=df_Male.Rank,
    y=df_Male.Count,
    xaxis='x2',
    yaxis='y2',
    name = "Male count"
)

data = [trace1, trace2]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        domain=[0.55, 1]
    ),   
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    title = 'Product_Category_1 counts purchased by Female and Male'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)