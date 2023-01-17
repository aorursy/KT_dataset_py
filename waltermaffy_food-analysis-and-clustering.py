import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py 

import plotly.graph_objs as go

import scipy.cluster.hierarchy as shc

from plotly.offline import init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation

from sklearn.preprocessing import StandardScaler

import os



df = pd.read_csv("../input/world-foodfeed-production/FAO.csv",  encoding = "ISO-8859-1")

df.head()
df.dtypes[:20]
#In order to not have problems of consistency:

df['Area'].replace(['Swaziland'], 'Eswatini', inplace=True)

df['Area'].replace(['The former Yugoslav Republic of Macedonia'], 'North Macedonia', inplace=True)



#GET NEW DATA

df_pop = pd.read_csv("../input/world-population/FAOSTAT_data_6-13-2019.csv")

df_area = pd.read_csv("../input/countries-area-2013/countries_area_2013.csv")

df_pop = pd.DataFrame({'Area': df_pop['Area'] , 'Population': df_pop['Value'] })

df_area = pd.DataFrame({'Area' : df_area['Area'], 'Surface': df_area['Value']})

#add missing line

df_area = df_area.append({'Area' : 'Sudan' , 'Surface' : 1886} , ignore_index=True)



#MERGE OF TABLES

d1 = pd.DataFrame(df.loc[:, ['Area', 'Item', 'Element']])

data = pd.merge(d1, df_pop, on='Area', how='left')

new_data = pd.merge(data, df_area, on='Area', how='left')



d2 = df.loc[:, 'Y1961':'Y2013']

data = new_data.join(d2)

data.head()
print('Number of different Countries: ' , df['Area'].unique().size)

print('Number of different Items: ' , df['Item'].unique().size)
#Graph of missing values

sns.heatmap(data.isnull(),cbar=False,cmap='viridis')   

plt.show()
# Total number of missing values per year

print('YEAR  MISSING VALUES')

print (df.loc[:, 'Y1961':'Y2013'].isnull().sum())
df1 = data[data.isna().any(axis=1)]

df1.head()
#Total number of missing values for Area

values_per_area = data.pivot_table(index=['Area'], aggfunc='size')

df1 = data[data.isna().any(axis=1)]

df_missing_area = df1.pivot_table(index=['Area'], aggfunc='size')

df_missing_area
year_list = list(df.iloc[:,10:].columns)

df_new = df.pivot_table(values=year_list,columns = 'Element', index=['Area'], aggfunc='sum') #for each country sum over years separatly Food&Feed

df_fao = df_new.T

df_fao.head()
# Finding the Top 5 producer of Feed and Food from 1961 to 2013

df_fao_tot = df_fao.sum(axis=0).sort_values(ascending=False).head()

df_fao_tot.plot(kind='bar', title='Top 5 Food & Feed producer', color='g')
#Producer of just Food

df_food = df_fao.xs('Food', level=1, axis=0)

df_food_tot = df_food.sum(axis=0).sort_values(ascending=False).head()

#Producer of just Feed

df_feed = df_fao.xs('Feed', level=1, axis=0)

df_feed = df_feed.sum(axis=0).sort_values(ascending=False).head()



#Plot

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

df_food_tot.plot(kind='bar', title='Top 5 Food producer', color='g', ax=ax1)

df_feed.plot(kind='bar', title='Top 5 Feed producer', color='g', ax=ax2 )
#Rank of most Produced Items 

df_item = df.pivot_table(values=year_list, columns='Element',index=['Item'], aggfunc='sum')

df_item = df_item.T

#FOOD

df_food_item = df_item.xs('Food', level=1, axis=0)

df_food_item = df_food_item.sum(axis=0).sort_values(ascending=False).head()

#FEED

df_feed_item = df_item.xs('Feed', level=1, axis=0)

df_feed_item = df_feed_item.sum(axis=0).sort_values(ascending=False).head()

#Plot

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

df_food_item.plot(kind='bar', title='Top 5 Food produced item', color='g', ax=ax1)

df_feed_item.plot(kind='bar', title='Top 5 Feed produced item', color='g' , ax=ax2)
# Visualization of the top 5 producer countries among years

plt.figure(figsize = (10,6))

top_5 = []

for i in df_food_tot.index:

    year = df_food[i]

    top_5.append(year)

    plt.plot(year, marker='p')

    plt.xticks(df_food.index, rotation='vertical')

    plt.legend(loc='right')
fig, ax = plt.subplots(figsize=(10,7))

sns.heatmap(data=pd.DataFrame(top_5),linewidths=0, ax=ax)

plt.show()
d3 = df.loc[:, 'Y1993':'Y2013'] #take only last 20 years

data1 = new_data.join(d3) #recap: new_data does not contains years data



d4 = data1.loc[data1['Element'] == 'Food'] #get just food

d5 = d4.drop('Element', axis=1)

d5 = d5.fillna(0) #substitute missing values with 0



year_list = list(d3.iloc[:,:].columns)

d6 = d5.pivot_table(values=year_list, index=['Area'], aggfunc='sum')



italy = d4[d4['Area'] == 'Italy']

italy = italy.pivot_table(values=year_list, index=['Item'], aggfunc='sum')

italy = pd.DataFrame(italy.to_records())



item = d5.pivot_table(values=year_list, index=['Item'], aggfunc='sum')

item = pd.DataFrame(item.to_records())



d5 = d5.pivot_table(values=year_list, index=['Area', 'Population', 'Surface'], aggfunc='sum')

area = pd.DataFrame(d5.to_records())

d6.loc[:, 'Total'] = d6.sum(axis=1)

d6 = pd.DataFrame(d6.to_records())

d = pd.DataFrame({'Area' : d6['Area'] , 'Total': d6['Total'] , 'Population': area['Population'], 'Surface': area['Surface']})
d.head()
data_ = dict(type = 'choropleth',

locations = d['Area'],

locationmode = 'country names',

z = d['Total'],

text = d['Area'],

colorbar = {'title':'Tons of food'})

layout = dict(title = 'Total Production of Food 1993-2013',

geo = dict(showframe = False,

projection = {'type': 'mercator'}))

choromap3 = go.Figure(data = [data_], layout=layout)

iplot(choromap3)
data_ = dict(type = 'choropleth',

locations = d['Area'],

locationmode = 'country names',

z = d['Population'],

text = d['Area'],

colorbar = {'title':'Tons of food'})

layout = dict(title = 'World Population of 2013',

geo = dict(showframe = False,

projection = {'type': 'mercator'}))

choromap3 = go.Figure(data = [data_], layout=layout)

iplot(choromap3)
data_ = dict(type = 'choropleth',

locations = d['Area'],

locationmode = 'country names',

z = d['Surface'],

text = d['Area'],

colorbar = {'title':'Tons of food'})

layout = dict(title = 'World Surface',

geo = dict(showframe = False,

projection = {'type': 'mercator'}))

choromap3 = go.Figure(data = [data_], layout=layout)

iplot(choromap3)
italy.head()
area.head()
item.head()
X = pd.DataFrame({'Total': d['Total'], 'Surface' : d['Surface'], 'Population' : d['Population']})

X.head()
X.describe()
fig = plt.figure(figsize=(20,26))



ax1 = fig.add_subplot(231)

ax1=sns.boxplot(x='Total',data=X, orient='v') 

ax2 = fig.add_subplot(232)

ax2=sns.boxplot(x='Surface',data=X,orient='v')

ax3 = fig.add_subplot(233)

ax3=sns.boxplot(x='Population',data=X, orient='v')
f,ax = plt.subplots(figsize=(6, 6))

sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
wcss = []

for i in range(1,8):

    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=7,random_state=0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1,8),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
def K_Means(X, n):

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    model = KMeans(n)

    model.fit(X)

    clust_labels = model.predict(X)

    cent = model.cluster_centers_

    return (clust_labels, cent)
clust_labels, cent = K_Means(X, 2)

kmeans = pd.DataFrame(clust_labels)

X.insert((X.shape[1]),'kmeans',kmeans)
def Plot3dClustering(n, X, type_c): 

    data = []

    clusters = []

    colors = ['rgb(228,26,28)','rgb(55,126,184)','rgb(77,175,74)']



    for i in range(n):

        name = i

        color = colors[i]

        x = X[ X[type_c] == i ]['Total']

        y = X[ X[type_c] == i ]['Population']

        z = X[ X[type_c] == i ]['Surface']



        trace = dict(

            name = name,

            x = x, y = y, z = z,

            type = "scatter3d",    

            mode = 'markers',

            marker = dict( size=4, color=color, line=dict(width=0) ) )

        data.append( trace )



        cluster = dict(

            color = color,

            opacity = 0.1,

            type = "mesh3d", 

            alphahull = 7,

            name = "y",

            x = x, y = y, z = z )

        data.append( cluster )



    layout = dict(

        width=800,

        height=550,

        autosize=False,

        title='3D Clustering Plot',

        scene=dict(

            xaxis=dict(

                gridcolor='rgb(255, 255, 255)',

                zerolinecolor='rgb(255, 255, 255)',

                showbackground=True,

                title='Total Production',

                backgroundcolor='rgb(230, 230,230)'

            ),

            yaxis=dict(

                gridcolor='rgb(255, 255, 255)',

                zerolinecolor='rgb(255, 255, 255)',

                showbackground=True,

                title='Population',

                backgroundcolor='rgb(230, 230,230)'

            ),

            zaxis=dict(

                gridcolor='rgb(255, 255, 255)',

                zerolinecolor='rgb(255, 255, 255)',

                showbackground=True,

                title='Surface Area',

                backgroundcolor='rgb(230, 230,230)'

            ),

            aspectratio = dict( x=1, y=1, z=0.7 ),

            aspectmode = 'manual'        

        ),

    )



    fig = dict(data=data, layout=layout)

    iplot(fig, filename='total_surface_population_plot', validate=False)

Plot3dClustering(n=2, X=X , type_c='kmeans')
cluster1 = pd.DataFrame(d[ X['kmeans'] == 1 ]['Area'])

cluster1
clust_labels, cent = K_Means(X, 3)

kmeans = pd.DataFrame(clust_labels)

del X['kmeans']

X.insert((X.shape[1]),'kmeans',kmeans)

Plot3dClustering(n=3, X=X, type_c='kmeans')
cluster2 = pd.DataFrame(d[ X['kmeans'] == 2 ]['Area'])

cluster2
new_d = d.drop(d[d.Total > 1e7].index)

new_d = new_d.drop(new_d[new_d.Surface > 5e5].index)

new_d = new_d.drop(new_d[new_d.Population > 5e5].index)

X_f = pd.DataFrame({'Total': new_d['Total'], 'Surface' : new_d['Surface'], 'Population' : new_d['Population']})
clust_labels, cent = K_Means(X_f, 2)

kmeans = pd.DataFrame(clust_labels)

X_f.insert((X_f.shape[1]),'kmeans',kmeans)
Plot3dClustering(n=2,X=X_f, type_c='kmeans')
cluster1 = pd.DataFrame(new_d[ X_f['kmeans'] == 1 ]['Area'])

cluster1.head()
cluster2 = pd.DataFrame(new_d[ X_f['kmeans'] == 0 ]['Area'])

cluster2
def Agglomerative(X, n): #number of clusters is not necessary but Python provides an option of providing the same for easy and simple use.

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    model = AgglomerativeClustering(n_clusters=n, affinity = 'euclidean', linkage = 'ward')

    clust_labels1 = model.fit_predict(X)

    return (clust_labels1)
clust_labels1 = Agglomerative(X, 2)

agglomerative = pd.DataFrame(clust_labels1)

X.insert((X.shape[1]),'agglomerative',agglomerative)

Plot3dClustering(n=3, X=X, type_c='agglomerative')
cluster0 = pd.DataFrame(d[ X['agglomerative'] == 0 ]['Area'])

cluster0
plt.figure(figsize=(25, 15))

plt.title("Customer Dendograms")

dend = shc.dendrogram(shc.linkage(X, method='ward'))