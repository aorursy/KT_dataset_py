import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import normalize
from pandas import DataFrame
import warnings
warnings.filterwarnings('ignore')
% matplotlib inline
df=pd.read_csv('../input/unit_labour_costs_and_labour_productivity_(employment_based).csv')
#Using the string method contains we create a DF for the quarterly ULCs in Spain
Id1=df['Country'].str.contains('Spain')
Id2=df['Subject'].str.contains('Unit Labour Costs')
Id3=df['Measure'].str.contains('Index')
SpainCLU=df[Id1&Id2&Id3]
# We won't need these columns, so we create a list of columns to drop
drop_col=['LOCATION','Country', 'SUBJECT', 'Subject', 'MEASURE', 'Measure',
       'FREQUENCY', 'Frequency', 'TIME', 'Unit Code', 'Unit',
       'PowerCode Code', 'PowerCode', 'Reference Period Code',
       'Reference Period', 'Flag Codes', 'Flags']
# We drop the columns
SpainCLU=SpainCLU.drop(drop_col, axis=1)
# We set the column time as our index
SpainCLU.set_index('Time',inplace=True)
# We drop our duplicated data
SpainCLU=SpainCLU[0:28]
# Let's plot the indexed quarterly evolution of Spain from 2010 to 2016

plt.rc('xtick', labelsize=24)                                           
plt.rc('ytick', labelsize=24)
ax=SpainCLU.plot(figsize=(20,10), kind='line', legend=False, use_index=True, grid=True, 
                 color='aqua')
plt.axhline(y=100)                          # we create a line for index ref. 2010=100
plt.xlabel('Quarterly evolution 2010-2016', size=22)                # x title label 
plt.ylabel('Unit Labour Costs - index ref. 2010', size=20)          # y title label 
plt.title('SPAIN Unit Labour Costs ULC 2010-2016',size=26)          # plot title label   

# We create an arrow to indicate the timing of the competivity gain
bbox_props = dict(boxstyle="RArrow,pad=1", fc="cyan", ec="b", lw=2)
t = ax.text(10, 99, "Competivity gain", ha="center", va="center", rotation=340,
            size=16, bbox=bbox_props)
# We follow the same steps than the spanish case
Id4=df['Country'].str.contains('Germany')
Id5=df['Subject'].str.contains('Unit Labour Costs')
Id6=df['Measure'].str.contains('Index')
GermanyCLU=df[Id4&Id5&Id6]
GermanyCLU.drop(drop_col, axis=1, inplace=True)
GermanyCLU.set_index('Time',inplace=True)
GermanyCLU=GermanyCLU[0:28]
ax1=GermanyCLU.plot(figsize=(20,10), kind='line', legend=False, 
                    use_index=True, grid=True, color='aqua')
plt.axhline(y=100)                                                    
plt.xlabel('Quarterly evolution since 2010', size=22)                  
plt.ylabel('Unit Labour Costs - index ref. 2010', size=20)              
plt.title('GERMANY Unit Labour Costs - ULC 2010-2016',size=26)                                                      
 
bbox_props = dict(boxstyle="RArrow,pad=1", fc="cyan",ec="b", lw=2)

t = ax1.text(13, 118, "Higher labour costs", ha="center", va="center", rotation=380, 
           size=16, bbox=bbox_props)
df1=pd.read_csv('../input/SNA_TABLE1_12082017110055171.csv')
# We use the same filtering method as before with the string function contains
m1=df1['Country'].str.contains('Germany')
m2=df1['Transaction'].str.contains('Exports of goods and services')
m3=df1['Measure'].str.contains('Current prices')
m4=df1['Unit'].str.contains('Euro')
m5=df1['TRANSACT'].str.contains('P6')
GermanyExports=df1[m1&m2&m3&m4&m5]
dcg=['LOCATION', 'Country', 'TRANSACT', 'Transaction', 'MEASURE', 'Measure',
       'TIME', 'Unit Code', 'Unit', 'PowerCode Code', 'PowerCode',
       'Reference Period Code', 'Reference Period', 'Flag Codes',
       'Flags']
GermanyExports=GermanyExports.drop(dcg, axis=1)
GermanyExports.set_index('Year', inplace=True)
#Let's compute the mean of our time-series
GermanyExports.mean()
GermanyExports.plot(figsize=(20,10), kind='line', legend=False, use_index=True, 
                    grid=True, color='aqua')

# we create a line for the 0% (Y=0
plt.axhline(y=1.292924e+06, label='Exports mean 2010-2016')           
plt.xlabel('Years', size=26)                                          
plt.ylabel('Value in current prices - €', size=26)                     
plt.title('German exports 2010-1016',size=30)                                                     
plt.legend(loc='upper left', prop={'size': 20}) 
#We can create a Series in our DataFrame with the Exports growth rate since 2010
GermanyExports['pct_change']=(GermanyExports['Value'].pct_change()*100) 
#We can also compute the mean of the series
GermanyExports['pct_change'].mean()
GermanyExports['pct_change'].plot(figsize=(20,10), kind='line', legend=False, 
                                  use_index=True, grid=True, color='g')

plt.axhline(y=4.827069653839518, label='Exports growth mean')
plt.xlabel('Years', size=22)                                          
plt.ylabel('Growth in %', size=22)                                     
plt.ylim((0,12))
plt.title('Germany exports growth 2010-2016',size=30)                                              
plt.legend(loc='upper right', prop={'size': 20}) 
pdCLU=pd.read_csv('../input/PDBI_I4_19082017122907545.csv')
pdCLU.columns
# We create a list of columns which we won't need
dropCLU=['LOCATION', 'SUBJECT', 'Subject', 'MEASURE', 'Measure',
       'ACTIVITY', 'Activity', 'TIME', 'Unit Code', 'Unit',
       'PowerCode Code', 'PowerCode', 'Reference Period Code',
       'Reference Period', 'Flag Codes', 'Flags']
# We drop the list of superfluous columns
pdCLU.drop(dropCLU, axis=1, inplace=True)
# Now we can pivot the DataFrame  
pdCLU=pdCLU.pivot(index='Time', columns='Country', values='Value').transpose()
# As we have many missing values we will fill them with the closest available value
pdCLU=pdCLU.fillna(method='bfill', axis=1)
pdCLU=pdCLU.fillna(method='ffill', axis=1)

# Let's drop the EU indicators as they don' represent any country
dropIndex=['Euro area (19 countries)','European Union (28 countries)']
pdCLU.drop(dropIndex, axis=0, inplace=True)
plt.rc('xtick', labelsize=18)                                           
plt.rc('ytick', labelsize=18) 

pdCLU.plot(figsize=(20,10), kind='box', legend=False, use_index=True, grid=True, 
           color='coral')

plt.axhline(y=100, label='Index 2010 = 100')                             
plt.xlabel('YEARS', size=22)                                               
plt.ylabel('ULC index ref. base 2010', size=22)                           
plt.title('Unit labour costs (ULC) OECD distribution',size=26)            
plt.legend(loc='upper left', prop={'size': 14}) 
pdCLU.mean().plot(figsize=(20,10), kind='bar', legend=False, use_index=True, 
                  grid=True, color='coral')

SIZE = 20
plt.rc('xtick', labelsize=SIZE)                                          
plt.rc('ytick', labelsize=SIZE)                                          
plt.axhline(y=100, label='Index 2010 = 100')                            
plt.xlabel('Years',size=20) 
plt.ylabel('ULC index ref. base 2010', size=22)                          
plt.title('Unit labour costs (ULC) OECD mean 1995-2016',size=26)          
# from sklearn we import PCA module and we fit our DataSet, we specify the number of PC and call the fit() method 

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(pdCLU)
# Now we can transform our DataSet
pd_2d = pca.transform(pdCLU)
# We generate a new Dataframe with our two components as variables
df_2d = pd.DataFrame(pd_2d)
df_2d.index = pdCLU.index
df_2d.columns = ['PC1','PC2']
# We can create a DataFrame with the eigenvalues Explained Variance
DataFrame(pca.explained_variance_.round(2), index=['PC'+str(i) for i in range(1,3)], 
          columns=['Explained Variance']).T
# We do the same step for the explained variance ratio
DataFrame(pca.explained_variance_ratio_.round(2), index=['PC'+str(i) for i in range(1,3)], 
          columns=['Explained Variance Ratio']).T
#We plot our new DataFrame. We also add an annotation tag for every OECD country
ax = df_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(10,10), fontsize=18, 
                c='aqua', marker='o')

plt.xlabel('PC2', size=16)                             
plt.ylabel('PC1', size=16)                             
plt.title('PC components',size=22)                                                     
for i, Country in enumerate(df_2d.index):
    ax.annotate(Country, (df_2d.iloc[i].PC2, df_2d.iloc[i].PC1), size=14, color='m')
# We create a colum with the scaled mean for each country
df_2d['country_mean'] = pdCLU.mean(axis=1)
country_mean_max = df_2d['country_mean'].max()
country_mean_min = df_2d['country_mean'].min()
country_mean_scaled = (df_2d.country_mean - country_mean_min) / country_mean_max
df_2d['country_mean_scaled'] = country_mean_scaled
# We plot our bubble chart with the scaled mean
ax1=df_2d.plot(kind='scatter', x='PC2', y='PC1', s=df_2d['country_mean_scaled']*10000, 
               figsize=(10,10), fontsize=18, color='coral')

plt.xlabel('PC2', size=16)                             
plt.ylabel('PC1', size=16)                             
plt.title('Bubble chart: ULCs Scaled Mean 1995-2006',size=22) 

bbox_props = dict(boxstyle="RArrow,pad=1", fc="coral",ec="r", lw=2)
t = ax1.text(-25, -55, "Members with high scaled ULCs Mean ", ha="center", va="center", 
             rotation=390,size=12, bbox=bbox_props)
# This time we don't annotate the countries names
pdCLU.columns=pdCLU.columns.astype(str)
# We follow the same steps as before
df_2d['country_change'] = pdCLU['2016']-pdCLU['1995']
country_change_max = df_2d['country_change'].max()
country_change_min = df_2d['country_change'].min()
df_2d['country_change_scaled'] = (df_2d.country_change - country_change_min) / country_change_max
ax1=df_2d.plot(kind='scatter', x='PC2', y='PC1', s=df_2d['country_change_scaled']*1500, 
               figsize=(10,10),fontsize=18, color='lightcoral')

plt.xlabel('PC2', size=16)                             
plt.ylabel('PC1', size=16)                             
plt.title('Bubble chart: ULCs Change 1995-2006',size=22)  
bbox_props = dict(boxstyle="DArrow,pad=1", fc="cyan",ec="b", lw=2)
t = ax1.text(4, 60, "Members with more significant ULCs change", ha="center", va="center", 
             rotation=350, size=12, bbox=bbox_props)
from sklearn.cluster import KMeans  
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit(pdCLU)
df_2d['cluster'] = pd.Series(clusters.labels_, index=df_2d.index)
ax2=df_2d.plot(kind='scatter',x='PC2',y='PC1', c=df_2d.cluster.astype(np.float), 
               figsize=(12,12), s=df_2d['country_mean_scaled']*10000)
ax2.set_facecolor('lightcyan')
plt.xlabel('PC2', size=16)                             
plt.ylabel('PC1', size=16)                             
plt.title('Binomial clustering of OECD members by ULCs mean 1995-2006 scaled',size=22) 
for i, Country in enumerate(df_2d.index):
    ax2.annotate(Country, (df_2d.iloc[i].PC2, df_2d.iloc[i].PC1), size=14, color='r') 
    bbox_props = dict(boxstyle="round4,pad=1", fc="cyan",ec="c", lw=6)
t = ax2.text(0, 60, "CLUSTER 1", ha="center", va="center", rotation=360,
            size=16, bbox=bbox_props)
t = ax2.text(-10, -45, "CLUSTER 0", ha="center", va="center", rotation=360,
            size=16, bbox=bbox_props)
ax3=df_2d.plot(kind='scatter',x='PC2',y='PC1', c=df_2d.cluster.astype(np.float), 
               figsize=(12,12), s=df_2d['country_change_scaled']*1500)
ax3.set_facecolor('lightcyan')
plt.xlabel('PC2', size=16)                             
plt.ylabel('PC1', size=16)                             
plt.title('Binomial clustering of OECD members by ULCs change 1995-2006',size=22) 
for i, Country in enumerate(df_2d.index):
    ax3.annotate(Country, (df_2d.iloc[i].PC2, df_2d.iloc[i].PC1), size=14, color='r')
    
    bbox_props = dict(boxstyle="round4,pad=1", fc="cyan",ec="c", lw=6)
t = ax3.text(0, 60, "CLUSTER 1", ha="center", va="center", rotation=360,
            size=16, bbox=bbox_props)
t = ax3.text(-10, -45, "CLUSTER 0", ha="center", va="center", rotation=360,
            size=16, bbox=bbox_props)
from sklearn.preprocessing import normalize
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_classification, make_regression
from sklearn.externals import six
import argparse
import json
import re
import os
import sys
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()
pca = PCA(n_components=3)
pca.fit(pdCLU)
# Now we can transform our DataSet
pdCLU_3d = pca.transform(pdCLU)
# We generate a new Dataframe with our three components as variables
pdCLU_3d = pd.DataFrame(pdCLU_3d)
pdCLU_3d.index = pdCLU.index
pdCLU_3d.columns = ['PC1','PC2','PC3']
clusters=kmeans.fit(pdCLU)
pdCLU_3d['cluster'] = pd.Series(clusters.labels_, index=pdCLU.index)
# We can visualize cluster shapes in 3d using plotly.
import plotly.plotly as py

cluster1=pdCLU_3d.loc[pdCLU_3d['cluster'] == 0]
cluster2=pdCLU_3d.loc[pdCLU_3d['cluster'] == 1]

scatter1 = dict(mode = "markers", name = "Cluster 0", type = "scatter3d",    
x = cluster1.as_matrix()[:,0], 
y = cluster1.as_matrix()[:,1], 
z = cluster1.as_matrix()[:,2],
marker = dict(size=6, color='green')
               )
scatter2 = dict(mode = "markers",name = "Cluster 1",type = "scatter3d",    
x = cluster2.as_matrix()[:,0], 
y = cluster2.as_matrix()[:,1], 
z = cluster2.as_matrix()[:,2],
marker = dict( size=6, color='blue')
              )

cluster1 = dict(alphahull = 5, name = "Cluster 1", opacity = .1, type = "mesh3d",    
x = cluster1.as_matrix()[:,0], y = cluster1.as_matrix()[:,1], z = cluster1.as_matrix()[:,2],
color='#ff0000', showscale = True)

cluster2 = dict(alphahull = 5,name = "Cluster 2",opacity = .1, type = "mesh3d",    
x = cluster2.as_matrix()[:,0], y = cluster2.as_matrix()[:,1], z = cluster2.as_matrix()[:,2],
color='cyan', showscale = True)

layout = dict(title = 'Interactive Cluster Shapes in 3D',
              scene = dict( xaxis = dict( zeroline=True ),yaxis = dict( zeroline=True ),
             zaxis = dict( zeroline=True )))

fig = dict( data=[scatter1, scatter2, cluster1, cluster2], layout=layout )

# Use py.iplot() for IPython notebook
plotly.offline.iplot(fig, filename='mesh3d_sample')
pdCLU['y']=pd.Series(clusters.labels_, index=pdCLU.index)

# split df into cluster groups
grouped = pdCLU.groupby(['y'], sort=True)

# compute sums for every column in every group
sums = grouped.sum()
sums
data = [go.Heatmap( z=sums.values.tolist(), 
                   y=['Cluster 0', 'Cluster 1'],
                   x=pdCLU.columns.tolist(),
                   colorscale='Viridis')]
plotly.offline.iplot(data, filename='pandas-heatmap')
from pandas.tools.plotting import parallel_coordinates
plt.style.use('ggplot')

plt.rc('xtick', labelsize=19)                                           
plt.rc('ytick', labelsize=32)
plt.figure(figsize=(20,17))
ax4=parallel_coordinates(pdCLU, 'y')     
plt.xlabel('YEARS', size=32)                
plt.ylabel('Unit Labour Costs - index ref. 2010', size=40)           
plt.title('PARALLEL PLOT ULCs 1995-2006 by clusters',size=40)          
plt.legend(loc='upper left', prop={'size': 38}) 
ax4.set_facecolor('lightcyan')
plt.show()