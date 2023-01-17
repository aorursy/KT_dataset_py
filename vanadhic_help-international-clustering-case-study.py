# Import the required packages for analysis and model building

import pandas as pd

import numpy as np 



import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt



import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree



import warnings

warnings.filterwarnings('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/help-international/Country-data.csv')
df.head()
df.info()
df.describe(percentiles= [.1, .05, .25, .5, .75, .95, .99])
# Check if there are any missing values

df.isnull().sum()
# Change the variables from percentage of Total GDPP to actual values



df['exports']=(df['exports']*df['gdpp'])/100

df['health']=(df['health']*df['gdpp'])/100

df['imports']=(df['imports']*df['gdpp'])/100



#Set the index value as Country to do analysis of numerical columns

df.set_index('country',inplace=True)



df.head()
plt.figure(figsize = (25,25))

sns.pairplot(df)
# Heatmap to determine the correlation between the features. 

sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")
def plotdata(data):

    color = data.life_expec

    area = 5e-2 * data.gdpp

    data.plot.scatter('child_mort','income',

                      s=area,c=color,

                      colormap=matplotlib.cm.get_cmap('Purples_r'), vmin=45, vmax=90,

                      linewidths=1,edgecolors='k',

                      figsize=(20,15))

    

    # labeling different cluster points with country names 

    for i, txt in enumerate(data.index):

        if txt == 'India':

            plt.annotate(txt, (data.child_mort[i],data.income[i]), fontsize=25, ha='left', rotation=25)

        plt.annotate(txt, (data.child_mort[i],data.income[i]), fontsize=10, ha='left', rotation=25)

    

    plt.title('Countries based on child_mort, income, gdpp and life_expec',fontsize=20)

    plt.xlabel('child_mort',  fontsize=20)

    plt.ylabel('income', fontsize=20)

    plt.tight_layout()    

    plt.show()
plotdata(df)
# Distplot to see the distribution of the features

plt.figure(figsize = (15,15))

features = df.columns[:-1]

for i in enumerate(features):

    plt.subplot(3,3,i[0]+1)

    sns.distplot(df[i[1]])
# Box plot to identify the outliers

plt.figure(figsize=(20, 15))

for i, x_var in enumerate(df.columns[:-1]):

    plt.subplot(3,3,i+1)

    sns.boxplot(x = x_var, data = df)
# To find out the one outlier Country for the inflation feature. 

df.sort_values('inflation', ascending=False).head(10)
# Copy the original dataset to a different dataset to cap the Outliers. This is essential to keep the original dataset

df_capped = df.copy()

cap_outliers = ['exports', 'health', 'imports', 'income', 'gdpp', 'inflation']
# For each of the features in remove_outlier, cap the outliers in the upper end. 

# Only the outliers in the upper end need to be capped. 

for i, var in enumerate(cap_outliers):

    q4 = df[var].quantile(0.95)

    df_capped[var][df_capped[var]>=q4] = q4
df_capped.describe()
# Plot the boxplot for the features 

plt.figure(figsize=(20, 15))

for i, x_var in enumerate(df_capped.columns[:-1]):

    plt.subplot(3,3,i+1)

    sns.boxplot(x = x_var, data = df_capped)
# Importing the scaling library - StandardScaler

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
# Scaling the dataset with Standard Scaler 

df_scaled=pd.DataFrame(scaler.fit_transform(df_capped),columns=df_capped.columns, index=df_capped.index)

df_scaled.head()
#Calculating the Hopkins statistic

from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

import numpy as np

from math import isnan

 

def hopkins(X):

    d = X.shape[1] # columns

    n = len(X) # rows

    m = int(0.1 * n) 

    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

 

    rand_X = sample(range(0, n, 1), m)

 

    ujd = []

    wjd = []

    for j in range(0, m):

        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)

        ujd.append(u_dist[0][1])

        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)

        wjd.append(w_dist[0][1])

 

    H = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(H):

        print(ujd, wjd)

        H = 0

 

    return H
hopkins(df_capped)
hopkins(df_scaled)
# Elbow-curve/SSD

ssd = []

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)

    kmeans.fit(df_scaled)

    ssd.append(kmeans.inertia_)

    

# plot the SSDs for each n_clusters starting from 2 clusters

xi = list(range(len(range_n_clusters)))

plt.plot(xi, ssd) 

plt.xlabel('Number of clusters')

plt.ylabel('Sum of squared distances') 

plt.xticks(xi, range_n_clusters)

plt.title('Elbow-curve Analysis')

plt.show()
# silhouette analysis

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

s_score = []



for num_clusters in range_n_clusters:

    # intialise kmeans

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)

    kmeans.fit(df_scaled)

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(df_scaled, cluster_labels)

    s_score.append(silhouette_avg)

    

    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg)) 

    

# plot the Silhouette score for each n_clusters starting from 2 clusters

xi = list(range(len(range_n_clusters)))

plt.plot(xi, s_score) 

plt.xlabel('Number of clusters')

plt.ylabel('Silhoutte scores') 

plt.xticks(xi, range_n_clusters)

plt.title('Silhoutte Score Analysis')

plt.show()
# This function is to plot the scatter plot of countries based on child_mort, income and gdpp. 

# The child_mort is along the x-axis, income along the y-axis and the size of the points relational to the gdpp



def plotdata_cluster(data, title):

    area = 5e-2 * data.gdpp

    colors = data.cluster_id.map({0: 'skyblue', 1: 'gold', 2: 'coral', 3: 'palegreen'})

    

    data.plot.scatter('child_mort','income',

                      s=area, c=colors,

                      linewidths=1,edgecolors='k',

                      figsize=(20,15))

        

    # labeling different cluster points with country names 

    for i, txt in enumerate(data.index):

        if txt == 'India':

            plt.annotate(txt, (data.child_mort[i],data.income[i]), fontsize=25, ha='left', rotation=25)

        plt.annotate(txt, (data.child_mort[i],data.income[i]), fontsize=12, ha='left', rotation=25)

    

    #plt.title('Countries clusters based on child_mort, income and gdpp after Outlier removal',fontsize=20)

    plt.title(plot_title,fontsize=20)

    plt.xlabel('child_mort',  fontsize=20)

    plt.ylabel('income', fontsize=20)

    plt.tight_layout()
# This function is to plot the scatter plot of countries based on child_mort, life_expec and health (expenditure). 

# The child_mort is along the x-axis, life_expec along the y-axis and the size of the points relational to the health.



def plotdata_health(data, plot_title):

    area = 50e-1 * data.health

    colors = data.cluster_id.map({0: 'skyblue', 1: 'gold', 2: 'coral', 3: 'palegreen'})

    

    data.plot.scatter('child_mort','life_expec',

                      s=area, c=colors,

                      linewidths=1,edgecolors='k',

                      figsize=(20,15))

        

    # labeling different cluster points with country names 

    for i, txt in enumerate(data.index):

        if txt == 'India':

            plt.annotate(txt, (data.child_mort[i],data.life_expec[i]), fontsize=25, ha='left', rotation=25)

        plt.annotate(txt, (data.child_mort[i],data.life_expec[i]), fontsize=12, ha='left', rotation=25)

    

    #plt.title('Countries clusters based on child_mort, life_expec and health expenditure',fontsize=20)

    plt.title(plot_title,fontsize=20)

    plt.xlabel('child_mort',  fontsize=20)

    plt.ylabel('life_expec', fontsize=20)

    plt.tight_layout()
# K-Means clustering with 3 clusters

# Set the random_state to a fixed value so that the labels do not change for each iteration

kmeans_3=KMeans(n_clusters=3, max_iter=100, random_state=50)    # k=3 and iteration=100

kmeans_3.fit(df_scaled)                                         # fitting the dataset
kmeans_3.labels_
# Appending the cluster labels to the original capped dataset 

df_cap_kmeans3 = df_capped.copy()

df_cap_kmeans3['cluster_id']=kmeans_3.labels_

#df_cap_kmeans3.head()
# Plot the scatter plot of countries for Kmeans clustering with 3 clusters

# child_mort along the x axis, income along the y-axis and size of the scatter point relative to the gdpp 

plot_title = 'Countries based on child_mort, income and gdpp'

plotdata_cluster(df_cap_kmeans3, plot_title)
# Final model: K Means clustering with k=4

kmeans_4=KMeans(n_clusters=4, max_iter=100, random_state=50)    # k=4 and iteration=500

kmeans_4.fit(df_scaled) # fitting the dataset
# Cluster labels for Kmeans with k=4

kmeans_4.labels_
# Appending the cluster labels to the original capped dataset  

df_cap_kmeans4 = df_capped.copy()

df_cap_kmeans4['cluster_id']=kmeans_4.labels_
# Plot the scatter plot of countries for Kmeans clustering with 3 clusters

# child_mort along the x axis, income along the y-axis and size of the scatter point relative to the gdpp 

plot_title = 'Countries based on child_mort, income and gdpp - 4 Clusters'

plotdata_cluster(df_cap_kmeans4, plot_title)
# This function is to plot the scatter plot of countries based on child_mort, life_expec and health (expenditure). 

# The child_mort is along the x-axis, life_expec along the y-axis and the size of the points relational to the health.



plot_title = 'Countries based on child_mort, life_expec and health expenditure - 4 Clusters'

plotdata_health(df_cap_kmeans4, plot_title)
# Define a function to create the snake plots for segment analysis

def snake_plot(data, plot_title):

    sns.set_style('whitegrid')

    df_melt = pd.melt(data.reset_index(), id_vars = ['country','cluster_id'], 

                      var_name = 'Feature', value_name = 'Value')



    plt.figure(figsize = (12,6))

    sns.lineplot(x = 'Feature', y = 'Value', data = df_melt,

                 hue = 'cluster_id', palette = "muted")

    

    plt.title(plot_title,fontsize=20)

    plt.xlabel('Features',  fontsize=20)

    plt.ylabel('Scaled Values', fontsize=20)

    plt.tight_layout()    

    

    # To print the number of Countries falling under each clusters

    print (title)

    print (df_sp.cluster_id.value_counts())
# Create Dataset for Snake plots - Scaled dataset should be used for the snake plot

df_sp = df_scaled.copy()
# Use the labels from the KMeans clustering with 3 clusters for snake plot

df_sp['cluster_id'] = kmeans_3.labels_ 



# snake plot for 3 clusters

title = 'Segment Analysis - KMeans with 3 clusters'

snake_plot(df_sp, title)
# Use the labels from the KMeans clustering with 4 clusters for snake plot

df_sp['cluster_id'] = kmeans_4.labels_ 



# snake plot for 3 clusters

plot_title = 'Segment Analysis - KMeans with 4 clusters'

snake_plot(df_sp, plot_title)
# Relative Importance plot to get the relative importance of each feature for the clusters 

def relative_imp_plot(data, plot_title):

    cluster_mean = data.groupby(['cluster_id']).mean()

    pop_mean = data.mean()

    relative_imp = cluster_mean / pop_mean - 1

    

    plt.figure(figsize=(10, 4))

    plt.title(plot_title, fontsize=20)

    sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdBu', vmin=-4, vmax=4)

    plt.show()
# Relative Importance plot to get the relative importance of each feature for the clusters - Kmeans 4 clusters

plot_title = 'Relative Importance plot - KMeans with 4 clusters'

relative_imp_plot(df_cap_kmeans4, plot_title)
# Relative Importance plot to get the relative importance of each feature for the clusters - Kmeans 3 clusters

plot_title = 'Relative Importance plot - KMeans with 3 clusters'

relative_imp_plot(df_cap_kmeans3, plot_title)
cluster_analysis = df_cap_kmeans4.groupby('cluster_id').agg({'child_mort': 'mean', 'exports': 'mean', 

                             'health': 'mean','imports': 'mean', 'income': 'mean', 'inflation': 'mean', 

                             'life_expec': 'mean', 'total_fer': 'mean', 'gdpp': ['mean', 'count']}).round(0)

cluster_analysis
# Analyse the following features for the summary and boxplot analysis

summary_cols = ['gdpp', 'income', 'life_expec',

                'child_mort', 'total_fer','inflation' ]
# Summary Analysis for mean of the features for each of the clusters

sns.set(style='whitegrid')

plt.figure(figsize=(20, 10))

for i, x_var in enumerate(summary_cols):

    plt.subplot(2,3,i+1)

    sns.barplot(x = cluster_analysis.reset_index().cluster_id, 

                y = cluster_analysis[x_var]['mean'])

    plt.ylabel(x_var, fontsize=15)

    plt.xlabel('cluster_id', fontsize=15)

plt.tight_layout()    

plt.show()    
profiling_cols = ['gdpp', 'income', 'child_mort']



sns.set(style='white')

# plt.title('Cluster Profiling based on gdpp, income and child_mort')

plt.figure(figsize=(12,6))



for i, x_var in enumerate(profiling_cols):

    plt.subplot(1,3,i+1)

    sns.boxplot(x = df_cap_kmeans4.cluster_id, 

                y = df_cap_kmeans4[x_var])

    plt.ylabel(x_var, fontsize=20)

    plt.xlabel('cluster_id', fontsize=20)

    plt.tight_layout()    

plt.show()    
# This function is to plot the scatter plot of countries based on child_mort, income and gdpp. 

# The child_mort is along the x-axis, income along the y-axis and the size of the points proportionalto the gdpp

def plotdata_aid(data, plot_title):

    area = 50e-2 * data.gdpp

    colors = data.cluster_id.map({0: 'skyblue', 1: 'gold', 2: 'coral', 3: 'palegreen'})

    

    data.plot.scatter('child_mort','income',

                      s=area,c=colors,

                      linewidths=1,edgecolors='k',

                      figsize=(20,15))

        

    # labeling different cluster points with country names 

    for i, txt in enumerate(data.index):

        if txt == 'India':

            plt.annotate(txt, (data.child_mort[i],data.income[i]), fontsize=20, ha='left', rotation=25)

        plt.annotate(txt, (data.child_mort[i],data.income[i]), fontsize=12, ha='left', rotation=25)

    

    plt.title(plot_title,fontsize=20)

    plt.xlabel('child_mort',fontsize=20)

    plt.ylabel('income',fontsize=20)
# Plot the scatter plot for the Countries in need of financial aid 

# The child_mort is along the x-axis, income along the y-axis and the size of the points relational to the gdpp



plot_title = 'Socio-economically poor and developing Countries - KMeans Clusters 2 and 0'

plotdata_aid(df_cap_kmeans4[(df_cap_kmeans4.cluster_id == 2) | (df_cap_kmeans4.cluster_id == 0)], plot_title)
# Get the countries that are in need of aid

df_aid = df.copy()

df_aid['cluster_id'] = kmeans_4.labels_
# Get the countries from Cluster 2 (Poor Countries)

df_aid = df_aid.query('cluster_id == 2').sort_values(['child_mort', 'gdpp', 'income', 'life_expec' ], 

                                                   ascending = (False, True, True, True))

df_aid.head()
# Get the rank for individal features for each Country

df_aid['cmrank'] = df_aid.child_mort.rank(method='dense',ascending = False).astype(int)

df_aid['lerank'] = df_aid.life_expec.rank(method='dense',ascending = True).astype(int)

df_aid['gdrank'] = df_aid.gdpp.rank(method='dense',ascending = True).astype(int)

df_aid['inrank'] = df_aid.income.rank(method='dense',ascending = True).astype(int)

df_aid['ifrank'] = df_aid.inflation.rank(method='dense',ascending = False).astype(int)
df_aid.head(5)
# * **'child_mort' - Top 5 Countries with highest child mortality rate**

# * **'life_expec' - Top 3 Countries with lowest life expectancy**

# * **'gdpp'       - Top 3 countries with lowest gdpp**

# * **'income'     - Top 3 countries with lowest income**

# * **'inflation'  - Nigeria has abnormally high inflation rate.** 



df_final = df_aid[(df_aid.cmrank <=5) | (df_aid.lerank <=3) | 

                  (df_aid.gdrank <=3) | (df_aid.inrank <=3) | (df_aid.ifrank <=1)]

df_final
df_final.index
# Plot the Hierarchical Clustering Dendrogram for Single linkage

def plot_dendrogram_single(data):

    plt.figure(figsize=(15,8))             # Setting the size of the figure

    sns.set_style('white')                  # Setting style



    # setting the labels on axes and title

    plt.title('Hierarchical Clustering Dendrogram - Single linkage',fontsize=20)

    plt.xlabel('Country',fontsize=20)

    plt.ylabel('Values',fontsize=20)



    mergings_s = linkage(data, method = "single", metric='euclidean') # Use the df_scaled dataset

    dendrogram(mergings_s, labels=data.index, leaf_rotation=90, leaf_font_size=6)

    plt.show()

    return mergings_s
# Plot the Hierarchical Clustering Dendrogram for Complete linkage

def plot_dendrogram_complete(data):

    plt.figure(figsize=(15,8))             # Setting the size of the figure

    sns.set_style('white')                  # Setting style



    # setting the labels on axes and title

    plt.title('Hierarchical Clustering Dendrogram - Complete linkage',fontsize=20)

    plt.xlabel('Country',fontsize=20)

    plt.ylabel('Values',fontsize=20)



    mergings_c = linkage(data, method = "complete", metric='euclidean')

    dendrogram(mergings_c, labels=data.index, leaf_rotation=90, leaf_font_size=6)

    plt.show()

    return mergings_c
# Plot the dendrogram for the normalized dataset with single linkage

mergings_s = plot_dendrogram_single(df_scaled)
# Plot the dendrogram for the normalized dataset with single linkage

mergings_c = plot_dendrogram_complete(df_scaled)
cap_outliers
# Let us cap the outliers for both the upper and lower ends and then do the Hierarchical clusering

df_capped1 = df.copy()



for i, var in enumerate(cap_outliers):

    q1 = df[var].quantile(0.05)

    q4 = df[var].quantile(0.95)

    df_capped1[var][df_capped1[var]<=q1] = q1

    df_capped1[var][df_capped1[var]>=q4] = q4



df_scaled1=pd.DataFrame(scaler.fit_transform(df_capped1),columns=df_capped1.columns, index=df_capped1.index)

df_scaled1.head()
mergings_cc = plot_dendrogram_complete(df_scaled1)
# 3 clusters - use the mergings from Complete linkage

cluster_labels_3 = cut_tree(mergings_cc, n_clusters=3).reshape(-1, )

cluster_labels_3
# To visualise let us use the df_capped1 dataset with the Outliers capped for upper outliers and lower outliers 

df_cap_hier3 = df_capped1.copy()



# assign cluster labels

df_cap_hier3['cluster_id'] = cluster_labels_3

df_cap_hier3.head()
# Cluster plot for Hierarchical cluster - 3 Clusters

plot_title = 'Cluster plot for Hierarchical cluster - 3 Clusters'

plotdata_cluster(df_cap_hier3, plot_title)
# 4 clusters - use the mergings from Complete linkage

cluster_labels_4 = cut_tree(mergings_cc, n_clusters=4).reshape(-1, )

cluster_labels_4
# To visualise let us use the df_capped1 dataset with the Outliers capped for upper outliers and lower outliers 

df_cap_hier4 = df_capped1.copy()



# assign cluster labels

df_cap_hier4['cluster_id'] = cluster_labels_4

df_cap_hier4.head()
# Cluster plot for Hierarchical cluster - 3 Clusters

plot_title = 'Cluster plot for Hierarchical cluster - 4 Clusters'

plotdata_cluster(df_cap_hier4, plot_title)
# Snake plot for Hierarchical clustering with 3 clusters 

# Use the labels from the Hierarchical clustering with 3 clusters for snake plot

df_sp['cluster_id'] = cluster_labels_3



# snake plot for 3 clusters

title = 'Segment Analysis - Hierarchical with 3 clusters'

snake_plot(df_sp, title)
# Snake plot for Hierarchical clustering with 4 clusters 

# Use the labels from the Hierarchical clustering with 4 clusters for snake plot

df_sp['cluster_id'] = cluster_labels_4



# snake plot for 3 clusters

title = 'Segment Analysis - Hierarchical with 4 clusters'

snake_plot(df_sp, title)
# Relative importance plot for the Hierarchical clustering with 4 clusters

plot_title = 'Relative importance plot for the Hierarchical clustering with 4 clusters'

relative_imp_plot(df_cap_hier4, plot_title)
# Relative importance plot for the Hierarchical clustering with 4 clusters

plot_title = 'Relative importance plot for the Hierarchical clustering with 3 clusters'

relative_imp_plot(df_cap_hier3, plot_title)
# The Countries in need of aid (Poor Countries) are in Cluster 0 

df_aid_hier4 = df.copy()

df_aid_hier4['cluster_id'] = cluster_labels_4

df_aid_hier4 = df_aid_hier4.query('cluster_id == 0').sort_values(['child_mort', 'life_expec', 'income', 'gdpp' ], 

                                                   ascending = (False, True, True, True))

df_aid_hier4.head(20)
# Get the rank for individal features for each Country

df_aid_hier4['cmrank'] = df_aid_hier4.child_mort.rank(method='dense',ascending = False).astype(int)

df_aid_hier4['lerank'] = df_aid_hier4.life_expec.rank(method='dense',ascending = True).astype(int)

df_aid_hier4['gdrank'] = df_aid_hier4.gdpp.rank(method='dense',ascending = True).astype(int)

df_aid_hier4['inrank'] = df_aid_hier4.income.rank(method='dense',ascending = True).astype(int)

df_aid_hier4['ifrank'] = df_aid_hier4.inflation.rank(method='dense',ascending = False).astype(int)
df_aid_hier4.head()
# * **'child_mort' - Top 5 Countries with highest child mortality rate**

# * **'life_expec' - Top 3 Countries with lowest life expectancy**

# * **'gdpp'       - Top 3 countries with lowest gdpp**

# * **'income'     - Top 3 countries with lowest income**

# * **'inflation'  - Nigeria has abnormally high inflation rate.** 



df_final_hier4 = df_aid_hier4[(df_aid_hier4.cmrank <=5) | (df_aid_hier4.lerank <=3) | 

                              (df_aid_hier4.gdrank <=3) | (df_aid_hier4.inrank <=3) | (df_aid_hier4.ifrank <=1)]

df_final_hier4
print (f'The following {len(df_final.index)} countries are ranked after KMeans clustering')

df_final.index
print (f'The following {len(df_final_hier4.index)} countries are ranked after Hierarchical clustering')

df_final_hier4.index
# This function is to plot the scatter plot of countries based on child_mort, income and gdpp. 

# The child_mort is along the x-axis, income along the y-axis and the size of the points proportionalto the gdpp

def plotdata_aid_final(data, plot_title):

    area = 50e-2 * data.gdpp

    colors = data.cluster_id.map({0: 'skyblue', 1: 'gold', 2: 'coral', 3: 'palegreen'})

    

    aid_countries = ['Haiti', 'Sierra Leone', 'Chad', 'Central African Republic', 'Mali',

                     'Nigeria', 'Congo, Dem. Rep.', 'Lesotho', 'Burundi', 'Liberia']

    data.plot.scatter('child_mort','income',

                      s=area,c=colors,

                      linewidths=1,edgecolors='k',

                      figsize=(20,15))

        

    # labeling different cluster points with country names 

    for i, txt in enumerate(data.index):

        if txt in aid_countries:

            plt.annotate(txt, (data.child_mort[i],data.income[i]), fontsize=20, ha='left', rotation=25)

        #plt.annotate(txt, (data.child_mort[i],data.income[i]), fontsize=12, ha='left', rotation=25)

    

    plt.title(plot_title,fontsize=20)

    plt.xlabel('child_mort',fontsize=20)

    plt.ylabel('income',fontsize=20)
# Plot the scatter plot for the Countries in need of financial aid 

# The child_mort is along the x-axis, income along the y-axis and the size of the points relational to the gdpp



plot_title = 'Final list of countries recommended for aid'

plotdata_aid_final(df_cap_kmeans4[(df_cap_kmeans4.cluster_id == 2) | (df_cap_kmeans4.cluster_id == 0)], plot_title)