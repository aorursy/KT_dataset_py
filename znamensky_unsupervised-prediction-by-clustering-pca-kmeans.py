import numpy as np

import pandas as pd

import seaborn as sns

sns.set_palette('husl')

import matplotlib.pyplot as plt

%matplotlib inline





from sklearn import metrics



from sklearn.preprocessing import LabelEncoder

from itertools import combinations



from sklearn.model_selection import train_test_split



from sklearn.preprocessing import StandardScaler



from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
#Read data and define proportion of classes



add = "../input/mushroom-classification/mushrooms.csv"

data = pd.read_csv(add)

y=data['class'].copy()

print('Proportion of e/p','\n',

      data['class'].value_counts())



def encode (data):

    label=LabelEncoder()

    for c in  data.columns:

        if(data[c].dtype=='object'):

            data[c]=label.fit_transform(data[c])

        else:

            data[c]=data[c]
# Creating combination from features



data1=data.drop('class',axis=1)



a=pd.DataFrame()

color_col=data1.columns

comb = combinations(color_col, 2) 

comb_feat=[]

# Print the obtained combinations 

for i in list(comb): 

    comb_feat.append(i)

comb_feat=pd.DataFrame(comb_feat)







for i in range(0,len(comb_feat),1):

    col1=comb_feat[0][i]

    col2=comb_feat[1][i]

    a[col1+col2]=data[col1]+data[col2]

    

df_cor=pd.concat([a,y],axis=1)

encode(df_cor)

corrmat= abs(df_cor.corr())



corrmat['class'].sort_values(ascending=False)[1:31]
new_feat=pd.DataFrame(corrmat['class'].sort_values(ascending=False)[1:31]).reset_index()

new_feat.columns=['feat','corr']



for i in new_feat['feat']:

    data1[i]=a[i]



data1.shape
data1= pd.get_dummies(data1)
encode(data1)
data1.columns
y=pd.DataFrame(data['class'].copy())

#data_x=data1.drop(['class'],axis=1)

encode(y)

y=y['class'].copy()

sc = StandardScaler()

data_x_sc=pd.DataFrame(sc.fit_transform(data1),columns=data1.columns)
#Define n_component to cover 90% of feature variation-110

def Reduction_compon(red_model,X,n_comp):

    red_ = red_model(n_components = n_comp)

    principalComponents = red_.fit_transform(X)



    # Plot the explained variances

    features = range(red_.n_components_)

    fig=plt.figure()

    

    plt.bar(features, red_.explained_variance_ratio_, color='black')

    plt.xlabel('PCA features')

    plt.ylabel('variance %')

    #plt.xticks(features)

    plt.show()

    # Save components to a DataFrame

    Red_components = pd.DataFrame(principalComponents)

    print('Cumsum_expl_var',np.cumsum(red_.explained_variance_ratio_))

    

    
#Should be 90%

Reduction_compon(PCA,data_x_sc,110)

#Reduction_compon(PCA,X_train,110)
#Look at the PCA features scatter plot

#Define n_clusters by Elbow_point

def red_model_scatter(red_model,comp,X,y):

    red_ = red_model(n_components = comp)

    principalComponents = red_.fit_transform(X)



    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y, 

                edgecolor='none', alpha=0.7, s=40,

                #cmap=plt.cm.get_cmap('nipy_spectral', 10)

               )

    fig=plt.figure()

    plt.xlabel('PCA 1')

    plt.ylabel('PCA 2')

    print('Model,components',str(red_model),comp)

    

    reduc_matrix=pd.DataFrame(principalComponents)

    

    #Find Elbow point and cluster number

    ks = range(1, 30)

    inertias = []

    for k in ks:

        # Create a KMeans instance with k clusters: model

        model = KMeans(n_clusters=k)



        # Fit model to samples

        model.fit(reduc_matrix.iloc[:,:3])



        # Append the inertia to the list of inertias

        inertias.append(model.inertia_)



    plt.plot(ks, inertias, '-o', color='black')

    plt.xlabel('number of clusters, k')

    plt.ylabel('inertia')

    plt.xticks(ks)

    #plt.show()

    print('Choose n_cluster by Elbow point')
red_model_scatter(PCA,110,data_x_sc,y)

#Function for dataset clusterization 

def clusteriz(n_cl,red_model,comp,X):

    

    red_ = red_model(n_components = comp)

    principalComponents = red_.fit_transform(X)

    reduc_matrix=pd.DataFrame(principalComponents)

    

    kmeans = KMeans(n_clusters=n_cl)

    kmeans.fit(reduc_matrix)

    

    y_kmeans = kmeans.predict(reduc_matrix)

    X_clust=pd.DataFrame(y_kmeans)

    X_clust.columns=['cluster']

    

    cluster_df = pd.DataFrame()



    cluster_df['cluster'] = X_clust['cluster']

    cluster_df['class'] = y

    sns.factorplot(col='cluster', y=None, x='class', data=cluster_df, kind='count', order=[1,0], palette=(["#7d069b","#069b15"]))

    return X_clust

X_clust=clusteriz(4,PCA,110,data_x_sc)

# We have 4 clusters, lets define which cluster means eatable/poison mushrooms

cluster_df = pd.DataFrame()



cluster_df['cluster'] = X_clust['cluster']

cluster_df['class'] = y





a=(pd.DataFrame(cluster_df.groupby(['cluster','class'])['class'].count()).unstack()).fillna(0)

a.iloc[:, a.columns.get_level_values('class')==0][:1]



cluster_pred_to_biclust=[]

clusters=[]



for i in range(0,len(a)):

    class0_val=int(a.iloc[:, a.columns.get_level_values('class')==0][i:i+1].values)

    class1_val=int(a.iloc[:, a.columns.get_level_values('class')==1][i:i+1].values)

    

    if class0_val>=class1_val:

        

        cluster_value=0

    else:

        cluster_value=1

        

    cluster_pred_to_biclust.append(cluster_value)

    

cluster_pred_to_biclust=pd.DataFrame(cluster_pred_to_biclust)

a=pd.concat([a,cluster_pred_to_biclust],axis=1)

list(a[0])
#Lets recode our 4 predicted clusters to 2 values and Calculate Accuracy

X_clust['clus_to_bi_pred']=0



X_clust['clus_to_bi_pred'] = (

    np.select(

        condlist=[X_clust['cluster']==0,

                  X_clust['cluster']==1,

                  X_clust['cluster']==2,

                  X_clust['cluster']==3],

        choicelist=list(a[0]), 

        default='-'))

X_clust['clus_to_bi_pred']=X_clust['clus_to_bi_pred'].astype(int)

accuracy=pd.Series(metrics.accuracy_score((X_clust['clus_to_bi_pred']).values,y))



print('Clustering Accuracy',accuracy)
