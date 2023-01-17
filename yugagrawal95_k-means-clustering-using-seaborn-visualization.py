#import important library such as pandas, seaborn, sklearn etc.

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#import data from csv to dataframe using pandas..
df = pd.read_csv('../input/cars.csv')

#as if you see dataframe, column name contain spaces on left side for convenience removing it... 
df.columns = df.columns.str.replace(' ', '')
#dataset
df.head()
# in data there is 3 brand US, Europe and Japan as categorical variable,
# converting categorical variable to numerical variable....

le = LabelEncoder().fit(df['brand'])
df['brand'] = le.transform(df['brand'])


#we have certain blank spaces value which needed to remove at start..
df = df.loc[df.cubicinches != ' ']
df = df.loc[df.weightlbs != ' ']


# both column have object type lets convert it to integer type.
df[['cubicinches', 'weightlbs']] = df[['cubicinches', 'weightlbs']].astype(int)
# as we have converted brand into numerical value for refrence vcalue mapping.
L = list(le.inverse_transform(df['brand']))
d = dict(zip(le.classes_, le.transform(le.classes_)))
print (d)
#check info of data..
df.info()
sns.violinplot('cylinders','mpg',data=df,palette='coolwarm')
sns.countplot(x='year',hue='brand',data=df)
g = sns.FacetGrid(col='year',hue='brand',data=df,legend_out=False)
g.map(sns.scatterplot,'hp','time-to-60')
g = sns.FacetGrid(col='cylinders',data=df,legend_out=False)
g.map(sns.scatterplot,'hp','mpg')
g = sns.FacetGrid(col='brand',data=df,legend_out=False)
g.map(sns.distplot,'weightlbs')
g = sns.FacetGrid(col='cylinders',data=df,legend_out=False)
g.map(sns.distplot,'weightlbs')
sns.scatterplot(df.cubicinches,df.weightlbs)
sns.boxplot('cylinders','time-to-60',data=df)
# get right number of cluster for K-means so we neeed to loop from 1 to 20 number of cluster and check score.
#Elbow method is used to represnt that. 
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(df).score(df) for i in range(len(kmeans))]
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
#fitting data in Kmeans theorem.
kmeans = KMeans(n_clusters=3, random_state=0).fit(df)
# this creates a new column called cluster which has cluster number for each row respectively.
df['cluster'] = kmeans.labels_
df.loc[df.cluster == 0].count()
sns.swarmplot(df.cluster,df.mpg)
sns.swarmplot(df.cluster,df.cylinders)
sns.boxplot(df.cluster,df.weightlbs)
sns.factorplot('cluster','hp',data=df)
sns.jointplot(x='cluster', y='year', data=df, alpha=.25,
              color='k', marker='*')
sns.jointplot(x='cluster', y='cubicinches', data=df, size=8, alpha=.25,
              color='k', marker='*')
sns.swarmplot(df.cluster,df['time-to-60'])
sns.countplot(x='cluster',hue='brand',data=df)
