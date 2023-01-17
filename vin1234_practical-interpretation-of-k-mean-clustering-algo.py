import numpy as np                                   # Implemennts milti-dimensional array and matrices

np.set_printoptions(precision=4)                     # To display values only upto four decimal places. 



import pandas as pd                                  # For data manipulation and analysis

pd.set_option('mode.chained_assignment', None)       # To suppress pandas warnings.

pd.set_option('display.max_colwidth', -1)            # To display all the data in the columns.

pd.options.display.max_columns = 40                  # To display all the columns.



import pandas_profiling



import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn-whitegrid')    # To apply seaborn whitegrid style to the plots.

plt.rc('figure', figsize=(10, 8))     # Set the default figure size of plots.

%matplotlib inline





import warnings

warnings.filterwarnings('ignore')     # To suppress all the warnings in the notebook.



from sklearn.metrics import classification_report, confusion_matrix
# Importing the Dataset

college_data = pd.read_csv("/kaggle/input/us-news-and-world-reports-college-data/College.csv",index_col=0)
college_data.head()
college_data.info()
college_data.describe()
# Performing pandas profiling before data preparation

# Saving the output as data_before_preprocessing.html



# To output pandas profiling report to an external html file.

'''

profile = college.profile_report(title='Pandas Profiling before Data Preprocessing')

profile.to_file(output_file="data_before_preprocessing.html")

'''



# To output the pandas profiling report on the notebook.



college_data.profile_report(title='Pandas Profiling before Data Preprocessing', style={'full_width':True})
# Droping the Highly correlated features 

college_data.drop(['Apps','Enroll','F.Undergrad'],inplace=True,axis=1)
college_data.head()
college_data['Private'].value_counts()
# map the feature to integer integer value.

college_data['Private']=college_data['Private'].map({'Yes':1,"No":0})
college_data.head()
# To output the pandas profiling report on the notebook.



college_data.profile_report(title='Pandas Profiling after Data Preprocessing', style={'full_width':True})
sns.set_style('whitegrid')

sns.lmplot('Room.Board','Grad.Rate',data=college_data, hue='Private',

           palette='coolwarm',size=6,aspect=1,fit_reg=False)
sns.set_style('whitegrid')

sns.lmplot('Outstate','P.Undergrad',data=college_data, hue='Private',

           palette='coolwarm',size=6,aspect=1,fit_reg=False)
c=college_data[college_data['P.Undergrad']>10000].index.values

c
college_data.head()
college_data.shape
# Storing the target variable

y=college_data['Private']



college_data.drop(['Private'],inplace=True,axis=1)
college_data.head()
# Import KMeans from SciKit Learn.

from sklearn.cluster import KMeans



# Create an instance of a K Means model with 2 clusters

kmeans=KMeans(n_clusters=2)
from sklearn import preprocessing

# Scaling the data

minmax_processed = preprocessing.MinMaxScaler().fit_transform(college_data)

college_data.columns
df_numeric_scaled = pd.DataFrame(minmax_processed, index=college_data.index, columns=college_data.columns)
df_numeric_scaled.head()
# Fit the model to all the data except for the Private label.

kmeans.fit(df_numeric_scaled)
# What are the cluster center vectors?

kmeans.cluster_centers_
print(confusion_matrix(y,kmeans.labels_))

# print(classification_report(y,kmeans.labels_))
#Let's fit cluster size 1 to 20 on our data and take a look at the corresponding score value.

Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(df_numeric_scaled).score(df_numeric_scaled) for i in range(len(kmeans))]
plt.plot(Nc,score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()
from sklearn.decomposition import PCA
# Here we compress the data to two dimension

pca=PCA(n_components=2)

principalComponents = pca.fit_transform(df_numeric_scaled)
principalComponents
# Plot the explained variances

# features = range(pca.n_components_)

# plt.bar(features, pca.explained_variance_ratio_, color='black')

# plt.xlabel('PCA features')

# plt.ylabel('variance %')

# plt.xticks(features)
# Save components to a DataFrame

PCA_components = pd.DataFrame(principalComponents)
PCA_components.head()
k_means2=KMeans(n_clusters=2)
# Computer cluster centers and predict cluster indices 

X_clustered=k_means2.fit_predict(PCA_components)
# Define your own color map

Label_color_map={0:'r',1:'g'}

label_color=[Label_color_map[i] for i in X_clustered]
# Plot the scatter diagram

plt.figure(figsize=(7,5))

plt.scatter(principalComponents[:,0],principalComponents[:,1],c=label_color,alpha=0.5)

plt.show()
# plot the centroid

center=k_means2.cluster_centers_



plt.scatter(principalComponents[:,0],principalComponents[:,1],c=label_color,alpha=0.5)

plt.scatter(center[:, 0], center[:, 1], c='blue', s=300, alpha=0.9,label = 'Centroids')



plt.show()