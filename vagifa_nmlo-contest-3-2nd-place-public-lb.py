# Essential Imports
import numpy as np 
import pandas as pd 


# Visualisations
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics
from scipy.stats import skew,boxcox_normmax
from scipy.special import boxcox1p

# Misc
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

# Models
from sklearn.linear_model import SGDRegressor
from sklearn.cluster import KMeans,AgglomerativeClustering

# For warnings
import warnings
warnings.filterwarnings(action="ignore")


# Read in train and test set and set our seaborn plot style
train = pd.read_csv('/kaggle/input/nmlo-contest-3/train.csv')
test = pd.read_csv('/kaggle/input/nmlo-contest-3/test.csv')

sns.set_style('darkgrid')
# Get shapes of data
print('Train Shape:' + str(train.shape))
print('Test Shape:' + str(test.shape))
# Get the head of our training set to see what our features look like
train.head()
# Some info on our data
train.info()
# Describe our data to get some basic statistics about it
train.describe()
# Get skewness and kurtosis of data
print('Median Income skew in train set: ' + str(skew(train['inc'])))
print('Median Income kurtosis in train set: ' + str((train['inc'].kurt())))
print('')
print('Population skew in train set: ' + str(skew(train['pop'])))
print('Population kurtosis in train set: ' + str((train['pop'].kurt())))
print('')
print('Target feature skew in train set: ' + str(skew(train['cases'])))
print('Target feature kurtosis in train set: ' + str((train['cases'].kurt())))
print('')
print('Education feature skew in train set: ' + str(skew(train['ed'])))
print('Education feature kurtosis in train set: ' + str((train['ed'].kurt())))
# Box plot of the cases log transformed 
sns.boxplot(np.log1p(train['cases']))
plt.title('Box Plot of cases in the train set')
plt.show()
# Histogram of the cases log transformed 
sns.distplot(np.log1p(train['cases']))
plt.title('Distribution Plot of cases in the train set')
plt.show()
# Box plot of the population log transformed 
sns.boxplot(np.log1p(train['pop']))
plt.title('Box Plot of population in the train set')
plt.show()
# Histogram of the population log transformed 
sns.distplot(np.log1p(train['pop']))
plt.title('Distribution Plot of population in the train set')
plt.show()
# Box plot of the eduction feature
sns.boxplot(train['ed'])
plt.title('Box Plot of education levels in the train set')
plt.show()
# Box plot of the eduction feature
sns.distplot(train['ed'])
plt.title('Distribtion Plot of education levels in the train set')
plt.show()
# Box plot of the income feature
sns.boxplot(np.log1p(train['inc']))
plt.title('Box Plot of income levels in the train set')
plt.show()
# Box plot of the income feature
sns.distplot((train['inc']))
plt.title('Distribution Plot of income levels in the train set')
plt.show()
# Pairplot of all training features
sns.pairplot(np.log1p(train))
plt.show()
# Linear Model plot of cases and population
sns.lmplot('pop','cases',np.log1p(train),line_kws={'color': 'red'})
plt.title('Linear model plot of cases vs population')
plt.show()
# Scatter plot of inc vs population split by number of cases
plt.scatter('pop','inc',data=np.log1p(train),c='cases',alpha=0.3,cmap='coolwarm')
plt.title('Population vs income split by cases on train set')
plt.show()
# Correlation heatmap
plt.figure(figsize=(10,10))
sns.heatmap(train.corr(),annot=True,cmap='coolwarm')
plt.show()
# Check if there are any rows where amount of cases is more that the population)
train.iloc[np.where(train['cases'] > train['pop'])[0]]
# Remove that example and then drop the id columns completey
train = train[train['id'] != 724]
train.drop('id',axis=1,inplace=True)
# See if there is any education feature that is zero
test[test['ed'] == 0]
# Replace the outlier values with the 25% percentile of the education feature 
test['ed'] = test['ed'].replace(0.0,test['ed'].quantile([0.25][0]))
# Log transform cases so they follow normal distribution
train['cases'] = np.log1p(train['cases'])
# Remove all examples with target value greater than 11
train = train[(train['cases'] < 11)]

# Remove all income values with a value greater than 110000
train = train[(train['inc'] < 110000)]

# Remove all population values with a population greater than 4000000
train = train[(train['pop'] < 4000000)]
# Select labels and features and merge train and test data together
train_features = train.drop('cases',axis=1)
test_features = test
y = train.cases

combined = pd.concat([train_features,test_features],axis=0).reset_index(drop=True)
combined.shape
skewed_features = combined.apply(lambda x:skew(x)).sort_values(ascending=False)

high_skew = skewed_features[skewed_features > 0.5]
skew_index = high_skew.index
skewness = pd.DataFrame({'Skew' :high_skew})
skewness.head()
for i in skew_index:
    combined[i] = boxcox1p(combined[i],boxcox_normmax(combined[i] + 1))
from sklearn.preprocessing import normalize
# Fuction to create a new feature that clusters similiar examples together using Hierarchical Clustering
def hierarchical_cluster_feature():
    data_scaled = normalize(combined)
    data_scaled = pd.DataFrame(data_scaled,columns=combined.columns)
    cluster = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
    cluster.fit(data_scaled)
    combined['hierarchical_cluster_feature'] = cluster.labels_
    
hierarchical_cluster_feature()
# Fuction to create a new feature that clusters similiar examples together using KMeans Clustering
def generate_cluster_feature():
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined.drop('hierarchical_cluster_feature',axis=1))
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_scaled)
    kmeans = KMeans(n_clusters=2,max_iter=400,random_state=42)
    kmeans.fit(combined_pca)
    combined['cluster_feature'] = kmeans.labels_

# Call the function to generate clusters
generate_cluster_feature()
# Divide income by population
combined['inc_over_pop'] = combined['inc'] / combined['pop']
# Divide education over population
combined['ed_over_pop'] = combined['ed'] / combined['pop']
# Divide education over income
combined['inc_over_ed'] = combined['inc'] / combined['ed']
# Function to calculate log transformation of features
def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   
        res.columns.values[m] = l + '_log'
        m += 1
    return res

# Call the function
combined = logs(combined, ['ed','pop','ed_over_pop'])

# Function to calculate square transformation of features
def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   
        res.columns.values[m] = l + '_sq'
        m += 1
    return res 

# Call the function
combined = squares(combined, ['ed','pop','ed_log','pop_log','ed_over_pop_log','ed_over_pop'])
X = combined.iloc[:len(train),:]
X_test = combined.iloc[len(train):,:]

X.shape,X_test.shape
X
# Setup cross validation KFolds
kf = KFold(n_splits=12,random_state=42,shuffle=True)
# Split data into train and validation set
for train_index,val_index in kf.split(X):
    X_train,X_val = X.iloc[train_index],X.iloc[val_index],
    y_train,y_val = y.iloc[train_index],y.iloc[val_index]
# Initialize SGD Model and fit to training set
sgd = make_pipeline(StandardScaler(),SGDRegressor(random_state=42,n_iter_no_change=33))
sgd.fit(X_train,y_train)
np.sqrt(mean_squared_error(y_val,sgd.predict(X_val)))
# Read in sample_submission dataframe
submission = pd.read_csv("../input/nmlo-contest-3/sample.csv")
submission.shape
# Append predictions from sgd model
submission.iloc[:,1] = np.floor(np.expm1(sgd.predict(X_test)))
# Scale predictions
submission['cases'] *= 1.001619
submission.to_csv("submission_sgd.csv", index=False)