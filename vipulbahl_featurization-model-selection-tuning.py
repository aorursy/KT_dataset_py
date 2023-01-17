import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import zscore
import math as math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



%matplotlib inline
# sns.set(color_codes=True)
sns.set(style="darkgrid", color_codes=True)
#step 2.1: Read the dataset
rdata=pd.read_csv('/kaggle/input/yeh-concret-data/Concrete_Data_Yeh.csv')
# step 2.1: browse through the first few columns
rdata.head()
# Step 2.2: Understand the shape of the data
shape_data=rdata.shape
print('The shape of the dataframe is',shape_data,'which means there are',shape_data[0],'rows of observations and',shape_data[1],'attributes of age, ingredients and concrete compression strength')
# Step 2.3: Identify Duplicate records in the data 
# It is very important to check and remove data duplicates. 
# Else our model may break or report overly optimistic / pessimistic performance results
dupes=rdata.duplicated()
print(' The number of duplicates in the dataset are:',sum(dupes), '\n')
dupes_record=pd.DataFrame(rdata[dupes])
print(' The duplicate observations are:') 
dupes_record
# step 2.3.1: Remove duplicates from the data
t1data=rdata.copy()
t1data=t1data.drop_duplicates(keep="first")

dupes1=t1data.duplicated()
print(' The number of duplicates in the new dataset are:',sum(dupes1), '\n',
      'Clearly evident that now there are no duplicates in the dataset.')
t1data.columns=['cement','slag','ash','water','superplastic','coarseagg','fineagg','age','strength']
shape_t1data=t1data.shape
print('The shape of the new dataframe is',shape_t1data,'which means there are',shape_t1data[0],'rows of observations and',shape_data[1],'attributes of age, ingredients and concrete compression strength')
# Step 2.4: Lets analyze the data types
t1data.info()
#EDA 1: lets evaluate statistical details of the dataset. 
cname=t1data.columns
data_desc=t1data.describe().T
data_desc
# Attributes in the Group
Atr1g1='cement'
Atr2g1='slag'
Atr3g1='ash'
Atr4g1='water'
Atr5g1='superplastic'
Atr6g1='coarseagg'
Atr7g1='fineagg'
Atr8g1='age'
Atr9g1='strength'
#EDA 1: Outliar Detection leveraging Box Plot
data=t1data
fig, ax = plt.subplots(1,9,figsize=(38,16)) 
sns.boxplot(x=Atr1g1,data=data,ax=ax[0],orient='v') 
sns.boxplot(x=Atr2g1,data=data,ax=ax[1],orient='v')
sns.boxplot(x=Atr3g1,data=data,ax=ax[2],orient='v')
sns.boxplot(x=Atr4g1,data=data,ax=ax[3],orient='v')
sns.boxplot(x=Atr5g1,data=data,ax=ax[4],orient='v')
sns.boxplot(x=Atr6g1,data=data,ax=ax[5],orient='v')
sns.boxplot(x=Atr7g1,data=data,ax=ax[6],orient='v')
sns.boxplot(x=Atr8g1,data=data,ax=ax[7],orient='v')
sns.boxplot(x=Atr9g1,data=data,ax=ax[8],orient='v')
data=t1data
#EDA 2: Skewness check
Atr1g1_skew=round(stats.skew(data[Atr1g1]),4)
Atr2g1_skew=round(stats.skew(data[Atr2g1]),4)
Atr3g1_skew=round(stats.skew(data[Atr3g1]),4)
Atr4g1_skew=round(stats.skew(data[Atr4g1]),4)
Atr5g1_skew=round(stats.skew(data[Atr5g1]),4)
Atr6g1_skew=round(stats.skew(data[Atr6g1]),4)
Atr7g1_skew=round(stats.skew(data[Atr7g1]),4)
Atr8g1_skew=round(stats.skew(data[Atr8g1]),4)
Atr9g1_skew=round(stats.skew(data[Atr9g1]),4)

print(' The skewness of',Atr1g1,'is', Atr1g1_skew)
print(' The skewness of',Atr2g1,'is', Atr2g1_skew)
print(' The skewness of',Atr3g1,'is', Atr3g1_skew)
print(' The skewness of',Atr4g1,'is', Atr4g1_skew)
print(' The skewness of',Atr5g1,'is', Atr5g1_skew)
print(' The skewness of',Atr6g1,'is', Atr6g1_skew)
print(' The skewness of',Atr7g1,'is', Atr7g1_skew)
print(' The skewness of',Atr8g1,'is', Atr8g1_skew)
print(' The skewness of',Atr9g1,'is', Atr9g1_skew)
##EDA 3: Spread
data=t1data
fig, ax = plt.subplots(1,9,figsize=(16,8)) 
sns.distplot(data[Atr1g1],ax=ax[0]) 
sns.distplot(data[Atr2g1],ax=ax[1]) 
sns.distplot(data[Atr3g1],ax=ax[2])
sns.distplot(data[Atr4g1],ax=ax[3])
sns.distplot(data[Atr5g1],ax=ax[4])
sns.distplot(data[Atr6g1],ax=ax[5])
sns.distplot(data[Atr7g1],ax=ax[6])
sns.distplot(data[Atr8g1],ax=ax[7])
sns.distplot(data[Atr9g1],ax=ax[8])
# Step 2.6: Lets visually understand if there is any correlation between the independent variables. 
usecols =[i for i in t1data.columns if i != 'strength']
sns.pairplot(t1data,diag_kind='kde');
# Step 2.7: lets evaluate correlation between different attributes.
# The dependent attribute strength has been ignored from the correlation heatmap. 
#The reason for the same will be explained in the next section.
corr=t1data.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr,annot=True,linewidth=0.05,ax=ax, fmt= '.2f');
### Analyzing Dependent variable (Strength) vs Independent variable (cement, age and water)
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(y="strength", x="cement", hue="water", size="age", data=t1data, ax=ax, sizes=(50, 300),
                palette='RdYlGn', alpha=0.9)
ax.set_title("Strength vs Cement, Age, Water")
ax.legend()
plt.show()
### Analyzing Dependent variable (Strength) vs Independent variable (FineAgg, Ash, Superplastic)
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(y="strength", x="fineagg", hue="ash", size="superplastic", data=t1data, ax=ax, sizes=(50, 300),
                palette='RdYlBu', alpha=0.9)
ax.set_title("Strength vs FineAgg, Ash, Superplastic")
ax.legend(loc="upper left", bbox_to_anchor=(1,1)) # Moved outside the chart so it doesn't cover any data
plt.show()
# before proceeding further, lets first create a copy of the data-set.
# we will use simple imputer and strategy as median for addressing outliars
t2data = t1data.copy()
#EDA 2: Outliar Detection leveraging Box Plot
data=t2data
fig, ax = plt.subplots(1,9,figsize=(38,16)) 
sns.boxplot(x=Atr1g1,data=data,ax=ax[0],orient='v') 
sns.boxplot(x=Atr2g1,data=data,ax=ax[1],orient='v')
sns.boxplot(x=Atr3g1,data=data,ax=ax[2],orient='v')
sns.boxplot(x=Atr4g1,data=data,ax=ax[3],orient='v')
sns.boxplot(x=Atr5g1,data=data,ax=ax[4],orient='v')
sns.boxplot(x=Atr6g1,data=data,ax=ax[5],orient='v')
sns.boxplot(x=Atr7g1,data=data,ax=ax[6],orient='v')
sns.boxplot(x=Atr8g1,data=data,ax=ax[7],orient='v')
sns.boxplot(x=Atr9g1,data=data,ax=ax[8],orient='v')
def outliar_detection(col):
    Q1=t2data[col].quantile(0.25)
    Q3=t2data[col].quantile(0.75)
    IQR=Q3-Q1
    Lower_Whisker = Q1-1.5*IQR
    Upper_Whisker = Q3+1.5*IQR
    t2data[col][t2data[col]> Upper_Whisker] = np.nan
    t2data[col][t2data[col]< Lower_Whisker] = np.nan
    return t2data[col][t2data[col].isnull()]
for i in usecols:
    outliar_detection(i)
t2data.info()
# Imputing the missing values with median
columns=t2data.columns
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

imp_median.fit_transform(t2data)
# imp_median.fit(t2data)
t2data=pd.DataFrame(imp_median.transform(t2data))
t2data.columns=columns
data=t2data
fig, ax = plt.subplots(1,9,figsize=(38,16)) 
sns.boxplot(x=Atr1g1,data=data,ax=ax[0],orient='v') 
sns.boxplot(x=Atr2g1,data=data,ax=ax[1],orient='v')
sns.boxplot(x=Atr3g1,data=data,ax=ax[2],orient='v')
sns.boxplot(x=Atr4g1,data=data,ax=ax[3],orient='v')
sns.boxplot(x=Atr5g1,data=data,ax=ax[4],orient='v')
sns.boxplot(x=Atr6g1,data=data,ax=ax[5],orient='v')
sns.boxplot(x=Atr7g1,data=data,ax=ax[6],orient='v')
sns.boxplot(x=Atr8g1,data=data,ax=ax[7],orient='v')
sns.boxplot(x=Atr9g1,data=data,ax=ax[8],orient='v')
t2data['w/c ratio']=t2data['water']/t2data['cement']
t2data.head()
# It is always better to make a copy of the data before applying any transformation on data
t3data=t2data.copy()
t3data_scaled=t3data.apply(zscore)
X_scaled=t3data_scaled.drop('strength',axis=1)
covMatrix = np.cov(X_scaled,rowvar=False)
# choosing PCA components to be 8 and fitting it on the scaled data. 
#The count of 8 has been selected randomly to check the variance explained by 8 components; 
#We will finalize the components basis the count of components required to explain 95% variance
pca = PCA(n_components=8)
pca.fit(X_scaled)
#Computing the eigen Values
print(pca.explained_variance_)
#Lets compute the eigen Vectors
print(pca.components_)
print(pca.explained_variance_ratio_)
plt.bar(list(range(1,9)),pca.explained_variance_ratio_,alpha=0.5, align='center')
plt.ylabel('Variation explained')
plt.xlabel('eigen Value')
plt.show()
plt.step(list(range(1,9)),np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Cum of variation explained')
plt.xlabel('eigen Value')
plt.show()
# cumulating explained variance ratio to identify how many principal components are required to explain 95% of the variance
cum_var_exp = np.cumsum(pca.explained_variance_ratio_)
# print("Cumulative Variance Explained", cum_var_exp)
pd.DataFrame(cum_var_exp,columns=['Cumul Variance Explanation'],index=['1','2','3','4','5','6','7','8'])
# 6 components explains over 95% of the variance. Hence we will take 6 components
pca6 = PCA(n_components=6)
pca6.fit(X_scaled)
print(pca6.components_)
print(pca6.explained_variance_ratio_)
Xpca6 = pca6.transform(X_scaled)
Y = t3data_scaled['strength']
X_train_pca, X_test_pca, y_train_pca, y_test_pca=train_test_split(Xpca6,Y,test_size=0.30,random_state=1)
# lets check split of data

print("{0:0.2f}% data is in training set".format((len(X_train_pca)/len(t3data.index)) * 100))
print("{0:0.2f}% data is in test set".format((len(X_test_pca)/len(t3data.index)) * 100))
kdata=t2data.copy()
# expect 3 to four clusters from the pair plot visual inspection hence restricting from 2 to 5

cluster_range = range( 2, 6 )
cluster_errors = []
for num_clusters in cluster_range:
  clusters = KMeans( num_clusters, n_init = 5)
  clusters.fit(kdata)
  labels = clusters.labels_
  centroids = clusters.cluster_centers_
  cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
clusters_df[0:15]
# Elbow plot to ascertain the number of clusters

plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
# The elbow plot confirms our visual analysis that there are likely 3 good clusters
kdata_z = kdata.apply(zscore)

cluster = KMeans( n_clusters = 3, random_state = 1 )
cluster.fit(kdata_z)

prediction=cluster.predict(kdata_z)
kdata_z["GROUP"] = prediction     # Creating a new column "GROUP" which will hold the cluster id of each record

kdata_z_copy = kdata_z.copy(deep = True)  # Creating a mirror copy for later re-use instead of building repeatedly
centroids = cluster.cluster_centers_
centroids
centroid_df = pd.DataFrame(centroids, columns = list(kdata) )
centroid_df
kdata_z.boxplot(by = 'GROUP',figsize = (40,18), layout = (2,15));
### strength Vs cement

var = 'cement'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'strength',data=kdata_z,hue='GROUP')
plot.set(ylim = (-3,3));
# strength Vs water

var = 'water'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'strength',data=kdata_z,hue='GROUP')
plot.set(ylim = (-3,3));
# strength Vs fineagg

var = 'fineagg'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'strength',data=kdata_z,hue='GROUP')
plot.set(ylim = (-3,3))
# strength Vs slag

var = 'slag'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'strength',data=kdata_z,hue='GROUP')
plot.set(ylim = (-3,3));
# strength Vs ash

var = 'ash'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'strength',data=kdata_z,hue='GROUP')
plot.set(ylim = (-3,3));
# strength Vs Superplasticizer

var = 'superplastic'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'strength',data=kdata_z,hue='GROUP')
plot.set(ylim = (-3,3));
# strength Vs Coarse Aggregate

var = 'coarseagg'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'strength',data=kdata_z,hue='GROUP')
plot.set(ylim = (-3,3));
# strength Vs age

var = 'age'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'strength',data=kdata_z,hue='GROUP')
plot.set(ylim = (-3,3));
# lets build our regression model
# before proceeding further we'll scale the data so that we can analyse them further
# Linear models are not impacted by scaling; however when we use regularization models like ridge and lasso; they are impacted by scaling. 
# Hence, to be on safe side lets scaling the data; since we might use regulaization of the data.

t2data_scaled=t2data.apply(zscore)
X=t2data_scaled.drop('strength',axis=1)
y = t2data_scaled['strength']

# splitting the data into train and test
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.30,random_state=1)
# lets check split of data
print("{0:0.2f}% data is in training set".format((len(X_train)/len(t2data.index)) * 100))
print("{0:0.2f}% data is in test set".format((len(X_test)/len(t2data.index)) * 100))
### Building the model with all the attributes
# Fit the model on train data
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
# Let us explore the coefficients for each of the independent attributes

for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[idx]))
intercept = regression_model.intercept_

print("The intercept for our model is {}".format(regression_model.intercept_))
regression_model.score(X_train, y_train)
score_LR= regression_model.score(X_test, y_test)
score_LR
#### Building the model with reduced dimensionality (PCA)
regression_model_pca = LinearRegression()
regression_model_pca.fit(X_train_pca, y_train_pca)
regression_model_pca.coef_
intercept_pca = regression_model_pca.intercept_

print("The intercept for our model is {}".format(regression_model_pca.intercept_))
y_predict_LR_pca = regression_model_pca.predict(X_test_pca)
regression_model_pca.score(X_train_pca, y_train_pca)
score_LR_PCA = regression_model_pca.score(X_test_pca, y_test_pca)
score_LR_PCA
### Building the model with all the attributes
clf = svm.SVR()
clf.fit(X_train, y_train)
y_predict_SVR = clf.predict(X_test)
clf.score(X_train, y_train)
score_SVR = clf.score(X_test, y_test)
score_SVR
#### Building the model with reduced dimensionality (PCA)
clf_pca = svm.SVR() 
clf_pca.fit(X_train_pca, y_train_pca)
clf_pca.score(X_train_pca, y_train_pca)
score_SVR_PCA=clf_pca.score(X_test_pca, y_test_pca)
score_SVR_PCA
### Building the model with all the attributes
ridge = Ridge(alpha=0.3)
ridge.fit(X_train,y_train)
print("Ridge model:",ridge.coef_)
ridge.score(X_train,y_train)
score_ridge = ridge.score(X_test,y_test)
score_ridge
#### Building the model with reduced dimensionality (PCA)
ridge_pca = Ridge(alpha=0.3)
ridge_pca.fit(X_train_pca,y_train_pca)
print("Ridge model:",ridge.coef_)
ridge_pca.score(X_train_pca,y_train_pca)
score_ridge_PCA = ridge_pca.score(X_test_pca,y_test_pca)
score_ridge_PCA
### Building the model with all the attributes
lasso=Lasso(alpha=0.1)
lasso.fit(X_train,y_train)
print("Lasso Model",lasso.coef_)
lasso.score(X_train,y_train)
score_lasso = lasso.score(X_test,y_test)
score_lasso
#### Building the model with reduced dimensionality (PCA)
lasso_pca=Lasso(alpha=0.1)
lasso_pca.fit(X_train_pca,y_train_pca)
print("Lasso Model",lasso.coef_)
lasso_pca.score(X_train_pca,y_train_pca)
score_lasso_PCA=lasso_pca.score(X_test_pca, y_test_pca)
score_lasso_PCA
### Building the model with all the attributes
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_train3 = poly.fit_transform(X_train)
X_test3 = poly.fit_transform(X_test)

poly_clf = linear_model.LinearRegression()

poly_clf.fit(X_train3, y_train)

y_pred = poly_clf.predict(X_test3)

#print(y_pred)

#In sample (training) R^2 will always improve with the number of variables!
print(poly_clf.score(X_train3, y_train))
score_LR_poly = poly_clf.score(X_test3, y_test)
score_LR_poly
#### Building the model with reduced dimensionality (PCA)
poly_pca = PolynomialFeatures(degree=4, interaction_only=True)
X_train_poly = poly_pca.fit_transform(X_train_pca)
X_test_poly = poly_pca.fit_transform(X_test_pca)

poly_clf_pca = linear_model.LinearRegression()

poly_clf_pca.fit(X_train_poly, y_train_pca)

y_pred = poly_clf_pca.predict(X_test_poly)

#print(y_pred)

#In sample (training) R^2 will always improve with the number of variables!
print(poly_clf_pca.score(X_train_poly, y_train_pca))
score_LR_poly_PCA = poly_clf_pca.score(X_test_poly, y_test_pca)
score_LR_poly_PCA
### Building the model with all the attributes. We will also compute feature importance
regressor = DecisionTreeRegressor(random_state=1,max_depth=5)
regressor.fit(X_train, y_train)
feature_importances = regressor.feature_importances_
feature_names=X_train.columns
summary = {'Features' : feature_names,'Feature Importance' : feature_importances
          }
Feature_Importance_df = pd.DataFrame(summary)
print('The feature importance is:','\n')
Feature_Importance_df
y_pred_DTR = regressor.predict(X_test)
score_DTR= regressor.score(X_test, y_test)
score_DTR
#### Building the model with reduced dimensionality (PCA)
regressor_pca = DecisionTreeRegressor(random_state=1,max_depth=5)
regressor_pca.fit(X_train_pca, y_train_pca)
y_pred_dtr_pca = regressor_pca.predict(X_test_pca)
score_DTR_PCA = regressor_pca.score(X_test_pca, y_test_pca)
score_DTR_PCA
### Building the model with all the attributes
model_rf = RandomForestRegressor() 
# n_estimators = 50,random_state=1,max_features=3
model_rf = model_rf.fit(X_train, y_train)
y_predict_rf = model_rf.predict(X_test)
score_RF = model_rf.score(X_test, y_test)
score_RF
#### Building the model with reduced dimensionality (PCA)
model_rf_pca = RandomForestRegressor() 
# n_estimators = 50,random_state=1,max_features=3
model_rf_pca = model_rf_pca.fit(X_train_pca, y_train_pca)
y_predict_rf_pca = model_rf_pca.predict(X_test_pca)
score_RF_PCA = model_rf_pca.score(X_test_pca, y_test_pca)
score_RF_PCA
### Building the model with all the attributes
bgcl = BaggingRegressor()
#n_estimators=50,random_state=1
bgcl = bgcl.fit(X_train, y_train)
y_predict_bag = bgcl.predict(X_test)
score_bag = bgcl.score(X_test , y_test)
score_bag
#### Building the model with reduced dimensionality (PCA)
bgcl_pca = BaggingRegressor()
#n_estimators=50,random_state=1
bgcl_pca = bgcl_pca.fit(X_train_pca, y_train_pca)
y_predict_bag_pca = bgcl_pca.predict(X_test_pca)
score_bag_PCA = bgcl_pca.score(X_test_pca , y_test_pca)
score_bag_PCA
### Building the model with all the attributes
AdaBC = AdaBoostRegressor()
# n_estimators=50, random_state=1
#abcl = AdaBoostClassifier( n_estimators=50,random_state=1)
AdaBC = AdaBC.fit(X_train, y_train)
y_predict_ada = AdaBC.predict(X_test)
score_AdaBC = AdaBC.score(X_test , y_test)
score_AdaBC
#### Building the model with reduced dimensionality (PCA)
AdaBC_pca = AdaBoostRegressor()
# n_estimators=50, random_state=1
#abcl = AdaBoostClassifier( n_estimators=50,random_state=1)
AdaBC_pca = AdaBC_pca.fit(X_train_pca, y_train_pca)
y_predict_ada_pca = AdaBC_pca.predict(X_test_pca)
score_AdaBC_PCA = AdaBC_pca.score(X_test_pca , y_test_pca)
score_AdaBC_PCA
### Building the model with all the attributes
GraBR = GradientBoostingRegressor()
# n_estimators=50, random_state=1
#abcl = AdaBoostClassifier( n_estimators=50,random_state=1)
GraBR_fit = GraBR.fit(X_train, y_train)
y_predict_GraBR = GraBR.predict(X_test)
## Testing the model on train data
score_GraBR_train = GraBR.score(X_train , y_train)
score_GraBR_train
## Testing the model on the test data
score_GraBR = GraBR.score(X_test , y_test)
score_GraBR
#### Building the model with reduced dimensionality (PCA)
GraBR_pca = GradientBoostingRegressor()
# n_estimators=50, random_state=1
#abcl = AdaBoostClassifier( n_estimators=50,random_state=1)
GraBR_pca = GraBR_pca.fit(X_train_pca, y_train_pca)
y_predict_GraBR_pca = GraBR_pca.predict(X_test_pca)
score_GraBR_PCA = GraBR_pca.score(X_test_pca , y_test_pca)
score_GraBR_PCA
summary = {'Score': [score_LR, score_lasso,score_ridge, score_LR_poly, score_SVR, score_DTR,score_RF,score_bag,score_AdaBC, score_GraBR],

                    'Score for models trained with 6 Principal Components': [score_LR_PCA,score_lasso_PCA,score_ridge_PCA,score_LR_poly_PCA, score_SVR_PCA, score_DTR_PCA, score_RF_PCA, score_bag_PCA, score_AdaBC_PCA, score_GraBR_PCA]

                     }

models=['Linear Regression','Lasso','Ridge','Polynomial Regression','SVR', 'Decision Tree Regressor','Random Forest','Bagging','Ada Boost','Gradient Boost']
sum_df = pd.DataFrame(summary,models)
sum_df
estimator = GradientBoostingRegressor()
estimator.get_params()
estimator=GradientBoostingRegressor()
search_grid={'n_estimators':[100,200,300,400,500,600],'learning_rate':[.001,0.01,.1],'max_depth':[1,2,3,4,5],'subsample':[.5,.75,1],'random_state':[1]}
search=GridSearchCV(estimator=estimator,param_grid=search_grid,scoring='neg_mean_squared_error',n_jobs=1,cv=10)
search.fit(X_train,y_train)
search.best_params_
## Creating the Gradient Boosting Regressor with the best parameters
GraBR = GradientBoostingRegressor(learning_rate= 0.1,max_depth= 3,n_estimators= 600,random_state= 1,subsample= 1)

GraBR_fit = GraBR.fit(X_train, y_train)
y_predict_GraBR = GraBR.predict(X_test)
### Testng on the train data

score_GraBR = GraBR.score(X_train , y_train)
score_GraBR
### Testing on the test data
score_GraBR = GraBR.score(X_test , y_test)
score_GraBR
scores = cross_val_score(GraBR, X, y, cv=10)
CV_score_acc_GraBR = scores.mean()
CV_score_std_GraBR = scores.std()

print(scores)
print("Accuracy: %.3f%% (%.3f%%)" % (CV_score_acc_GraBR*100.0, CV_score_std_GraBR*100.0))
