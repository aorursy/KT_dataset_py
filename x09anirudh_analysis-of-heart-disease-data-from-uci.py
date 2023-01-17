# 1.0 Clear memory

%reset -f



# 1.1 Call data manipulation libraries

import pandas as pd

import numpy as np



# 1.2 Feature creation libraries

from sklearn.random_projection import SparseRandomProjection as sr  # Projection features

from sklearn.cluster import KMeans                    # Cluster features

from sklearn.preprocessing import PolynomialFeatures  # Interaction features



# 1.3 For feature selection

# Ref: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_classif  # Selection criteria



# 1.4 Data processing

# 1.4.1 Scaling data in various manner

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale

# 1.4.2 Transform categorical (integer) to dummy

from sklearn.preprocessing import OneHotEncoder



# 1.5 Splitting data

from sklearn.model_selection import train_test_split



# 1.6 Decision tree modeling

# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree

# http://scikit-learn.org/stable/modules/tree.html#tree

from sklearn.tree import  DecisionTreeClassifier as dt



# 1.7 RandomForest modeling

from sklearn.ensemble import RandomForestClassifier as rf



# 1.8 Plotting libraries to plot feature importance

import matplotlib.pyplot as plt

import seaborn as sns



# 1.9 Misc

import os, time, gc





# 2.1 Read train/test files

heart = pd.read_csv("../input/heart.csv")



heart.head(5)

heart.shape         ## 303 x 14
# 2.2 Split into Test and Training Data

X_train, X_test, y_train, y_test = train_test_split(

        heart.drop('target', 1), 

        heart['target'], 

        test_size = 0.3, 

        random_state=10

        ) 
# 2.3 Look at data

X_train.head(2)

X_train.shape                        # 212 x 13

X_test.shape                         # 91 x 13



y_test.shape                        # 91 x 

y_train.shape                       # 212 x



# Data types

X_train.dtypes.value_counts()   # All afeatures re integers 

# 2.4 Target classes are almost balanced

heart.target.value_counts()

# 3 Check if there are Missing values? None

X_train.isnull().sum().sum()  # 0

X_test.isnull().sum().sum()   # 0
## i)   Shooting in dark. These features may help or may not help

## ii)  There is no exact science as to which features will help



#  4. Feature 1: Row sums of features 1:13. More successful

#                when data is binary.



X_train['sum'] = X_train.sum(numeric_only = True, axis=1)  # numeric_only= None is default

X_test['sum'] = X_test.sum(numeric_only = True,axis=1)
# 4.1 Assume that value of '0' in a cell implies missing feature

#     Transform train and test dataframes

#     replacing '0' with NaN

#     Use pd.replace()

tmp_train = X_train.replace(0, np.nan)

tmp_test = X_test.replace(0,np.nan)
# 4.2 Check if tmp_train is same as train or is a view

#     of train? That is check if tmp_train is a deep-copy



tmp_train is X_train                # False

tmp_train._is_view                # False
# 4.3 Check if 0 has been replaced by NaN

tmp_train.head(1)

tmp_test.head(1)

# 5. Feature 2 : For every row, how many features exist

#                that is are non-zero/not NaN.

#                Use pd.notna()

tmp_train.notna().head(1)

X_train["count_not0"] = tmp_train.notna().sum(axis = 1)

X_test['count_not0'] = tmp_test.notna().sum(axis = 1)
# 6. Similary create other statistical features

#    Feature 3

#    Pandas has a number of statistical functions

#    Ref: https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#computations-descriptive-stats



feat = [ "var", "median", "mean", "std", "max", "min"]

for i in feat:

    X_train[i] = tmp_train.aggregate(i,  axis =1)

    X_test[i]  = tmp_test.aggregate(i,axis = 1)

# 7 Delete not needed variables and release memory

del(tmp_train)

del(tmp_test)

gc.collect()
# 7.1 So what do we have finally

X_train.shape                

X_train.head(1)

X_test.shape                 

X_test.head(2)
# 8. Before we proceed further, keep target feature separately

target = y_train

target.tail(2)
# 9. Store column names of our data somewhere

#     We will need these later (at the end of this code)

colNames = X_train.columns.values

colNames



# 10. Random projection is a fast dimensionality reduction feature

#     Also used to look at the structure of data

# 11. Generate features using random projections

#     First stack train and test data, one upon another

tmp = pd.concat([X_train,X_test],

                axis = 0,            # Stack one upon another (rbind)

                ignore_index = True

                )

# 12.1

tmp.shape     # 303 X 21

# 12.2 Transform tmp t0 numpy array

#      Henceforth we will work with array only

tmp = tmp.values

tmp.shape       # 303 X 21

# 13. Let us create 8 random projections/columns

#     This decision, at present, is arbitrary

NUM_OF_COM = 8
# 13.1 Create an instance of class

rp_instance = sr(n_components = NUM_OF_COM)
# 13.2 fit and transform the (original) dataset

#      Random Projections with desired number

#      of components are returned

rp = rp_instance.fit_transform(tmp[:, :13])
# 13.3 Look at some features

rp[: 5, :  3]

# 13.4 Create some column names for these columns

#      We will use them at the end of this code

rp_col_names = ["r" + str(i) for i in range(8)]

rp_col_names

# 14. Before clustering, scale data

# 15.1 Create a StandardScaler instance

se = StandardScaler()

# 15.2 fit() and transform() in one step

tmp = se.fit_transform(tmp)

# 15.3

tmp.shape               # 303 X 21 
# 16. Perform kmeans using 13 features.

#     No of centroids is no of classes in the 'target'

centers = target.nunique()    # 2 unique classes

centers               # 2
# 17.1 Begin clustering

start = time.time()



# 17.2 First create object to perform clustering

kmeans = KMeans(n_clusters=centers, # How many clusters

                n_jobs = 4)         # Parallel jobs for n_init



# 17.3 Next train the model on the original data only

kmeans.fit(tmp[:, : 13])



end = time.time()

(end-start)/60.0      # 0.007 minutes
# 18 Get clusterlabel for each row (data-point)

kmeans.labels_

kmeans.labels_.size   # 303



# 19. Cluster labels are categorical. So convert them to dummy

# 19.1 Create an instance of OneHotEncoder class

ohe = OneHotEncoder(sparse = False)

# 19.2 Use ohe to learn data

#      ohe.fit(kmeans.labels_)

ohe.fit(kmeans.labels_.reshape(-1,1))     # reshape(-1,1) recommended by fit()

                                          # '-1' is a placeholder for actual

# 19.3 Transform data now

dummy_clusterlabels = ohe.transform(kmeans.labels_.reshape(-1,1))

dummy_clusterlabels

dummy_clusterlabels.shape    # 303 X 2 (as many as there are classes)



# 19.4 We will use the following as names of new two columns

#      We need them at the end of this code



k_means_names = ["k" + str(i) for i in range(2)]

k_means_names



# 21. Will require lots of memory if we take large number of features

#     Best strategy is to consider only impt features



degree = 2

poly = PolynomialFeatures(degree,                 # Degree 2

                          interaction_only=True,  # Avoid e.g. square(a)

                          include_bias = False    # No constant term

                          )



# 21.1 Consider only first 8 features

#      fit and transform

df =  poly.fit_transform(tmp[:, : 8])





df.shape     # 303 X 36

# 21.2 Generate some names for these 36 columns

poly_names = [ "poly" + str(i)  for i in range(36)]

poly_names







# 22 Append now all generated features together

# 22 Append random projections, kmeans and polynomial features to tmp array



tmp.shape          # 303 X 21

#  22.1 If variable, 'dummy_clusterlabels', exists, stack kmeans generated

#       columns also else not. 'vars()'' is an inbuilt function in python.

#       All python variables are contained in vars().



if ('dummy_clusterlabels' in vars()):               #

    tmp = np.hstack([tmp,rp,dummy_clusterlabels, df])

else:

    tmp = np.hstack([tmp,rp, df])       # No kmeans      <==





tmp.shape          # 303 X 67

# 22.1 Combine train and test into X and y to split compatible datasets

X = tmp

X.shape        # 303 X 67

# 22.2 Combine y_train and y_test into y to split into compatible datasets later

y = pd.concat([y_train,y_test],

                axis = 0,            # Stack one upon another (rbind)

                ignore_index = True

                )

y.shape        # 303,
# 22.3 Delete tmp - as a good programming practice

del tmp

gc.collect()

# 23. Split the feature engineered data into new training and test dataset

X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    y,

                                                    test_size = 0.3)
# 23.1

X_train.shape    # 212 X 67 

X_test.shape     # 91 X 67
# 24 Decision tree classification

# 24.1 Create an instance of class

clf1_dt = dt(min_samples_split = 5,

         min_samples_leaf= 3

        )
start = time.time()

# 24.2 Fit/train the object on training data

#      Build model

clf1_dt = clf1_dt.fit(X_train, y_train)

end = time.time()

(end-start)/60                     # << 1 minute

# 24.3 Use model to make predictions

classes1_dt = clf1_dt.predict(X_test)
# 24.4 Check accuracy

(classes1_dt == y_test).sum()/y_test.size      
# 25. Instantiate RandomForest classifier

clf1_rf = rf(n_estimators=50)

# 25.1 Fit/train the object on training data

#      Build model



start = time.time()

clf1_rf = clf1_rf.fit(X_train, y_train)

end = time.time()

(end-start)/60                    # 0.001 min

# 25.2 Use model to make predictions

classes1_rf = clf1_rf.predict(X_test)
# 25.3 Check accuracy

(classes1_rf == y_test).sum()/y_test.size      
# 26. Get feature importance

clf1_rf.feature_importances_        # Column-wise feature importance

clf1_rf.feature_importances_.size   # 67
# 26.1 To our list of column names, append all other col names

#      generated by random projection, kmeans (onehotencoding)

#      and polynomial features

#      But first check if kmeans was used to generate features



if ('dummy_clusterlabels' in vars()):       # If dummy_clusterlabels labels are defined

    colNames = list(colNames) + rp_col_names+ k_means_names + poly_names

else:

    colNames = colNames = list(colNames) + rp_col_names +  poly_names      # No kmeans      <==
# 26.1.1 So how many columns?

len(colNames)           # 67 with kmeans 
# 26.2 Create a dataframe of feature importance and corresponding

#      column names. Sort dataframe by importance of feature

feat_imp = pd.DataFrame({

                   "importance": clf1_rf.feature_importances_ ,

                   "featureNames" : colNames

                  }

                 ).sort_values(by = "importance", ascending=False)
feat_imp.shape                   # 67 X 2 

feat_imp.head(13)
# 26.3 Plot feature importance for first 20 features

g = sns.barplot(x = feat_imp.iloc[  : 20 ,  1] , y = feat_imp.iloc[ : 20, 0])

g.set_xticklabels(g.get_xticklabels(),rotation=90)



# 27 Select top 13 columns and get their indexes

#      Note that in the selected list few kmeans

#      columns also exist

newindex = feat_imp.index.values[:13]

newindex
# 28 Use these top 13 columns for classification

# 28.1  Create DTree classifier object

clf2_dt = dt(min_samples_split = 5, min_samples_leaf= 3)
# 28.2 Train the object on data

start = time.time()

clf2_dt = clf2_dt.fit(X_train[: , newindex], y_train)

end = time.time()

(end-start)/60                     # 3.322362899780273e-05 minute
# 28.3  Make prediction

classes2_dt = clf2_dt.predict(X_test[: , newindex])
# 28.4 Accuracy?

(classes2_dt == y_test).sum()/y_test.size 
# 28.5  Create RForest classifier object

clf2_rf = rf(n_estimators=50)
# 28.6 Traion the object on data

start = time.time()

clf2_rf = clf2_rf.fit(X_train[: , newindex], y_train)

end = time.time()

(end-start)/60                     # 0.0008496324221293132 minute
# 28.7  Make prediction

classes2_rf = clf2_rf.predict(X_test[: , newindex])

# 28.8 Accuracy?

(classes2_rf == y_test).sum()/y_test.size  

# 29 Select top 20 columns and get their indexes

#      Note that in the selected list few kmeans

#      columns also exist

newindex2 = feat_imp.index.values[:20]

newindex2
# 30.1  Create DTree classifier object

clf3_dt = dt(min_samples_split = 5, min_samples_leaf= 3)

# 30.2 Train the object on data

start = time.time()

clf3_dt = clf3_dt.fit(X_train[: , newindex2], y_train)

end = time.time()

(end-start)/60                     # 0 minute

# 30.3  Make prediction

classes3_dt = clf3_dt.predict(X_test[: , newindex2])

# 30.4 Accuracy?

(classes3_dt == y_test).sum()/y_test.size

# 30.5  Create RForest classifier object

# increasing the number of estimators to 100 from 50...

clf3_rf = rf(n_estimators=100)

# 30.6 Train the object on data

start = time.time()

clf3_rf = clf3_rf.fit(X_train[: , newindex2], y_train)

end = time.time()

(end-start)/60                     # 0.0014 minute

# 30.7  Make prediction

classes3_rf = clf3_rf.predict(X_test[: , newindex2])

# 26.4 Accuracy?

(classes3_rf == y_test).sum()/y_test.size 
