# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



#Function to turn non numeric values into categories

def categorize_df(df):

    for column,series in df.items():

        if is_string_dtype(series): 

            df[column] = series.astype('category').cat.as_ordered()

    return df



# function to let me know how many numeric and non numeric columns are there

from pandas.api.types import is_string_dtype, is_numeric_dtype



def numeric_count(df):

    non_numericals = 0

    numericals = 0

    for c in df.columns:

        if is_numeric_dtype(df[c].dtype):

            numericals +=1

        else:

            df[c] = pd.to_numeric(df[c],errors='ignore')

            if is_numeric_dtype(df[c].dtype):

                numericals +=1

            else:

                non_numericals +=1

    return print('{} numbers and {} non numbers'.format(numericals,non_numericals))



#function to replace categories for codes in dataframe

# adds 1 as pandas replaces missing values with -1

def replace_cats(df):

    for column in df.columns:

        if df[column].dtype.name == 'category':

            df[column] = df[column].cat.codes +1

    return df



#function to add na_columns when missing values are present

def na_columns(df):

    for column, series in df.items():

        if df[column].isnull().sum():

            df[column+'_na'] = df[column].isnull()

    return df



#function to fill null values with the mean of the column

def mean_df(df): return df.fillna(df.mean())



#function to separate training set from evaluation variable

def x_and_y(df,target):

    y = df[target].values

    X = df.drop(target, axis=1)

    return X, y



# create a forest regressor

from sklearn.ensemble import RandomForestRegressor



m = RandomForestRegressor(n_estimators=80,n_jobs=-1,oob_score=True)



#split a training set

import math as math

def split_df(X,y,split):

    Xt = X[:math.floor(len(X)*split)]

    Xv = X[math.floor(len(X)*split):]

    yt = y[:math.floor(len(y)*split)]

    yv = y[math.floor(len(y)*split):]

    return Xt,Xv,yt,yv



#function to get log root mean square error

def lrmse(predicted,y): return math.sqrt(((np.log(predicted)-np.log(y))**2).mean())



#function to get root mean square error

def rmse(predicted,y): return math.sqrt(((predicted-y)**2).mean())



#calculate and print errors

def print_errors (Xt,Xv,yt,yv):

    errors = []

    errors.append(rmse(m.predict(Xt),yt))

    errors.append(rmse(m.predict(Xv),yv))

    errors.append(m.score(Xt,yt))

    errors.append(m.score(Xv,yv))

    if hasattr(m, 'oob_score_'): errors.append(m.oob_score_)

    print(errors)



def training_pipeline(df):

    df = categorize_df(df)

    df = replace_cats(df)

    df = na_columns(df)

    #df = df.select_dtypes([np.number])

    df = mean_df(df)

    X,y = x_and_y(df,'SalePrice')

    #activate log of y during modeling as the evaluation is based on the lrmse

    #y = np.log(y)

    Xt,Xv,yt,yv = split_df(X,y,0.7)

    m.fit(Xt,yt)

    print_errors(Xt,Xv,yt,yv)

    return X, y, Xt, Xv, yt, yv



def output_pipeline(df,X,output):

    df = categorize_df(df)

    df = replace_cats(df)

    df = na_columns(df)

    #df = df.select_dtypes([np.number])

    df = mean_df(df)

    X, X_test = X.align(df,join='inner',axis=1)

    predictions = m.predict(X_test)

    my_submission = pd.DataFrame({'Id': df.Id, 'SalePrice': predictions})

    # you could use any filename. We choose submission here

    my_submission.to_csv("submission{}.csv".format(output), index=False)



# draw a tree, taken from fastai library

from IPython.display import Image

from sklearn.tree import export_graphviz

def draw_tree(t, df, size=10, ratio=0.6, precision=0):

    """ Draws a representation of a random forest in IPython.



    Parameters:

    -----------

    t: The tree you wish to draw

    df: The data used to train the tree. This is used to get the names of the features.

    """

    s=export_graphviz(t, out_file="simple-tree.dot", feature_names=df.columns, filled=True,

                      special_characters=True, rotate=True, precision=precision)

    !dot -Tpng simple-tree.dot -o simple-tree.png -Gdpi=600



#draw a plot of R2 score for each additional tree added to the forrest

from sklearn import metrics

import matplotlib.pyplot as plt



def plot_r2_evolution(Xv,yv):

    predictions = np.stack([t.predict(Xv) for t in m.estimators_])

    r2 = []

    last_r2 = 0

    right_tree = 0

    for i in range (len(predictions)):

        r2.append(metrics.r2_score(yv, np.mean(predictions[:i+1], axis=0)))

        if right_tree == 0:

            if abs(r2[-1] - last_r2) > 0.00003:

                last_r2 = r2[-1]

            else:

                print("after {} trees, no significant improvement".format(i))

                right_tree = i

    plt.plot(r2)

    #print(r2)
df_train = pd.read_csv('../input/train.csv',low_memory=False)

df_test = pd.read_csv('../input/test.csv',low_memory=False)
df_train = pd.read_csv('../input/train.csv',low_memory=False)

#df_train.head()

# we remove the ID column, we don't need it

Id_column = df_train['Id']

df_train = df_train.drop('Id',axis=1)

#print(df_train.head())

df_train.isnull().sum().sort_values(ascending=False)

df_train.PoolQC.fillna("None", inplace=True)

#df_train.PoolQC.head()

df_train.MiscFeature.fillna("None", inplace=True)

#df_train.MiscFeature.head()

#df_train['Neighborhood'].head(100)

import seaborn as sns

#sns.countplot(df_train.Neighborhood)

None_columns = ['Alley','MasVnrType','MasVnrArea','FireplaceQu','Fence','GarageQual','GarageYrBlt','GarageCond','GarageFinish','GarageType','BsmtFinType2','BsmtExposure','BsmtQual','BsmtFinType1','BsmtCond']

df_train[None_columns] = df_train[None_columns].fillna("None")

df_train['Electrical'].fillna(df_train['Electrical'].mode()[0],inplace=True)

#len(df_train.columns)



# How to replace missing values from column LotFrontage with the mean of LotFrontage based on the Neighborhood: Houses must have similar LotFrontage if they're from the same neighborhood

# # Drop all rows where LotFrontage is null

df_no_lot = df_train.dropna(subset=['LotFrontage'])

# # Dataframe with Neighborhood and the mean for each neighborhood

df_Neighmeans = df_no_lot.groupby('Neighborhood', as_index=False)['LotFrontage'].mean()

# # Make a dictionary out of it

dict = df_Neighmeans.set_index('Neighborhood').T.to_dict('list')

# # The mean is stored in a list. Turn the list into float, since it's only one value

dict = {k:float(dict[k][0]) for k in dict}

# # Replace!

df_train.LotFrontage = df_train.LotFrontage.fillna(df_train.Neighborhood.map(dict))



X_test.columns
print("baseline")

m = RandomForestRegressor(n_jobs=-1)

X, y, Xt, Xv, yt, yv = training_pipeline(df_train)

#output_pipeline(df_test,X,10)

print("tiny tree")

m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)

m.fit(Xt,yt)

print_errors (Xt,Xv,yt,yv)

print("only one tree")

m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)

m.fit(Xt,yt)

print_errors (Xt,Xv,yt,yv)

print("baseline")

m = RandomForestRegressor(n_jobs=-1)

m.fit(Xt,yt)

print_errors (Xt,Xv,yt,yv)

print("50 trees")

m = RandomForestRegressor(n_estimators=50,n_jobs=-1)

m.fit(Xt,yt)

print_errors (Xt,Xv,yt,yv)

print("50 trees, 3 samples per leaf")

m = RandomForestRegressor(n_estimators=50,min_samples_leaf=3,n_jobs=-1)

m.fit(Xt,yt)

print_errors (Xt,Xv,yt,yv)

print("50 trees, 5 samples per leaf")

m = RandomForestRegressor(n_estimators=50,min_samples_leaf=5,n_jobs=-1)

m.fit(Xt,yt)

print_errors (Xt,Xv,yt,yv)

print("80 trees, 3 samples per leaf, 0.5 features")

m = RandomForestRegressor(n_estimators=80,max_features=0.5,min_samples_leaf=3,n_jobs=-1)

X, y, Xt, Xv, yt, yv = training_pipeline(df_train)

output_pipeline(df_test,X,13)

print("50 trees, 5 samples per leaf, sqrt features")

m = RandomForestRegressor(n_estimators=50,max_features="sqrt",min_samples_leaf=3,n_jobs=-1)

X, y, Xt, Xv, yt, yv = training_pipeline(df_train)

print("50 trees, 5 samples per leaf, log2 features")

m = RandomForestRegressor(n_estimators=50,max_features="log2",min_samples_leaf=3,n_jobs=-1)

X, y, Xt, Xv, yt, yv = training_pipeline(df_train)

print("gridsearch result top rank")

m = RandomForestRegressor(bootstrap=False, max_depth=None, max_features=10, min_samples_split= 2, n_estimators=100,n_jobs=-1)

X, y, Xt, Xv, yt, yv = training_pipeline(df_train)

output_pipeline(df_test,X,15)

# from sklearn.model_selection import GridSearchCV

# from time import time



# # build a classifier

# clf = RandomForestRegressor(n_jobs=-1)



# # Utility function to report best scores

# def report(results, n_top=3):

#     for i in range(1, n_top + 1):

#         candidates = np.flatnonzero(results['rank_test_score'] == i)

#         for candidate in candidates:

#             print("Model with rank: {0}".format(i))

#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

#                   results['mean_test_score'][candidate],

#                   results['std_test_score'][candidate]))

#             print("Parameters: {0}".format(results['params'][candidate]))

#             print("")



# # use a full grid over all parameters

# param_grid = {"n_estimators": [10, 20,40,50,80,100],

#               "max_depth": [3, None],

#               "max_features": [1, 3, 10, "sqrt", "log2"],

#               "min_samples_split": [2, 3, 10],

#               "bootstrap": [True, False]}

# # run grid search

# grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)

# start = time()

# grid_search.fit(X, y)





# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."

#       % (time() - start, len(grid_search.cv_results_['params'])))

# report(grid_search.cv_results_)

for i in range(len(df_test.select_dtypes(np.object).columns)):

    print(df_test.select_dtypes(np.object).columns[:i+1])