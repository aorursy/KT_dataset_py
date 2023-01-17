# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.rc('font', size=12) 
plt.rc('figure', figsize = (12, 5))

# Settings for the visualizations
import seaborn as sns
print(sns.__version__)
assert sns.__version__ >= "0.10"
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2,'font.family': [u'times']})

import pandas as pd
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 50)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
train_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/train_set.csv',index_col=0) 
test_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/test_set.csv',index_col=0) 
# print the dataset size
print("There is", train_set.shape[0], "samples")
print("Each sample has", train_set.shape[1], "features")
# print the top elements from the dataset
train_set.head()
# As it can be seen the database contains several features, some of them numerical and some of them are categorical.
# It is important to check each of the to understand it.
# Check each features. If the features are numerical it is imporant to check the distributions a.
#                      If the features are categorical it is important to check the number of the categories and the disitribution
train_set.dtypes
# print those categorical features
train_set.select_dtypes(include=['object']).head()
# We can check how many different type there is in the dataset using the folliwing line
train_set["Type"].value_counts()
sns.countplot(y="Type", data=train_set, color="c")
sns.distplot(train_set["Price"])
plt.show()
## the features

features = ['Rooms','Landsize', 'BuildingArea', 'YearBuilt']
## DEFINE YOUR FEATURES
X = train_set[features].fillna(0)
y = train_set[['Price']]

## the model
# KNeighborsRegressor
from sklearn import neighbors
n_neighbors = 3 # you can modify this paramenter (ONLY THIS ONE!!!)
model = neighbors.KNeighborsRegressor(n_neighbors)

## fit the model
model.fit(X, y)

## predict training set
y_pred = model.predict(X)

## Evaluate the model and plot it
from sklearn.metrics import mean_squared_error, r2_score
print("----- EVALUATION ON TRAIN SET ------")
print("RMSE",np.sqrt(mean_squared_error(y, y_pred)))
print("R^2: ",r2_score(y, y_pred))


plt.scatter(y, y_pred)
plt.xlabel('Price')
plt.ylabel('Predicted price');
plt.show()

## predict the test set and generate the submission file
X_test = test_set[features].fillna(0)
y_pred = model.predict(X_test)

df_output = pd.DataFrame(y_pred)
df_output = df_output.reset_index()
df_output.columns = ['index','Price']

df_output.to_csv('baseline.csv',index=False)
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import neighbors
from sklearn.preprocessing import OneHotEncoder

def extract_data(features):
    
    
    # Retrieve train and test splits based on selected features
    X_train = train_set[features]
    X_test = test_set[features]
    
    Y_train = train_set[['Price']].fillna(train_set[['Price']].mean())
    
    
    # Process data to improve performance
    # ------------------------------------
    columns = X_train.select_dtypes(include=['object']).columns
    # Replace NaN categorical data with the most common data
    for col in columns:
        X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
        X_test[col] = X_test[col].fillna(X_test[col].mode()[0])
    
    
    columns = set(X_train.columns)-set(X_train.select_dtypes(include=['object']).columns)
    #columns.remove('Price')
    # Replace NaN non-categorical data with the mean
    for col in columns:
        X_train[col] = X_train[col].fillna(X_train[col].median())
        X_test[col] = X_test[col].fillna(X_test[col].mean())
        
        
    
    # Categorical data must be encoded.
    # I choose the simplest encode which consist assigning a label to each category.
    encoder = LabelEncoder()
    
    # creating instance of one-hot-encoder
    #encoder = OneHotEncoder(handle_unknown='ignore')
    columns = X_train.select_dtypes(include=['object']).columns
    
    for col in columns:
        # Set is useful to get unique values of a list (skip repeated)
        #unique_values = list(set(list(X_train[col]) + list(X_test[col])))
        #encoder.fit(unique_values)
        #
        # Replace col by encoded col
        X_train[col] = pd.get_dummies(X_train[col])#encoder.transform(X_train[col])
        X_test[col] = pd.get_dummies(X_test[col])#encoder.transform(X_test[col])
    
    # Normalize data
    X_train=(X_train - X_train.mean())/X_train.std()
    X_test=(X_test - X_test.mean())/X_test.std()
    
    
    return (X_train,Y_train),(X_test)
from sklearn.model_selection import GridSearchCV

def train_model(X_train,X_test,Y_train,l=None,n_neighbors=None):
    #Try Linear Regression
    if l: model = LinearRegression()
    else: 
        # or KNN regressiom
        model = neighbors.KNeighborsRegressor()
        # Grid Search is going to find the best n_neigh for our model
        param_grid = {'n_neighbors': np.arange(1,25)}
        search = GridSearchCV(model,param_grid,cv=5)
        search.fit(X_train, Y_train)
        print(str(search.best_params_))
        model = neighbors.KNeighborsRegressor(n_neighbors=search.best_params_['n_neighbors'])
    
    ## fit the model
    model.fit(X_train, Y_train)
    
    ## predict training set
    y_pred_train = model.predict(X_train)
    
    
    ## Evaluate the model
    rmse = np.sqrt(mean_squared_error(Y_train, y_pred_train))
    r2 = r2_score(Y_train, y_pred_train)
    
    y_pred_test = model.predict(X_test)
    
    # Write Result
    df_output = pd.DataFrame(y_pred_test)
    df_output = df_output.reset_index()
    df_output.columns = ['index','Price']

    # Return data using rmse as "key" for select min
    return rmse,r2,search.best_params_,(Y_train,y_pred_train),df_output
from time import time
from joblib import Parallel, delayed

def execute(features,l=None,n_neighbors=None):
    (X_train,Y_train),(X_test) = extract_data(features)
    return train_model(X_train,X_test,Y_train,l=l,n_neighbors=n_neighbors)

def select_best_model(l=None,n_neighbors=None):
    ## All the features
    features = ['Suburb','Address','Rooms','Type','Method',
                'SellerG','Date','Distance','Postcode','Bedroom2',
                'Bathroom','Car','Landsize', 'BuildingArea', 'YearBuilt',
                'CouncilArea','Lattitude','Longtitude',
                'Regionname','Propertycount']
    
    # Create all possible combinations of feature's slices
    all_features = [features[i:j] for i in range(len(features)) for j in range(len(features)) if len(features[i:j]) > 0]
    
    # Compute model with every combination and choose the one with smaller rmse
    # We use joblib to parallelize
    rmse,r2,best_params,(Y_train,y_pred),df_output = min(Parallel(n_jobs=-1)(delayed(execute)(feature,l,n_neighbors) for feature in all_features))

    # Plot info about the best result
    plt.scatter(Y_train, y_pred)
    plt.xlabel('Price')
    plt.ylabel('Predicted price');
    plt.show()
    
    print("RMSE: "+str(rmse))
    print("R2 Score: "+str(r2))
    print("Params: "+str(best_params))
    
    # Save result
    df_output.to_csv('submit_improved.csv',index=False)
    


select_best_model(False,None)
