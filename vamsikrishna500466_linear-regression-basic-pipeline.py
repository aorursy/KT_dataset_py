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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings; warnings.simplefilter('ignore')



pd.set_option('display.max_rows',5000)

pd.set_option('display.max_columns',5000)

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df=train.copy()
df.head()
# List of variables that contain year information

year_feature = [feature for feature in df if 'Yr' in feature or 'Year' in feature]



year_feature
continous_feature = [feature for feature in df if len(df[feature].unique()) > 80 and feature not in year_feature + ['Id']]



print('Number of train continous feature : ',len(continous_feature))  

continous_feature
train_continous_feature = df[continous_feature]

train_continous_feature.head()
train_categorical_features = df.select_dtypes(exclude=[np.number])

train_categorical_features.head()
train_year_feature = df[year_feature]

train_year_feature.head()
df_numerical_features = df.select_dtypes(include=[np.number])

# Numerical variables are usually of 2 type

# 1. Continous variable and Discrete Variables



discrete_feature = [feature for feature in df_numerical_features if len(df_numerical_features[feature].unique()) < 80 

                                                                     and feature not in year_feature + ['Id']]



print('Number of discrete features : ',len(discrete_feature))  

discrete_feature
train_discrete_feature = df[discrete_feature]

train_discrete_feature.head()
#step1 divide data based on the types of the  data

print('train_categorical_features:',train_categorical_features.shape),

print('train_discrete_feature:',train_discrete_feature.shape),

print('train_continous_feature:',train_continous_feature.shape),

print('train_year_feature:',train_year_feature.shape)
train_continous_feature_nan = train_continous_feature.isnull().sum()

train_continous_feature_nan=train_continous_feature_nan[train_continous_feature_nan>0]



train_discrete_feature_nan = train_discrete_feature.isnull().sum()

train_discrete_feature_nan=train_discrete_feature_nan[train_discrete_feature_nan>0]



train_categorical_features_nan = train_categorical_features.isnull().sum()

train_categorical_features_nan=train_categorical_features_nan[train_categorical_features_nan>0]



train_year_feature_nan = train_year_feature.isnull().sum()

train_year_feature_nan=train_year_feature_nan[train_year_feature_nan>0]



print('train_continous_feature_nan:',

      train_continous_feature_nan.sort_values(ascending = False))



print('train_discrete_feature_nan:',

      train_discrete_feature_nan.sort_values(ascending = False))



print('train_categorical_features_nan:',

      train_categorical_features_nan.sort_values(ascending = False))



print('train_year_feature_nan:',

      train_year_feature_nan.sort_values(ascending = False))
#fill train_continous_feature NAN values with mean

train_continous_feature['LotFrontage']=train_continous_feature['LotFrontage'].fillna(train_continous_feature['LotFrontage'].mean())

train_continous_feature['MasVnrArea']=train_continous_feature['MasVnrArea'].fillna(train_continous_feature['MasVnrArea'].mean())
train_continous_feature.head()
#categorical fetures with more NAN values

train_categorical_features.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1,inplace=True)

train_categorical_features['GarageCond']=train_categorical_features['GarageCond'].fillna(train_categorical_features['GarageCond'].mode()[0])

train_categorical_features['GarageQual']=train_categorical_features['GarageQual'].fillna(train_categorical_features['GarageQual'].mode()[0])

train_categorical_features['GarageFinish']=train_categorical_features['GarageFinish'].fillna(train_categorical_features['GarageFinish'].mode()[0])

train_categorical_features['GarageType']=train_categorical_features['GarageType'].fillna(train_categorical_features['GarageType'].mode()[0])

train_categorical_features['BsmtFinType2']=train_categorical_features['BsmtFinType2'].fillna(train_categorical_features['BsmtFinType2'].mode()[0])

train_categorical_features['BsmtExposure']=train_categorical_features['BsmtExposure'].fillna(train_categorical_features['BsmtExposure'].mode()[0])

train_categorical_features['BsmtFinType1']=train_categorical_features['BsmtFinType1'].fillna(train_categorical_features['BsmtFinType1'].mode()[0])

train_categorical_features['BsmtCond']=train_categorical_features['BsmtCond'].fillna(train_categorical_features['BsmtCond'].mode()[0])

train_categorical_features['BsmtQual']=train_categorical_features['BsmtQual'].fillna(train_categorical_features['BsmtQual'].mode()[0])

train_categorical_features['MasVnrType']=train_categorical_features['MasVnrType'].fillna(train_categorical_features['MasVnrType'].mode()[0])

train_categorical_features['Electrical']=train_categorical_features['Electrical'].fillna(train_categorical_features['Electrical'].mode()[0])
#by using manual check of year data

train_year_feature['GarageYrBlt']=train_year_feature['GarageYrBlt'].fillna(1980)
train_continous_feature_nan = train_continous_feature.isnull().sum()

train_continous_feature_nan=train_continous_feature_nan[train_continous_feature_nan>0]



train_discrete_feature_nan = train_discrete_feature.isnull().sum()

train_discrete_feature_nan=train_discrete_feature_nan[train_discrete_feature_nan>0]



train_categorical_features_nan = train_categorical_features.isnull().sum()

train_categorical_features_nan=train_categorical_features_nan[train_categorical_features_nan>0]



train_year_feature_nan = train_year_feature.isnull().sum()

train_year_feature_nan=train_year_feature_nan[train_year_feature_nan>0]



print('train_continous_feature_nan:',

      train_continous_feature_nan.sort_values(ascending = False))



print('train_discrete_feature_nan:',

      train_discrete_feature_nan.sort_values(ascending = False))



print('train_categorical_features_nan:',

      train_categorical_features_nan.sort_values(ascending = False))



print('train_year_feature_nan:',

      train_year_feature_nan.sort_values(ascending = False))
# Temporal Variables (Date Time Variables)

# Basically we are capturing the difference of years here



for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:

       

    train_year_feature[feature] = train_year_feature['YrSold'] - train_year_feature[feature]

train_year_feature.head()
categorical_features = [feature for feature in train_categorical_features.columns if train_categorical_features[feature].dtype == 'O']

categorical_features
train_categorical_features1 = pd.concat([train_categorical_features,df[['SalePrice']]], axis=1)

train_categorical_features1.head()
for feature in categorical_features:

    temp = train_categorical_features1.groupby(feature)['SalePrice'].count()/len(train_categorical_features1)

    train_categorical_features2 = temp[temp > 0.01].index

    train_categorical_features1[feature] = np.where(train_categorical_features1[feature].isin(train_categorical_features2), train_categorical_features1[feature], 'Rare_Var')
train_categorical_features1.head()
# Let's map the categories to some specific values 

for feature in categorical_features:

    labels_ordered = train_categorical_features1.groupby([feature])['SalePrice'].mean().sort_values().index

    labels_ordered = {k:i for i,k in enumerate(labels_ordered)}

    train_categorical_features1[feature] = train_categorical_features1[feature].map(labels_ordered)
train_categorical_features1.head()
train_categorical_features1.drop(['SalePrice'],axis=1,inplace=True)

train_categorical_features1.head()
final_df = pd.concat([train_year_feature,train_categorical_features1,train_categorical_features1,train_continous_feature], axis=1)

final_df.head()
# Creating X_train and y_train 

X_train = final_df.drop(['SalePrice'], axis = 1)

y_train = final_df['SalePrice']
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import Lasso
model_sel_feature = SelectFromModel(Lasso(alpha = 0.005, random_state = 49))

model_sel_feature.fit(X_train,y_train)
# get_support() will show an array of boolean values i.e. which features are selected and which are not

model_sel_feature.get_support()
selected_feat = X_train.columns[model_sel_feature.get_support()]



# Let's print some stats

print(f"Total Features : {len(X_train.columns)}")

print(f"Features Selected : {len(selected_feat)}")

print(f"features with coefficients shrank to zero: {np.sum(model_sel_feature.estimator_.coef_ == 0)}")
selected_feat
X_train = X_train[selected_feat]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=42, test_size=.33)



from sklearn import linear_model

lr = linear_model.LinearRegression()



model = lr.fit(X_train, y_train)



print ("R^2 is: \n", model.score(X_test, y_test))



predictions = model.predict(X_test)



from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_test, predictions))



actual_values = y_test

plt.scatter(predictions, actual_values, alpha=.7,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')

plt.show()
y=final_df['SalePrice']

X=final_df.drop(['SalePrice'],axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)



from sklearn import linear_model

lr = linear_model.LinearRegression()



model = lr.fit(X_train, y_train)



print ("R^2 is: \n", model.score(X_test, y_test))



predictions = model.predict(X_test)



from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_test, predictions))



actual_values = y_test

plt.scatter(predictions, actual_values, alpha=.7,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

from sklearn import tree

clf = tree.DecisionTreeRegressor()

model = clf.fit(X_train, y_train)



print ("R^2 is: \n", model.score(X_test, y_test))



predictions = model.predict(X_test)



from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_test, predictions))



actual_values = y_test

plt.scatter(predictions, actual_values, alpha=.7,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Decission tree Model')

plt.show()
y=final_df['SalePrice']

X=final_df.drop(['SalePrice'],axis=1)
### Feature Importance



from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt

model = ExtraTreesRegressor()

model.fit(X,y)
print(model.feature_importances_)
#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor()



n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

print(n_estimators)



from sklearn.model_selection import RandomizedSearchCV

#Randomized Search CV



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]
# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}



print(random_grid)


# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
rf_random.best_score_
predictions=rf_random.predict(X_test)
sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)

from sklearn import metrics


print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_test, predictions))



print ("R^2 is: \n", model.score(X_test, y_test))



actual_values = y_test

plt.scatter(predictions, actual_values, alpha=.7,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('random forest  Model')

plt.show()