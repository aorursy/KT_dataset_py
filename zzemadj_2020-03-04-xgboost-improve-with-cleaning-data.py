# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
X_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col='Id')

X_test_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

#delete outliers

X_full = X_full.drop(X_full['LotFrontage'] [X_full['LotFrontage']>200].index)

X_full = X_full.drop(X_full['LotArea'] [X_full['LotArea']>100000].index)

X_full = X_full.drop(X_full['BsmtFinSF1'] [X_full['BsmtFinSF1']>4000].index)

X_full = X_full.drop(X_full['TotalBsmtSF'] [X_full['TotalBsmtSF']>6000].index)

X_full = X_full.drop(X_full['1stFlrSF'] [X_full['1stFlrSF']>4000].index)

X_full = X_full.drop(X_full.GrLivArea [(X_full['GrLivArea']>4000) & (X_full['SalePrice']<300000)].index)

X_full = X_full.drop(X_full.LowQualFinSF    [X_full['LowQualFinSF']>550].index)

#Categorical values fill none

cat_cols_fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',

                     'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType',

                     'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond',

                     'MasVnrType']

for cat in cat_cols_fill_none:

    X_full[cat] = X_full[cat].fillna("None")

    

#numerical cols with high correlation delete

attributes_drop = ['MiscVal', 'MSSubClass', 'MoSold', 'YrSold', 

                   'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd'] # high corr with other attributes



X_full = X_full.drop(attributes_drop, axis=1)

X_full
y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='mean')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])


from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.neighbors import KNeighborsClassifier #k ближайших соседей

from sklearn.linear_model import Lasso



model_1 = RandomForestRegressor(n_estimators=100, random_state=0)

model_2 = XGBRegressor(n_estimators=1000,learning_rate=0.04, n_jobs=4, random_state=0)

model_3 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

           metric_params=None, n_neighbors=5, p=2, weights='uniform')

model_4 = Lasso(alpha=0.0005, random_state=5)

from sklearn.metrics import mean_absolute_error

from numpy import log as log

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model_4)

                             ])

my_pipeline.fit(X_train, log(y_train))

preds = my_pipeline.predict(X_valid)

# Evaluate the model

score = mean_absolute_error(y_valid, np.e**preds)

print('MAE:', score)

print('r2: ', my_pipeline.score(X_valid,log(y_valid)))
my_pipeline.fit(X_full[my_cols], log(y))

preds = my_pipeline.predict(X_test_full[my_cols])
#lasso with log is better

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': np.e**preds})

output.to_csv('submission.csv', index=False)
output.head()
#тест использования gridsearchCV

'''from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



# the models that you want to compare

models = {

    'RandomForestRegressor': RandomForestRegressor(),

#    'KNeighboursClassifier': KNeighborsClassifier(),

#    'LogisticRegression': LogisticRegression()

}



# the optimisation parameters for each of the above models

parameters = dict(model__n_estimators=[10, 30, 100,200],

                    preprocessor__categorical_transformer__strategy=['mean', 'median', 'most_frequent'] 

)   

CV = GridSearchCV(my_pipeline, parameters, scoring = 'neg_mean_absolute_error', n_jobs= 1)

CV.fit(X_train, y_train)   



print(CV.best_params_)    

print(CV.best_score_)    '''





'''from sklearn.model_selection import GridSearchCV



def fit(train_features, train_actuals):

        """

        fits the list of models to the training data, thereby obtaining in each 

        case an evaluation score after GridSearchCV cross-validation

        """

        for name in models.keys():

            est = models[name]

            est_params = params[name]

            gscv = GridSearchCV(my_pipeline, param_grid=est_params, cv=5)

            gscv.fit(train_features, train_actuals)

            print("best parameters are: {}".format(gscv.best_estimator_))

fit(X_train,y_train)'''