# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#готовим обработаные данные

from sklearn.model_selection import train_test_split

X_full = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

X_test_full = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['Survived'], inplace=True)

X_ish=X_full.copy()

y = X_full.Survived

X_full.drop(['Survived'], axis=1, inplace=True)



# Impute Missing Value of Pclass = 3

X_full.loc[X_full.Fare.isna(),'Fare'] = X_full.Fare[X_full.Pclass==3].mean() 

#change huge fares

X_full.loc[X_full.Fare>300,'Fare'] = 300



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

numerical_cols.remove("Pclass")

custom_cols= ["Pclass"]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols+custom_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
X_full[numerical_cols].info()
X_full[numerical_cols].iloc[:,2]
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



fig = plt.figure(figsize=(12,18))

for i in range(len(numerical_cols)-2):

    fig.add_subplot(9,4,i+1)

    sns.distplot(X_full[numerical_cols].iloc[:,i].dropna())

    plt.xlabel(X_full[numerical_cols].columns[i])



plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(12, 18))



for i in range(len(numerical_cols)):

    fig.add_subplot(9, 4, i+1)

    sns.boxplot(y=X_full[numerical_cols].iloc[:,i])



plt.tight_layout()

plt.show()
#null values in categorical columns

X_full[categorical_cols].isna().sum().sort_values(ascending=False).head(17)
X_test_full.info()
X_full[my_cols].head()

X_full["Pclass"].isnull().values.any()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

import category_encoders as ce



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='median')

cust_transformer =  SimpleImputer(strategy='constant',fill_value=0)



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols),

        ('cust',cust_transformer,custom_cols)

    ])
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier #k ближайших соседей





model_1 = RandomForestRegressor(n_estimators=100, random_state=0)

model_2 = XGBClassifier(criterion='gini',n_estimators=200,learning_rate=0.05, n_jobs=4, random_state=0)

model_3 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

           metric_params=None, n_neighbors=5, p=2, weights='uniform')
from sklearn.metrics import mean_absolute_error

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model_2)

                             ])

my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_valid)

# Evaluate the model

score = mean_absolute_error(y_valid, preds)

print('MAE:', score)

print('r2: ', my_pipeline.score(X_train, y_train))



from sklearn.model_selection import cross_val_score



# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(my_pipeline, X_full[my_cols], y,

                              cv=5,

                              scoring='neg_mean_absolute_error')

print(scores)
preds
#тест использования gridsearchCV

from sklearn.ensemble import RandomForestRegressor

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

parameters = dict(model__n_estimators=[10, 30, 100,200],model__n_jobs= [1,3]

                    #preprocessor__categorical_transformer__strategy=['mean', 'median', 'most_frequent'] 

)   

CV = GridSearchCV(my_pipeline, parameters, scoring = 'neg_mean_absolute_error')

CV.fit(X_train, y_train)   



print(CV.best_params_)    

print(CV.best_score_)    
from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

model_4 = RFC(criterion='gini',random_state=0)

model_5 = SVC(kernel='rbf',C = 100,random_state=0)

model_6 = LogisticRegression(random_state=0,n_jobs=-1,verbose=0)



my_pipeline4 = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model_4)

                             ])

my_pipeline5 = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model_5)

                             ])

my_pipeline6 = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model_6)

                             ])



my_pipeline4.fit(X_train, y_train)

preds4 = my_pipeline4.predict(X_valid)

# Evaluate the model

score4 = mean_absolute_error(y_valid, preds4)

print('MAE:', score4)

print('r2: ', my_pipeline4.score(X_train, y_train))



my_pipeline5.fit(X_train, y_train)

preds5 = my_pipeline5.predict(X_valid)

# Evaluate the model

score5 = mean_absolute_error(y_valid, preds5)

print('MAE:', score5)

print('r2: ', my_pipeline5.score(X_train, y_train))



my_pipeline6.fit(X_train, y_train)

preds6 = my_pipeline6.predict(X_valid)

# Evaluate the model

score6 = mean_absolute_error(y_valid, preds6)

print('MAE:', score6)

print('r2: ', my_pipeline6.score(X_train, y_train))
((preds5+preds4+preds6+preds)/4).astype(int)
CV.fit(X_full[my_cols], y)

preds = CV.predict(X_test_full[my_cols])

#output = pd.DataFrame({'PassengerId': X_test.index,

#                       'Survived': np.absolute(preds.round()).astype(int)})

#output.to_csv('submission.csv', index=False)
my_pipeline4.fit(X_full[my_cols], y)

my_pipeline5.fit(X_full[my_cols], y)

my_pipeline6.fit(X_full[my_cols], y)

preds4=my_pipeline4.predict(X_test_full[my_cols])

preds5=my_pipeline5.predict(X_test_full[my_cols])

preds6=my_pipeline6.predict(X_test_full[my_cols])

output = pd.DataFrame({'PassengerId': X_test.index,

                       'Survived': ((preds5+preds4+preds6+preds)/4).astype(int)})

output.to_csv('submission.csv', index=False)
preds4
#np.absolute(preds.round()).astype(int)

output