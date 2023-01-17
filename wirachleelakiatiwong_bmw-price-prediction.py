# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import joblib



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/bmw.csv')

df.head()
df.info()
#Age calculation (present year - year of purchased)

df['age'] = 2020 - df['year']

df = df.drop(columns = 'year')

df.head()
#Since there might be error in data gathered (There is petrol and diesel car that have engine size of 0)

df[df['engineSize'] == 0]
#Let's drop instances which fuelType are Diesel or Petrol but have 0.0 engineSize out.

df = df.drop(df[(df['engineSize'] == 0) & (df['fuelType'].isin(['Diesel','Petrol']))].index)

df[df['engineSize'] == 0]
df.info()
cat_col = df.select_dtypes(include = object).columns.tolist()

num_col = df.select_dtypes(exclude = object).columns.tolist()
df.describe()
#numerical data

fig = plt.figure(figsize=(20,10))

sns.set_style('darkgrid')

for index,col in enumerate(num_col):

    plt.subplot(3,3,index+1)

    sns.set(font_scale = 1.0)

    sns.distplot(df[col], kde = False)

fig.tight_layout(pad=1.0)
#Categorical feature

fig = plt.figure(figsize=(20,5))

sns.set_style('darkgrid')

for index,col in enumerate(cat_col):

    plt.subplot(1,3,index+1)

    if(index == 0):

        plt.xticks(rotation=90)

    sns.set(font_scale = 1.0)

    sns.countplot(df[col], order = df[col].value_counts().index)



    

fig.tight_layout(pad=1.0)
#numerical data

fig = plt.figure(figsize=(20,20))

sns.set_style('darkgrid')

for index,col in enumerate(num_col):

    if col == 'price':

        plt.subplot(4,3,index+1)

        sns.heatmap(df.corr(), annot=True, cmap='RdBu')

    else:

        plt.subplot(4,3,index+1)

        sns.set(font_scale = 1.0)

        sns.scatterplot(data = df, x = col, y = 'price',alpha = 0.7)

fig.tight_layout(pad=1.0)
sns.set(style="ticks")



# Initialize the figure

f, ax = plt.subplots(figsize=(20, 10))



# Plot model vs price

plt.subplot(1,3,1)

sns.boxplot(x="price", y="model", data=df,

            whis=[0, 100], palette="vlag",

           order = df.groupby('model').median().sort_values(by = 'price').index)

ax.xaxis.grid(True)

sns.despine(trim=True, left=True)



# Plot transmission vs price

plt.subplot(1,3,2)

sns.boxplot(x="transmission", y="price", data=df,

            whis=[0, 100], palette="vlag")

ax.xaxis.grid(True)

sns.despine(trim=True, left=True)



# Plot fuelType vs price

plt.subplot(1,3,3)

sns.boxplot(x="fuelType", y="price", data=df,

            whis=[0, 100], palette="vlag")



ax.xaxis.grid(True)

sns.despine(trim=True, left=True)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



from sklearn.linear_model import ElasticNet

from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

import xgboost as xgb
X = df.copy().drop(columns='price')

y = df['price'].copy()

#Before further dealing with data, let's split it into train and test set , since we'll not use test set for model development (for evaluating model performance only).

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1, test_size = 0.2)
#Separate different attribute type to catergorical and numerical features.

cat_col = ['model', 'transmission', 'fuelType']

num_col = ['mileage', 'tax', 'mpg', 'age','engineSize']
#For numberical features, let's standardized it before feeding into our model

#For categorical features, perform one hot encoding before feeding into model

full_pipeline = ColumnTransformer([

    ('num', StandardScaler(), num_col),

    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_col)

])
#Apply data transformation to train set

X_train_prepared = full_pipeline.fit_transform(X_train)
#X_train_prepared

pd.DataFrame.sparse.from_spmatrix(X_train_prepared,columns = full_pipeline.transformers_[0][2]+full_pipeline.transformers_[1][1].get_feature_names().tolist()  )
model_list = [(ElasticNet(),'ElasticNet'),

              (SGDRegressor(),'SGDRegressor'),

              (SVR(kernel='linear'),'SVR-linear'),

              (SVR(kernel='rbf'),'SVR-rbf'),

              (RandomForestRegressor(),'RandomForestRegressor'),

              (xgb.XGBRegressor(),'XGBoost')

             ]



model_score = []



for m in model_list:

    model = m[0]

    score = cross_val_score(model,X_train_prepared,y_train,cv=4, scoring='r2')

    print(f'{m[1]} score = {score.mean()}')

    model_score.append([m[1],score.mean()])
%%script false --no-raise-error

from sklearn.model_selection import GridSearchCV



#Hyperparameter to be tweaked

param_grid = [

    {'n_estimators': [100,200,300],

    'max_depth' : [3,5,10],

    'reg_lambda' : [0.1,1,3,10,30],

    'reg_alpha': [0.1,1,3,10,30],

    'learning_rate': [0.15,0.3,0.5]}

]



xgb_regressor = xgb.XGBRegressor()

grid_search = GridSearchCV(xgb_regressor,param_grid, cv=4, scoring = 'r2', return_train_score = True)
%%script false --no-raise-error

grid_search.fit(X_train_prepared,y_train)
%%script false --no-raise-error

#Save model for later use

model = grid_search.best_estimator_

joblib.dump(model , 'XGBRegressor.pkl')
my_model = joblib.load('../input/bmw-price-prediction/XGBRegressor.pkl')
# Apply preprocessing to test set

X_test_prepared = full_pipeline.transform(X_test)

# Fit model to training set

my_model.fit(X_train_prepared,y_train)

prediction = my_model.predict(X_test_prepared)
#Result of model's prediction (y_predicted) compared to actual car price (y_test)

np.random.seed(0)

compare = pd.DataFrame(data = [np.array(y_test),prediction])

compare = compare.T

compare.columns = ['Actual_price','Predicted_price']

compare.sample(10)
from yellowbrick.regressor import prediction_error

# Create the train and test data

# Instantiate the linear model and visualizer

visualizer = prediction_error(my_model, X_train_prepared, y_train, X_test_prepared, y_test)
importance = pd.DataFrame(data=my_model.feature_importances_, index = full_pipeline.transformers_[0][2]+full_pipeline.transformers_[1][1].get_feature_names().tolist(),columns=['Importance_Score'])

importance.sort_values(by='Importance_Score',ascending=False)

sns.set(style="ticks")



# Initialize the figure

f, ax = plt.subplots(figsize=(16, 12))



sns.barplot(x = 'Importance_Score', y=importance.index, data=importance,

           order = importance.sort_values(by='Importance_Score',ascending=False).index,

            palette="vlag")

           

ax.xaxis.grid(True)

ax.set_title('Feature Importance Ranking')

sns.despine(trim=True, left=True)