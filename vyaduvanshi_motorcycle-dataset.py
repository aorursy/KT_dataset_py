# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/motorcycle-dataset/BIKE DETAILS.csv')

df.head()
df.info()
for i in df.select_dtypes(include='object'):

    print(df[i].value_counts(), end='\n'*3)
df.replace({'1st owner':1, '2nd owner':2, '3rd owner':3, '4th owner':4}, inplace=True)

df.rename(columns = {'owner':'prev_owners'}, inplace=True)
current_year = 2020

df['age'] = current_year - df['year']
df.drop(['year','name'], axis=1, inplace=True)
df_2 = df.copy()

new_df = df.dropna().copy()
impute = SimpleImputer(strategy = 'median')



df_2_num = df_2.select_dtypes(exclude='object')

df_2_cat = df_2['seller_type']

x = impute.fit_transform(df_2_num)
df_2 = pd.DataFrame(x, columns=df_2.select_dtypes(exclude='object').columns).copy()



df_2['seller_type'] = df_2_cat
new_df.head()
new_df.hist(figsize=(20,15), grid=0, bins=20);
new_df[new_df.selling_price>100000]
new_df['price_category'] = pd.cut(new_df['selling_price'], bins=[0,50000,100000,np.inf],labels=[1,2,3])
split = StratifiedShuffleSplit(random_state=42, n_splits=1, test_size=0.25)



for i,j in split.split(new_df, new_df['price_category']):

    new_df_train = new_df.iloc[i]

    new_df_test = new_df.iloc[j]
for i in (new_df_train, new_df_test):

    i.drop('price_category', axis=1, inplace=True)
new_df_train.corr()
sns.pairplot(new_df_train);
new_df_train_labels = new_df_train['selling_price']

new_df_train.drop('selling_price', axis=1, inplace=True)
num = new_df_train.select_dtypes(exclude='object').columns

cat = new_df_train.select_dtypes(include='object').columns
col_transformer = ColumnTransformer([('num',StandardScaler(), num),('cat', OneHotEncoder(), cat)])

col_transformer
new_df_train_prepared = col_transformer.fit_transform(new_df_train)

new_df_train_prepared
new_df_train_prepared = pd.DataFrame(new_df_train_prepared)

new_df_train_prepared
new_df_train_prepared.drop(5, axis=1, inplace=True)
lr = LinearRegression()

lr.fit(new_df_train_prepared, new_df_train_labels)
def pred_and_score(model, df, labels):

    pred = model.predict(df)

    mse = mean_squared_error(labels, pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(labels, pred)

    mae = mean_absolute_error(labels, pred)



    print(f"RMSE: {rmse}\nr2_score: {r2}\nMAE: {mae}")
pred_and_score(lr, new_df_train_prepared, new_df_train_labels)
custom_train = new_df_train_prepared[[1,2,3]].copy()
lr2 = LinearRegression()

lr2.fit(custom_train, new_df_train_labels)
pred_and_score(lr2, custom_train, new_df_train_labels)
dt = DecisionTreeRegressor(random_state=42)

dt.fit(new_df_train_prepared, new_df_train_labels)



pred_and_score(dt, new_df_train_prepared, new_df_train_labels)
x_test = new_df_test.drop('selling_price', axis=1)

y_test = new_df_test['selling_price']
test_prepared = col_transformer.transform(x_test)

test_prepared
test_prepared = pd.DataFrame(test_prepared).drop(5, axis=1)
pred_and_score(dt, test_prepared, y_test)
pred_and_score(lr, test_prepared, y_test)