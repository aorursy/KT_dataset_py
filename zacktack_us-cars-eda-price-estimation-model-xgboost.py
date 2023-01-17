# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from pandas_profiling import ProfileReport



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def plot_bar_vertical(df,figsize=(10,15),xlabel='Count Number',color='tab:blue'):

    ax = df.plot.barh(figsize=figsize,color=color)

    plt.xlabel(xlabel)

    for p in ax.patches:

        ax.text(p.get_x() + p.get_width(),p.get_y() + p.get_height()/2,f'{int(p.get_width())}')



        

def plot_donut_chart(df,figsize=(10,15),subplots=True,radius=0.7,pctdistance=0.8):

    df.plot.pie(figsize=figsize,subplots=subplots,pctdistance=pctdistance,explode=[0.1 for x in df.index])

    centre_circle = plt.Circle((0,0),radius,fc='white')

    fig = plt.gcf()

    fig.gca().add_artist(centre_circle)
car_us_df = pd.read_csv("/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv",index_col=0)

car_us_df.head()
car_us_df.info()
car_us_df.describe()
ProfileReport(car_us_df)
ax = car_us_df.price.plot.hist(figsize=(12,10))

plt.xlabel('Price')

plt.xticks(np.arange(0,100000,10000))

for p in ax.patches:

        ax.text(p.get_x() + p.get_width()/4,p.get_y() + p.get_height(),f'{int(p.get_height())}')
car_us_df.price.describe()
special_cases_price_zero = car_us_df[(car_us_df.price == 0)]

display(special_cases_price_zero.head(10))

special_cases_price_zero.shape
plot_bar_vertical(special_cases_price_zero.groupby(['brand','model']).brand.count().sort_values())
ax = plot_donut_chart(special_cases_price_zero.groupby(['brand','model']).brand.count(),radius=0.8)
special_cases_price_zero[special_cases_price_zero.title_status == 'clean vehicle']
special_cases_price_great = car_us_df[car_us_df.price >= 30000]

special_cases_price_great.head()
plot_bar_vertical(special_cases_price_great.groupby(['brand','model']).brand.count().sort_values().tail(10))
car_us_df[car_us_df.price == car_us_df.price.max()]
plot_bar_vertical(car_us_df.groupby(['brand','model']).brand.count().sort_values().tail(10),color='tab:green')
plot_bar_vertical(car_us_df.groupby(['brand','model']).price.mean().sort_values().tail(10),(12,15),'price',color='tab:green')
section_color = 'tab:brown'
plot_bar_vertical(car_us_df.title_status.value_counts(),(10,6),color=section_color)
print(f'Salvage insuranced rate: {round((car_us_df.title_status.value_counts()[1] / car_us_df.title_status.value_counts()[0]) * 100,2)}%')
plot_bar_vertical(car_us_df.groupby('title_status').price.mean(),(12,6),'Price',color=section_color)
car_us_df
section_color ='tab:cyan'
ax = car_us_df.plot(kind='scatter',x='mileage',y='price',figsize=(10,8),color=section_color)
print(f'{int(car_us_df.mileage.max())} miles')

car_us_df[car_us_df.mileage == car_us_df.mileage.max()]
section_color = 'tab:orange'
top_colors = [color if color != 'no_color' else '#FFFFFF' for color in car_us_df.color.value_counts().sort_values().tail(10).index]

ax = car_us_df.color.value_counts().sort_values().tail(10).plot.barh(color=top_colors,figsize=(12,10))

ax.set_facecolor('purple')

for p in ax.patches:

    ax.text(p.get_x() + p.get_width(),p.get_y() + p.get_height()/2,f'{int(p.get_width())}')
section_color ='tab:purple'
plt.ylabel('US State')

plot_bar_vertical(car_us_df.state.value_counts().sort_values().tail(10),color=section_color)
price_estimator_df = car_us_df.copy()

features_to_drop = ['vin','lot','country','condition']

price_estimator_df.drop(features_to_drop,axis=1,inplace=True)

price_estimator_df
cat_features = [col for col in price_estimator_df.select_dtypes('object')]

cat_features
train = price_estimator_df.drop('price',axis=1)

target = price_estimator_df.price
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict,GridSearchCV

from sklearn.metrics import r2_score

from xgboost import XGBRegressor
[(col,train[col].nunique()) for col in cat_features]
cat_transformer = LabelEncoder()



for col in cat_features:

    train[col] = cat_transformer.fit_transform(train[col])

train
X_train,X_val,y_train,y_val = train_test_split(train,target,test_size=0.2,shuffle=True,random_state=20)
model = XGBRegressor()

model.fit(X_train,y_train)

r2_score(y_pred=model.predict(X_val),y_true=y_val)
params_set1 = {'max_depth':[3,4,5],'gamma':[0,1,5]}



model = XGBRegressor()



clf = GridSearchCV(model,params_set1,cv=KFold(n_splits=5),scoring='r2',refit=True)

clf.fit(X_train,y_train)





display(clf.best_score_,clf.best_params_)
params_set2 = {'n_estimators':[50,100,500,1000],'learning_rate':[0.01,0.03,0.05]}



model = XGBRegressor(max_depth=4,gamma=0)



clf = GridSearchCV(model,params_set2,cv=KFold(n_splits=5),scoring='r2',refit=True)

clf.fit(X_train,y_train)





display(clf.best_score_,clf.best_params_)
model = XGBRegressor(gamma=0, max_depth= 4,learning_rate=0.03,n_estimators=1000)

model.fit(X_train,y_train)

r2_score(y_pred=model.predict(X_val),y_true=y_val)