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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


estate = pd.read_csv('/kaggle/input/Nedvijimost.csv')
estate.describe()
estate.head(10)
print(estate['Тип планировки'].unique())
estate['Район'].unique()
print(estate.columns)
y = estate['Стоимость (т.руб.)']
features = ['Тип планировки', 'Количество комнат',  'Общая площадь (м2)', 'Жилая площадь (м2)', 'Площадь кухни (м2)', 'Состояние']
X = estate[features]
# data split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

mln_mae_dict = {}

for mln in [10, 50, 100, 500, 1000, 2000, 5000, 10000]:
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('model', DecisionTreeRegressor(max_leaf_nodes=mln, random_state=1))
    ])

    pipeline.fit(train_X, train_y)
    preds = pipeline.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    print(mae)
    
    mln_mae_dict[mln] = mae
        
mln_mae_dict
# best model - with 1000 max leaf nodes

best_model =  Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('model', DecisionTreeRegressor(max_leaf_nodes=1000, random_state=1))
])

best_model.fit(train_X, train_y)
preds = best_model.predict(val_X)
mean_absolute_error(val_y, preds)