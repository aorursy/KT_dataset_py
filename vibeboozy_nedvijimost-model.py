import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
my_filepath = "../input/Nedvijimost.csv"
real_estate_data = pd.read_csv(my_filepath)
real_estate_data.head()
real_estate_data.describe()
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
y = real_estate_data['Стоимость (т.руб.)']
features = ['Район', 'Тип планировки', 'Количество комнат',  'Общая площадь (м2)', 'Жилая площадь (м2)', 'Площадь кухни (м2)', 'Состояние']
X = real_estate_data[features]
# data split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
mln_mae_dict = {}
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500, 1000, 2000]

for leaf_size in candidate_max_leaf_nodes:
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('model', DecisionTreeRegressor(max_leaf_nodes=leaf_size, random_state=1))
    ])

    pipeline.fit(train_X, train_y)
    preds = pipeline.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    mln_mae_dict[leaf_size] = mae
        
mln_mae_dict
best_tree_size = min(mln_mae_dict, key=mln_mae_dict.get)
best_tree_size
# best model - with best_tree_size max leaf nodes

best_model =  Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('model', DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1))
])

best_model.fit(train_X, train_y)
preds = best_model.predict(val_X)
mean_absolute_error(val_y, preds)