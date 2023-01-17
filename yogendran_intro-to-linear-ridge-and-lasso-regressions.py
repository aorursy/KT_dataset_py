# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Automobile_data.csv')

list(data)
data['horsepower'] = pd.to_numeric(data['horsepower'], errors = 'coerce')

data['price'] = pd.to_numeric(data['price'], errors = 'coerce')

# data.any().isna()

data.dropna(subset=['price', 'horsepower'], inplace=True)

# type(data['horsepower'][1])
from scipy.stats.stats import pearsonr

pearsonr(data['horsepower'], data['price'])

data['horsepower'].head()
from bokeh.io import output_notebook

from bokeh.plotting import ColumnDataSource, figure, show



# enable notebook output

output_notebook()



source = ColumnDataSource(data=dict(

    x=data['horsepower'],

    y=data['price'],

    make=data['make'],

))



tooltips = [

    ('make', '@make'),

    ('horsepower', '$x'),

    ('price', '$y{$0}')

]



p = figure(plot_width=600, plot_height=400, tooltips=tooltips)

p.xaxis.axis_label = 'Horsepower'

p.yaxis.axis_label = 'Price'



# add a square renderer with a size, color, and alpha

p.circle('x', 'y', source=source, size=8, color='blue', alpha=0.5)



# show the results

show(p)
from sklearn.model_selection import train_test_split

train, test=  train_test_split(data, test_size = 0.25)
from sklearn import linear_model

model = linear_model.LinearRegression()

training_x = np.array(train['horsepower']).reshape(-1,1)

training_y = np.array(train['price'])

model.fit(training_x, training_y)

slope = np.asscalar(np.squeeze(model.coef_))

intercept = model.intercept_

print('slope:', slope, 'intercept:', intercept)

# Now let's add the line to our graph

from bokeh.models import Slope

best_line = Slope(gradient=slope, y_intercept=intercept, line_color='red', line_width=3)

p.add_layout(best_line)

show(p)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# function to predict the mean_absolute_error, mean_squared_error and r-squared

def predict_metrics(lr, x, y):

    pred = lr.predict(x)

    mae = mean_absolute_error(y, pred)

    mse = mean_squared_error(y, pred)

    r2 = r2_score(y, pred)

    return mae, mse, r2



training_mae, training_mse, training_r2 = predict_metrics(model, training_x, training_y)



test_x = np.array(test['horsepower']).reshape(-1,1)

test_y = np.array(test['price'])



test_mae, test_mse, test_r2 = predict_metrics(model, test_x, test_y)



print('training mean error:', training_mae, 'training mse:', training_mse, 'training r2:', training_r2)

print('test mean error:', test_mae, 'test mse:', test_mse, 'test r2:', test_r2)
#Getting the correlation between other variables/columns



cols = ['horsepower', 'engine-size', 'peak-rpm', 'length', 'width', 'height']

for col in cols:

    data[col] = pd.to_numeric(data[col], errors = 'coerce')

data.dropna(subset = ['price', 'horsepower'], inplace = True)



# Let's see how strongly each column is correlated to price

for col in cols:

    print(col, pearsonr(data[col], data['price']))
# split train and test data as before

model_cols = ['horsepower', 'engine-size', 'length', 'width']

multi_x = np.column_stack(tuple(data[col] for col in model_cols))

y = data['price']



multi_train_x, multi_test_x, multi_train_y, multi_test_y = train_test_split(multi_x, y, test_size = 0.25)



# fit the model as before

multi_model = linear_model.LinearRegression()

multi_model.fit(multi_train_x, multi_train_y)

multi_model_intercept = multi_model.intercept_

multi_coefficient = dict(zip(model_cols,multi_model.coef_))

print('intercept:', multi_model_intercept)

print('Co-efficients:', multi_coefficient)

# calculate error metrics

m_train_mae, m_train_mse, m_train_r2 = predict_metrics(multi_model, multi_train_x, multi_train_y)

m_test_mae, m_test_mse, m_test_r2 = predict_metrics(multi_model, multi_test_x, multi_test_y)



print('m_train_mean_error:', m_train_mae, 'm_train_mae:', m_train_mse, 'm_train_r2', m_train_r2 )

print('m_test_mean_error:', m_test_mae, 'm_test_mae:', m_test_mse, 'm_test_r2', m_test_r2 )
from sklearn.linear_model import Ridge

ridge = Ridge(alpha = 0.05, normalize = True)

ridge.fit(multi_train_x, multi_train_y)



r_train_mae, r_train_mse, r_train_r2 = predict_metrics(ridge, multi_train_x, multi_train_y)

r_test_mae, r_test_mse, r_test_r2 = predict_metrics(ridge, multi_test_x, multi_test_y)



print('r_train_mean_error:', r_train_mae, 'r_train_mae:', r_train_mse, 'r_train_r2', r_train_r2 )

print('r_test_mean_error:', r_test_mae, 'r_test_mae:', r_test_mse, 'r_test_r2', r_test_r2 )

from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha = 0.05, normalize = True)

lasso_model.fit(multi_train_x, multi_train_y)



l_train_mae, l_train_mse, l_train_r2 = predict_metrics(lasso_model, multi_train_x, multi_train_y)

l_test_mae, l_test_mse, l_test_r2 = predict_metrics(lasso_model, multi_test_x, multi_test_y)



print('train_mean_error:', l_train_mae, 'train_mae:', l_train_mse, 'train_r2', l_train_r2 )

print('test_mean_error:', l_test_mae, 'test_mae:', l_test_mse, 'test_r2', l_test_r2 )
from sklearn.linear_model import ElasticNet

enet_model = ElasticNet(alpha=0.01, l1_ratio=0.5, normalize=False)

enet_model.fit(multi_train_x, multi_train_y)



el_train_mae, el_train_mse, el_train_r2 = predict_metrics(enet_model, multi_train_x, multi_train_y)

el_test_mae, el_test_mse, el_test_r2 = predict_metrics(enet_model, multi_test_x, multi_test_y)



print('train_mean_error:', el_train_mae, 'train_mae:', el_train_mse, 'train_r2', el_train_r2 )

print('test_mean_error:', el_test_mae, 'test_mae:', el_test_mse, 'test_r2', el_test_r2 )