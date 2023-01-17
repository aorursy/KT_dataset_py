import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
sales.info()
merge = pd.merge(sales,items)
merge.corr()
fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sales['shop_id'], sales['item_id'], sales['item_cnt_day'], marker='o')
ax.set_xlabel('shop_id')
ax.set_ylabel('item_id')
ax.set_zlabel('item_cnt_day')
plt.show()
sales['item_cnt_day'] = sales['item_cnt_day'].clip(0,50)
fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sales['shop_id'], sales['item_id'], sales['item_cnt_day'], marker='o')
ax.set_xlabel('shop_id')
ax.set_ylabel('item_id')
ax.set_zlabel('item_cnt_day')
plt.show()
from sklearn.model_selection import train_test_split

X = pd.DataFrame(sales['item_id'],sales['shop_id'])
y = sales['item_cnt_day']
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
from sklearn.neural_network import MLPRegressor
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
pred = regr.predict(X_test)
print("R2 score:",regr.score(X_test, y_test))
from sklearn.linear_model import ElasticNet
regr = ElasticNet(random_state=1)
regr.fit(X, y)
pred = regr.predict(X_test)
print("R2 score:",regr.score(X_test, y_test))
test_x = pd.DataFrame(test['item_id'],test['shop_id'])
pred = regr.predict(test_x)
pred
