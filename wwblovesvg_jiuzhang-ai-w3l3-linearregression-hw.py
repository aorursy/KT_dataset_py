import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型
from sklearn.model_selection import train_test_split # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
import math #数学库
from sklearn.metrics import r2_score, mean_squared_error
data = pd.read_csv("../input/kc_house_data.csv", header = 0, sep= ',')
data.head()
print(data.dtypes)
print(data.columns)
print(data.info())
print(data.describe())
origin_x_train = data[['bedrooms','bathrooms','sqft_living','floors']]
origin_y_train = data['price']
print(origin_x_train.shape, origin_y_train.shape)
split = train_test_split(origin_x_train, origin_y_train, test_size = 1/3, random_state = 0)
x_train, x_vali, y_train, y_vali = split
for i in range(4):
    print(split[i].shape)
x_train = np.asmatrix(x_train)
x_vali = np.asmatrix(x_vali)
y_train = np.ravel(y_train)
y_vali = np.ravel(y_vali)
plt.title("House price and living space scatter plot")
plt.scatter(data.sqft_living, data.price)
plt.xlabel("living space in square feet")
plt.ylabel("house price")
plt.show()
plt.title("Living space distribution")
plt.xlabel("Living space in square feet")
plt.ylabel("Count")
data.sqft_living.hist()
plt.show()
model = LinearRegression()
model.fit(x_train, y_train)
pd.DataFrame(list(zip(origin_x_train.columns, np.transpose(model.coef_))))
model.intercept_
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，预测房价
model.predict([[3,2,2500,2]])
y_pred = model.predict(x_vali)
print(mean_squared_error(y_vali, y_pred))
mse = ((y_vali - y_pred) ** 2).sum() / len(y_vali)
print(mse)
mre = (abs(y_vali - y_pred) / y_vali).sum() / len(y_pred)
print(mre)
# 手动算β
# beta = xtx-1xty
xt = np.transpose(x_train)
xtx = xt * x_train
xtx_inv = np.linalg.inv(xtx)
beta = xtx_inv * xt * np.transpose(np.asmatrix(y_train))
pd.DataFrame(list(zip(origin_x_train.columns, beta)))
pd.DataFrame(list(zip(origin_x_train.columns, np.transpose(model.coef_))))