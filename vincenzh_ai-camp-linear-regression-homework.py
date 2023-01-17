import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型
from sklearn.model_selection import train_test_split # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
import math #数学库
from sklearn.metrics import r2_score
data = pd.read_csv('../input/kc_house_data.csv')
data
data.dtypes
X = data[['bedrooms','bathrooms','sqft_living','floors','view','condition','sqft_above','yr_built','lat']]
Y = data['price']
plt.scatter(data['zipcode'], Y)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 1/3, random_state = 0)
xtrain = np.asmatrix(xtrain)
xtest = np.asmatrix(xtest)
ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)
plt.scatter(X['sqft_living'],Y)
plt.show()
X['sqft_living'].hist()
plt.show()
model = LinearRegression()
model.fit(xtrain, ytrain)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
model.intercept_
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，预测房价

pred = model.predict(xtrain)
((pred - ytrain)*(pred - ytrain)).sum() / len(ytrain)
(abs(pred - ytrain)/ytrain).sum() / len(ytrain)
r2_vali = r2_score(ytrain, pred)
adj_r2_vali = 1 - (1 - r2_vali) * ((len(ytrain) - 1)/(len(ytrain) - xtrain.shape[1] - 1))
print(r2_vali)
print(adj_r2_vali)
predtest = model.predict(xtest)
((predtest-ytest)*(predtest-ytest)).sum() / len(ytest)
(abs(predtest-ytest)/ytest).sum() / len(ytest)
r2_test = r2_score(ytest, predtest)
adj_r2_test = 1 - (1 - r2_test) * ((len(ytest) - 1)/(len(ytest) - xtest.shape[1] - 1))
print(r2_test)
print(adj_r2_test)