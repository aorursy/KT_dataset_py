import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

df = pd.read_csv("../input/kc_house_data.csv")

# splitting data
train_data,test_data = train_test_split(df,train_size = 0.8,random_state=3)
# Linear Model 
lr = linear_model.LinearRegression()
X_train = np.array(train_data['sqft_living'], dtype=pd.Series).reshape(-1,1)
y_train = np.array(train_data['price'], dtype=pd.Series)
# fitting the linear model
lr.fit(X_train,y_train)

# # Evaluate the simple model
x_test = np.array(test_data['sqft_living'], dtype=pd.Series).reshape(-1,1)
y_test = np.array(test_data['price'], dtype=pd.Series)

# 予測データを使用してMSE（平均二乗誤差）を求める。
y_pred = lr.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print("MSE=%s"%round(mse,3) )

msesm = format(np.sqrt(metrics.mean_squared_error(y_test,y_pred)),'.3f')
rtrsm = format(lr.score(X_train, y_train),'.3f')
rtesm = format(lr.score(x_test, y_test),'.3f')

print("RMSE=" + msesm)
print("coefficient（train）=" + rtrsm)
print("coefficient（test）=" + rtesm)

