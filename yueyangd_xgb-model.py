# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

#读取数据 因为这里面的‘Date’是Object类型，所以需要转换成Datetime类型
Chicago_Crime_data_2012_2017 = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv')
Chicago_Crime_data_2012_2017['Date'] = pd.to_datetime(Chicago_Crime_data_2012_2017['Date'])
my_data = Chicago_Crime_data_2012_2017.dropna()
#画一个每年的犯罪率统计
my_data.groupby(['Year']).count()['Primary Type'].plot()
#因为模型不能训练Date类型的特征，所以拆分成以下几个整数 年/月/日/时/分
my_data.insert(0,'Date_year',(my_data.Date.map(lambda x:x.year)))
my_data.insert(0,'Date_month',(my_data.Date.map(lambda x:x.month)))
my_data.insert(0,'Date_day',(my_data.Date.map(lambda x:x.day)))
my_data.insert(0,'Date_hour',(my_data.Date.map(lambda x:x.hour)))
my_data.insert(0,'Date_minute',(my_data.Date.map(lambda x:x.minute)))
#特征X设定: Date(_year, _month, _day, _hour, _minute), Arrest, Domestic, Primary Type(离散特征)
#预测目标y设定: 发现XGBoost模型的目标特征只能有一个，而这里需要经纬度，所以现在在下面把这两个特征放到两个模型分开训练

#X = my_data[['Date_year','Date_month','Date_day'，'Date_hour','Date_minute','Arrest','Domestic','Primary Type']]
X = my_data[['Date_year','Date_month','Date_hour','Date_day','Date_minute','District','Arrest','Domestic','Primary Type']]
y = my_data[['Latitude', 'Longitude']]
from sklearn.model_selection import train_test_split

#分train/test集
train_X, test_X, train_y, test_y = train_test_split(X,y)

#处理离散特征并且对齐train，test集
train_X = pd.get_dummies(train_X)
test_X = pd.get_dummies(test_X)
train_X, test_X = train_X.align(test_X, join = 'left', axis = 1)

#把y特征按照Laatitude和Longitude分开
train_y_Latitude = train_y.Latitude
train_y_Longitude = train_y.Longitude
test_y_Latitude = test_y.Latitude
test_y_Longitude = test_y.Longitude
from xgboost import XGBRegressor
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

#创建训练并返回一个XGB模型
def built_train_model(train_X, train_y, n_estimator = 100, learning_rate = 0.1, early_stopping_rounds = None):
    my_model = XGBRegressor(n_estimator = n_estimator, learning_rate = learning_rate)
    my_model.fit(train_X, train_y, early_stopping_rounds = early_stopping_rounds,  verbose=False)
    return my_model

#模型预测，返回经纬度
#因为XGB无法训练两个目标特征，所以将经纬度作为两个独立特征分开训练
def predict_location(model_Longitude, model_Latitude, test_X):
    predict_Longitude = model_Longitude.predict(test_X)
    predict_Latitude = model_Latitude.predict(test_X)
    return predict_Latitude, predict_Longitude

#通过计算MAE来验证模型，输出的是经度和纬度两个模型上的两个MAE误差
def predict_compute_MAE(model_Longitude, model_Latitude, test_X, test_y_Latitude, test_y_Longitude):
    predict_Latitude, predict_Longitude = predict_location(model_Longitude, model_Latitude, test_X)
    MAE_Latitude = mean_absolute_error(predict_Latitude, test_y_Latitude)
    MAE_Longitude = mean_absolute_error(predict_Longitude, test_y_Longitude)   
    return MAE_Latitude, MAE_Longitude

#在这里仅仅靠MAE来计算误差可能不是很恰当，这个方法可以‘画’出每个预测点的误差
def plot_error(test_X, test_y):
    test_y_Latitude_h = test_y.Latitude.tolist()
    test_y_Longitude_h = test_y.Longitude.tolist()
    predict_Latitude_h, predict_Longitude_h = predict_location(my_model_Longitude, my_model_Latitude, test_X)
    for i in range(0, len(test_y_Latitude_h)):
        plt.plot([test_y_Latitude_h[i],predict_Latitude_h[i]],[test_y_Longitude_h[i],predict_Longitude_h[i]])
    plt.show()
#训练模型，打印两个MAE误差
my_model_Longitude = built_train_model(train_X, train_y_Longitude)
my_model_Latitude = built_train_model(train_X, train_y_Latitude)
print('预测误差: ', predict_compute_MAE(my_model_Longitude, my_model_Latitude, test_X, test_y_Latitude, test_y_Longitude))
print(predict_location(my_model_Longitude, my_model_Latitude, test_X.head()))
print(test_y.head())
plot_error(test_X[10:60], test_y[10:60])