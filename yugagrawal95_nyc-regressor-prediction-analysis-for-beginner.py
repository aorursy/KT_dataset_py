#import all important library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#import 2 miliion record
train_data = pd.read_csv('../input/train.csv',nrows=20000)
test_data = pd.read_csv('../input/test.csv')
train_data.head()
test_data.head()
train_data.describe()
train_data.shape
# def memoryUsageCheck(dataset):
#     dataset.info(memory_usage='deep')

# print("memory usage for train data",memoryUsageCheck(train_data))
# print("--"*40)
# print("memory usage for test data",memoryUsageCheck(test_data))

# for dtype in ['float','int','object']:
#     selected_dtype = train_data.select_dtypes(include=[dtype])
#     mean_usage_byte = selected_dtype.memory_usage(deep=True).mean()
#     mean_usage_mb = mean_usage_byte / 1024 ** 2
#     print("Average memory usage in MB for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))
def changeDataType(dataset):
    dataset['passenger_count'] = dataset.passenger_count.astype('uint8')
    dataset['pickup_longitude'] = dataset.pickup_longitude.astype('float32')
    dataset['pickup_latitude'] = dataset.pickup_latitude.astype('float32')
    dataset['dropoff_longitude'] = dataset.dropoff_longitude.astype('float32')
    dataset['dropoff_latitude'] = dataset.dropoff_latitude.astype('float32')
    dataset['pickup_datetime'] = pd.to_datetime(arg=dataset['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')
    dataset.info()
    

changeDataType(train_data)
print("--"*40)
changeDataType(test_data)

train_data['fare_amount'] = train_data.fare_amount.astype('float32')

#dataset.info(memory_usage='deep')
train_data['pickup_datetime'].head()
train_data.describe()
train_data.isnull().sum()
train_data = train_data.dropna(axis=0)
train_data.isnull().sum()
pd.set_option('float_format', '{:f}'.format)
train_data.describe()
plt.figure(figsize=(8, 5), dpi=80)
# sns.set_style("darkgrid")
sns.distplot(train_data['fare_amount'],color='red',kde=False)
train_data = train_data.loc[train_data['fare_amount']>0]
train_data['fare_amount']
train_data.describe()

sns.distplot(a=train_data.fare_amount, kde=False)
p = pd.cut(train_data.fare_amount,3)
p.value_counts()
train_data = train_data[train_data.fare_amount<400]
sns.kdeplot(data=train_data.fare_amount)
sns.countplot(x=train_data.passenger_count)
train_data.passenger_count.describe()
train_data = train_data[train_data.passenger_count<=6]
sns.distplot(a=train_data.passenger_count,kde=False)
train_data.describe()
train_data = train_data.drop((train_data[train_data['pickup_latitude']<-90] | (train_data[train_data['pickup_latitude']>90])).index,axis=0)
train_data = train_data.drop((train_data[train_data['pickup_longitude']<-180] | (train_data[train_data['pickup_longitude']>180])).index,axis=0)
train_data = train_data.drop((train_data[train_data['dropoff_longitude']<-180] | (train_data[train_data['dropoff_longitude']>180])).index,axis=0)
train_data = train_data.drop((train_data[train_data['dropoff_latitude']<-90] | (train_data[train_data['dropoff_latitude']>90])).index,axis=0)
train_data = train_data[train_data.pickup_latitude.between(test_data.pickup_latitude.min(),test_data.pickup_latitude.max())]
train_data = train_data[train_data.pickup_longitude.between(test_data.pickup_longitude.min(),test_data.pickup_longitude.max())]
train_data = train_data[train_data.dropoff_latitude.between(test_data.dropoff_latitude.min(),test_data.dropoff_latitude.max())]
train_data = train_data[train_data.dropoff_longitude.between(test_data.dropoff_longitude.min(),test_data.dropoff_longitude.max())]
sns.scatterplot(x=train_data.pickup_latitude,y=train_data.pickup_longitude)
sns.scatterplot(x=train_data.dropoff_latitude,y=train_data.dropoff_longitude)
def degree_to_radion(degree):
    return degree*(np.pi/180)

def calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
    
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)
    
    radius = 6371.01
    
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = np.sin(lat_diff / 2)**2 + np.cos(degree_to_radion(from_lat)) * np.cos(degree_to_radion(to_lat)) * np.sin(long_diff / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return radius * c
train_data['distance'] = calculate_distance(train_data.pickup_latitude, train_data.pickup_longitude, train_data.dropoff_latitude, train_data.dropoff_longitude)
train_data.describe()
test_data['distance'] = calculate_distance(test_data.pickup_latitude, test_data.pickup_longitude, test_data.dropoff_latitude, test_data.dropoff_longitude)
test_data.describe()
p = pd.cut(train_data.distance,10)
p.value_counts()
train_data = train_data.loc[train_data.distance<200] #150
train_data.describe()
sns.distplot(train_data.distance,kde=False)
train_data = train_data.drop(columns='key')
train_data.describe()
test_data_key = test_data['key']
test_data = test_data.drop(columns='key')
test_data.head()
data = [train_data,test_data]
for i in data:
    i['Year'] = i['pickup_datetime'].dt.year
    i['Month'] = i['pickup_datetime'].dt.month
    i['Date'] = i['pickup_datetime'].dt.day
    i['Day of Week'] = i['pickup_datetime'].dt.dayofweek
    i['Hour'] = i['pickup_datetime'].dt.hour
train_data.head()
test_data.head()
sns.scatterplot(x=train_data['passenger_count'],y=train_data['fare_amount'])
sns.scatterplot(x=train_data['distance'],y=train_data['fare_amount'])
g = sns.FacetGrid(train_data,col='Year')
g.map(sns.scatterplot,"distance","fare_amount")
train_data.describe()
train_data[(train_data.distance>100) & (train_data.fare_amount<50)]
sns.scatterplot(x=train_data['Year'],y=train_data['fare_amount'])
train_data.groupby(['Month','Year']).count()['fare_amount']
sns.scatterplot(x=train_data['Month'],y=train_data['fare_amount'],hue=train_data['Year'])
sns.scatterplot(x=train_data['Month'],y=train_data['fare_amount'])
w = sns.FacetGrid(train_data,col='Year')
w.map(sns.scatterplot,"Month","fare_amount")
sns.barplot(x=train_data['Day of Week'],y=train_data['fare_amount'])
plt.figure(figsize=(10, 10), dpi=150)
w = sns.FacetGrid(train_data,col='Month')
w.map(sns.barplot,"Day of Week","fare_amount")
sns.barplot(x=train_data['Hour'],y=train_data['fare_amount'])
train_data = train_data.loc[train_data.pickup_latitude != 0]
train_data = train_data.loc[train_data.pickup_longitude != 0]
train_data = train_data.loc[train_data.dropoff_latitude != 0]
train_data = train_data.loc[train_data.dropoff_longitude != 0]
test_data = test_data.loc[test_data.pickup_latitude != 0]
test_data = test_data.loc[test_data.pickup_longitude != 0]
test_data = test_data.loc[test_data.dropoff_latitude != 0]
test_data = test_data.loc[test_data.dropoff_longitude != 0]
train_data = train_data.drop(columns='pickup_datetime',axis=1)
test_data = test_data.drop(columns='pickup_datetime',axis=1)
X = train_data.loc[:,train_data.columns != 'fare_amount']
y = train_data['fare_amount']
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score,f1_score

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
y_train = np.log(y_train)
y_test = np.log(y_test)
X_train.describe()
ran_for_reg = RandomForestRegressor(max_depth=400)
ran_for_reg.fit(X_train,y_train)
y_ranfor_pred = ran_for_reg.predict(X_test)
error = np.sqrt(mean_squared_error(y_test,y_ranfor_pred))
error
sns.barplot(x=ran_for_reg.feature_importances_,y=X_test.columns)
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
bagreg = BaggingRegressor(base_estimator=DecisionTreeRegressor(),n_estimators=10,bootstrap=True,random_state=0)
bagreg.fit(X_train,y_train)
y_bagg_pred = bagreg.predict(X_test)
error = np.sqrt(mean_squared_error(y_test,y_bagg_pred))
error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
adareg = AdaBoostRegressor(DecisionTreeRegressor())
adareg.fit(X_train,y_train)
y_adareg_pred = adareg.predict(X_test)
error = np.sqrt(mean_squared_error(y_test,y_adareg_pred))
error
from sklearn.ensemble import GradientBoostingRegressor
gradient_reg = GradientBoostingRegressor()
gradient_reg.fit(X_train,y_train)
y_gradient_pred = gradient_reg.predict(X_test)
error = np.sqrt(mean_squared_error(y_test,y_gradient_pred))
error
from xgboost import XGBRegressor
xgreg = XGBRegressor()
xgreg.fit(X_train,y_train)
y_xgreg_pred = xgreg.predict(X_test)
error = np.sqrt(mean_squared_error(y_test,y_xgreg_pred))
error
import lightgbm as lgb
model_lgb = lgb.LGBMRegressor()
model_lgb.fit(X_train,y_train)
y_lgb_pred = model_lgb.predict(X_test)
error = np.sqrt(mean_squared_error(y_test,y_lgb_pred))
error
# from sklearn.model_selection import cross_val_score,KFold,cross_val_predict
# kfold = KFold(n_splits=10,random_state=10)
# cv_result = cross_val_score(XGBRegressor(),X,y,cv=kfold,scoring='neg_mean_squared_error')
# cv_result 
# np.sqrt(sum(cv_result * -1)/len(cv_result))
#y_pred_final = xgreg.predict(test_data)

# submission = pd.DataFrame(
#     {'key': test_data_key, 'fare_amount': y_pred_final},
#     columns = ['key', 'fare_amount'])
# submission.to_csv('submission.csv', index = False)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# a stacking function which divides training and testing data and findouts prediction

n_folds=5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X_train,y_train,scoring='neg_mean_squared_error',cv=kf))
    return rmse

from sklearn.ensemble import GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
from xgboost import XGBRegressor
model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
import lightgbm as lgb
model_lgb1 = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
# score = rmsle_cv(GBoost)
# print("\Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(model_xgb)
# print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(model_lgb)
# print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
class AverageModel(BaseEstimator):
    def __init__(self,models):
        self.models = models
        
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=1)
average_model = AverageModel(models=(GBoost,model_xgb, model_lgb1))
score = rmsle_cv(average_model)
score
score.mean()
class stackingModel(BaseEstimator):
    def __init__(self,base_model, meta_model, k_fold=5):
        self.base_model = base_model
        self.meta_model = meta_model
        self.k_fold = k_fold
    
    def fit(self,X,y):
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=156)
        out_of_fold_predictions = np.zeros((X.shape[0],len(self.base_models)))
        for i, model in enumerate(self.base_model):
            for train_index,holdout_index in kfold.split(X,y):
                model.fit(X[train_index],y[train_index])
                y_pred = model.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        return self
    
    def predict(self,X):
        meta_feature = np.column_stack([np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
                                       for base_models in self.base_model])
        return self.meta_model.predict(meta_features)

#just checking....

base_model = [xgreg,gradient_reg,model_lgb]

def test1(X,y):
    kfold = KFold(n_splits=5,shuffle=True)
    out_of_fold_predictions = np.zeros((X.shape[0],len(base_model)))
    for i, model in enumerate(base_model):
            for train_index,holdout_index in kfold.split(X,y):
                    model.fit(X.iloc[train_index],y.iloc[train_index])
                    y_pred = model.predict(X.iloc[holdout_index])
                    out_of_fold_predictions[holdout_index, i] = y_pred
    return out_of_fold_predictions
    
out_of_fold_predictions = test1(X_train,y_train)

out_of_fold_predictions
#meta model training
meta_model = lgb.LGBMRegressor()
meta_model.fit(out_of_fold_predictions,y_train)
#meta_model.predict(np.column_stack([]))
base_model = [xgreg,gradient_reg,model_lgb]
feature_data = np.column_stack([ np.column_stack([model.predict(test_data) for model in base_model]).mean(axis=1) for base_models in base_model])
meta_y = meta_model.predict(feature_data)
meta_y = np.exp(meta_y)
# error = np.sqrt(mean_squared_error(y_test,meta_y))
# error
submission = pd.DataFrame(
    {'key': test_data_key, 'fare_amount': meta_y},
    columns = ['key', 'fare_amount'])
#meta_y
submission.to_csv('submission.csv', index = False)
meta_y