import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print('train shape:', train.shape, '\n', 'test shape:', test.shape)
train.head()


missing_numeric = pd.concat([train.isnull().sum(),train.isnull().sum()/train.isnull().count(), test.isnull().sum(),test.isnull().sum()/test.isnull().count()], axis=1, keys=['train_total','percent', 'test_total','percent'])
missing_numeric = missing_numeric[(missing_numeric['train_total']>0) & (missing_numeric['test_total']>0)]
missing_numeric.sort_values(by=['train_total', 'test_total'], ascending=False).head(20)
feature_drop = ['PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageYrBlt', 'MoSold', 'YrSold', 
                'LowQualFinSF', 'MiscVal', 'PoolArea']
datasets = [train, test]

train = train.drop((missing_numeric[missing_numeric['train_total'] > 1]).index,1)
test = test.drop((missing_numeric[missing_numeric['train_total'] > 1]).index,1)

    
print(train['SalePrice'].describe(), '\n')
print('Before Transformation Skew: ', train['SalePrice'].skew())

target = np.log1p(train['SalePrice'])
print('Log Transformation Skew: ', target.skew())

plt.rcParams['figure.figsize'] = (12, 5)
target_log_tran = pd.DataFrame({'befrore transformation':train['SalePrice'], 'log transformation': target})
target_log_tran.hist()
skewness = pd.DataFrame({'Skewness':train.select_dtypes(exclude=[object]).skew()})

print(skewness[skewness['Skewness']>0.8].sort_values(by='Skewness'), '\n')  
skewness_col= skewness[skewness['Skewness']>0.8].sort_values(by='Skewness').index.tolist()
print(skewness_col)
datasets = [train, test]
for df in datasets:
    for s in skewness_col:
        if s in df:
            df[s] = np.log1p(df[s])
            

skewness = pd.DataFrame({'Skewness':train.select_dtypes(exclude=[object]).skew()})

print(skewness[skewness['Skewness']>0.8].sort_values(by='Skewness'), '\n')  

corr = train.select_dtypes(exclude=[object]).corr()
numerical_data = train[corr['SalePrice'].sort_values(ascending=False)[:10].index.tolist()]
print(numerical_data.head())

categorical_data = train.select_dtypes(include=[object])
categorical_data.describe()

plt.rcParams['figure.figsize'] = (8, 120)
for index,ds in enumerate(categorical_data):
   plt.subplot(len(categorical_data.columns),1,index+1)
   sns.boxplot(categorical_data[ds], target)

train_ExterQual_dummy = pd.get_dummies(train['ExterQual'], prefix='ExterQual')
test_ExterQual_dummy = pd.get_dummies(test['ExterQual'], prefix='ExterQual')

train_ExterCond_dummy = pd.get_dummies(train['ExterCond'], prefix='ExterCond')
test_ExterCond_dummy = pd.get_dummies(test['ExterCond'], prefix='ExterCond')

train_SaleCondition_dummy = pd.get_dummies(train['SaleCondition'], prefix='SaleCondition')
test_SaleCondition_dummy = pd.get_dummies(test['SaleCondition'], prefix='SaleCondition')

train_CentralAir_dummy = pd.get_dummies(train['CentralAir'], prefix='CentralAir')
test_CentralAir_dummy = pd.get_dummies(test['CentralAir'], prefix='CentralAir')

train_KitchenQual_dummy = pd.get_dummies(train['KitchenQual'], prefix='KitchenQual')
test_KitchenQual_dummy = pd.get_dummies(test['KitchenQual'], prefix='KitchenQual')
train_exter_score = pd.Categorical(train.ExterQual).codes * pd.Categorical(train.ExterCond).codes
test_exter_score = pd.Categorical(test.ExterQual).codes * pd.Categorical(test.ExterCond).codes
print(type(train_exter_score))
#train_kitchen_score = train['KitchenAbvGr'] * train['KitchenQual']
#test_kitchen_score = test['KitchenAbvGr'] * test['KitchenQual']

#print(train_ExterCond_dummy)
data = pd.concat([numerical_data, train_ExterQual_dummy, train_SaleCondition_dummy, train_CentralAir_dummy, train_KitchenQual_dummy,pd.Series(train_exter_score)], axis=1)
y = data['SalePrice']
X = data.drop(['SalePrice'], axis=1)

lr  =  LinearRegression()
model_fit = lr.fit(X, y)
R2 = cross_val_score(model_fit, X, y, cv=10 , scoring='r2').mean()
MSE = -cross_val_score(lr, X, y, cv=10 , scoring='neg_mean_squared_error').mean()
print('R2 Score:', R2, '|', 'MSE:', MSE)
print(numerical_data.columns)
test_id = test['Id']
cols = numerical_data.columns.tolist()
cols.remove('SalePrice')
test = pd.concat([test[cols], test_ExterQual_dummy, test_SaleCondition_dummy, test_CentralAir_dummy, test_KitchenQual_dummy,pd.Series(test_exter_score)], axis=1)
test.head()

for i in test:
    test[i].fillna(0, inplace=True)
pred = lr.predict(test)
pred = np.expm1(pred)
prediction = pd.DataFrame({'Id':test_id, 'SalePrice':pred})
prediction.to_csv('linear_regression.csv', index=False)
prediction.head()
from keras.layers import Input, Dense
from keras.models import Model
import keras
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
scaler = StandardScaler()
X1 = scaler.fit_transform(X)


model = Sequential()
model.add(Dense(10,input_dim=X1.shape[1]))
model.add(Dense(5))
model.add(Dense(1))
model.add(Activation('linear'))

rmsprop = RMSprop(lr=0.01)
model.compile(loss='mse', optimizer=rmsprop)
history = model.fit(X1, y, epochs=1000,validation_split=0.33,verbose=False)
test1 = scaler.fit_transform(test)

pred = model.predict(test1).reshape(-1)


pred = np.expm1(pred)
prediction = pd.DataFrame({'Id':test_id, 'SalePrice':pred})
prediction.to_csv('neural_network.csv', index=False)
prediction.head()
from sklearn.tree import DecisionTreeRegressor



parameters = {'max_depth':range(3,10)}
regr = GridSearchCV(DecisionTreeRegressor(), parameters, n_jobs=4)
regr.fit(X, y)
y = regr.predict(test)
y = np.expm1(y)

prediction = pd.DataFrame({'Id':test_id, 'SalePrice':y})
prediction.to_csv('tree.csv', index=False)
prediction.head()