import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
df = pd.read_csv("../input/kc_house_data.csv")
df.head()
print(df.shape)
print('------------------------')
print(df.nunique())
print('------------------------')
print(df.dtypes)
df.isnull().sum()
sum((df['id'].value_counts()>=2)*1)
plt.hist(df['price'],bins=100)
plt.show()
df.describe()['price']
df.corr().style.background_gradient().format('{:.2f}')
for i in df.columns:
    if (i != 'price') & (i != 'date'):
        df[[i,'price']].plot(kind='scatter',x=i,y='price')
from sklearn.linear_model import LinearRegression
X_1 = df.drop(['price','id','date','yr_renovated','zipcode'],axis=1)
y_1 = df['price']

#trainとtestを3：1で分割（defo値）。random_stateは年齢を入れた:d
X_train_1,X_test_1,y_train_1,y_test_1 = train_test_split(X_1,y_1,random_state=50)

regr_train_1=LinearRegression(fit_intercept=True).fit(X_train_1,y_train_1)
y_pred_1 = regr_train_1.predict(X_test_1)
for i, coef in enumerate(regr_train_1.coef_):
    print(i,X_train_1.columns[i],':',coef)
#MAE = mean_absolute_error(y_test,y_pred)
#MSE = mean_squared_error(y_test,y_pred)

MAE_1 = mean_absolute_error(y_test_1,y_pred_1)
MSE_1 = mean_squared_error(y_test_1,y_pred_1)

print('MAE_1:',MAE_1,'/','MSE_1:',MSE_1)
df.date.head()
pd.to_datetime(df.date).map(lambda x:'dow'+str(x.weekday())).head()
pd.to_datetime(df.date).map(lambda x:'month'+str(x.month)).head()
df['dow'] = pd.to_datetime(df.date).map(lambda x:'dow'+str(x.weekday()))
df['month'] = pd.to_datetime(df.date).map(lambda x:'month'+str(x.month))
pd.get_dummies(df['dow']).head()
pd.get_dummies(df['month']).head()
df['zipcode'].astype(str).map(lambda x:x).head()
df['zipcode_str'] = df['zipcode'].astype(str).map(lambda x:'zip_'+x)
pd.get_dummies(df['zipcode_str']).head()
df['zipcode_str'] = df['zipcode'].astype(str).map(lambda x:'zip_'+x)
df_en = pd.concat([df,pd.get_dummies(df['zipcode_str'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df.dow)],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df.month)],axis=1)
df_en_fin = df_en.drop(['id','date','zipcode','month','dow','zipcode_str',],axis=1)
#データ形状とユニーク数を確認。説明変数は107.
print(df_en_fin.shape)
print('------------------------')
print(df_en_fin.nunique())
df_en_fin.head()
X = df_en_fin.drop(['price'],axis=1)
y = df_en_fin['price']
regr = LinearRegression(fit_intercept=True).fit(X,y)
model_2 = regr.score(X,y)
for i, coef in enumerate(regr.coef_):
    print(X.columns[i],':',coef)
df_vif = df_en_fin.drop(["price"],axis=1)
for cname in df_vif.columns:  
    y=df_vif[cname]
    X=df_vif.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    print(cname,":" ,1/(1-np.power(rsquared,2)))
df_vif = df_en_fin.drop(["price"],axis=1)
for cname in df_vif.columns:  
    y=df_vif[cname]
    X=df_vif.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    #print(cname,":" ,1/(1-np.power(rsquared,2)))
    if rsquared > 1. -1e-10:
        print(cname,X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])
df_en_fin = df_en_fin.drop(['sqft_above','zip_98001','month1','dow1'],axis=1)

df_vif = df_en_fin.drop(["price"],axis=1)
df_vif.shape
for cname in df_vif.columns:  
    y=df_vif[cname]
    X=df_vif.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    print(cname,":" ,1/(1-np.power(rsquared,2)))
    if rsquared > 1. -1e-10:
        print(cname,X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])
X = df_en_fin.drop(['price'],axis=1)
y = df_en_fin['price']
df_en_fin.shape
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=50)
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest,f_regression

N = len(X)

aic_array = np.zeros((len(X.columns)-1),float)#AICの値を格納する配列（重回帰なので101のゼロ配列）
mae_array = np.zeros((len(X.columns)-1),float)#MAEの値を格納する配列。同上。

#一連のSelectKBestの流れは覚えておく
#SelectKBest　各説明変数と目的変数の関係を計算し、最も高い確信度で関連特徴量を選択
#get_support()    各特徴量を選択したか否かをbooleanで取得（True or False）
#transpose()[sup] 選択した特徴量の列のみ取得

for k in range(1,len(X.columns)):
    skb = SelectKBest(f_regression,k=k).fit(X_train,y_train)
    sup = skb.get_support()
    X_selected = X.transpose()[sup].transpose()
    regr = linear_model.LinearRegression()
    model = regr.fit(X_selected,y)
    met = mean_absolute_error(model.predict(X_selected),y)
    aic = N*np.log((met**2).sum()/N) + 2*k
    
    #aicの値を配列に格納
    #最初のk-1は0（1-1=0）。0～101個（重回帰なので）。pythonは0に。k＝1は挙動せず、k-2だと最終列に最初の数値が来る。
    #print(aic_array)で試行するとよくわかる。
    
    aic_array[k-1] = aic 
    mae_array[k-1] = met
    print('k:',k,'MAE:',met,'AIC:',aic)
    
print('--------------------------------------------------------')
print('AICの最小値のkは'+str(np.argmin(aic_array)+1)+'番目です。')

plt.plot(range(1,102), aic_array)
plt.xlabel("The number of explanatory variables")
plt.title("AIC")
plt.show()

plt.plot(range(1,102), mae_array)
plt.xlabel("The number of explanatory variables")
plt.title("MAE")
plt.show()
reg = linear_model.LinearRegression(fit_intercept=True).fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)

mse_reg = mean_squared_error(y_test, y_pred_reg)
mae_reg = mean_absolute_error(y_test, y_pred_reg)
print('MSE:{:.3f}'.format(mse_las))
print('RMSE:{:.3f}'.format(np.sqrt(mse_las)))
print('MAE:{:.3f}'.format(mae_las))
#一応、相関
print("correlation:{:.4f}".format(np.corrcoef(y_test,y_pred_reg)[0,1]))
from sklearn.linear_model import Lasso

X = df_en_fin.drop(['price'],axis=1)
y = df_en_fin['price']
lasso = Lasso(fit_intercept=True).fit(X,y)
#モデルのハイパラメータの最適化を行う際、ランダムに行うのを「ランダムサーチ」と言うのに対し
#全組合せで行うのを「グリッドサーチ」と言う。そのメソッドがGridSearchCV。
#モデルのハイパラとはday2資料P.26参照
#https://qiita.com/SE96UoC5AfUt7uY/items/c81f7cea72a44a7bfd3a
#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

from sklearn.model_selection import train_test_split, GridSearchCV

#iloc[:,1:]は、（df_en_finの）データセットの行と列の番号を指定する。
#この場合、行は全て、列は1列目から全て。列が1列目なのは0列目が'price'だから。
#print(df_en_fin.iloc[:,1:])ですぐわかる。ちなみにvaluesは値のこと。

X, y = df_en_fin.iloc[:,1:].values, df_en_fin['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

param_grid = {'alpha':[1e-4,1e-3, 1e-2, 1e-1,1,5,10]}
cv = GridSearchCV(Lasso(),param_grid=param_grid,cv=5)
#cv.fit(X_test,y_test)
#訓練データでCVするのが正しい。
cv.fit(X_train,y_train)
cv.best_params_
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_pred_las = cv.best_estimator_.predict(X_test)
mse_las = mean_squared_error(y_test, y_pred_las)
mae_las = mean_absolute_error(y_test, y_pred_las)

print('MSE:{:.3f}'.format(mse_las))
print('RMSE:{:.3f}'.format(np.sqrt(mse_las)))
print('MAE:{:.3f}'.format(mae_las))
#MAEを比較
if mae_reg > mae_las:
    result = 'mae_las'
else:
    result = 'mae_reg'

text = f'{result}'
print(text)
print('MSE_decrease:{:.2f}'.format(1- mse_reg / MSE_1),'/','MAE_decrease:{:.2f}'.format(1-mae_reg / MAE_1))