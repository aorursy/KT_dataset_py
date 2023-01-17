import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.shape
df_test.shape
df = pd.concat([df_train,df_test])
df.shape
df.head()
df.info()
df.describe()
df['Marital_Status'].unique()
a = df.groupby('Marital_Status')['Purchase'].mean()
a.plot.bar()
df['Occupation'].unique()
a = df.groupby('Occupation')['Purchase'].mean()
a.plot.bar()
df['Gender'].unique()
a = df.groupby('Gender')['Purchase'].mean()
a.plot.bar()
df['Product_Category_1'].unique()
a = df.groupby('Product_Category_1')['Purchase'].count()
a.plot.bar()
df['Product_Category_2'].unique()
a = df.groupby('Product_Category_2')['Purchase'].count()
a.plot.bar()
df['Product_Category_3'].unique()
a = df.groupby('Product_Category_3')['Purchase'].count()
a.plot.bar()
df['Age'].unique()
a=df.groupby('Age')['Purchase'].mean()
a.plot.bar()
df['City_Category'].unique()
a=df.groupby('City_Category')['Purchase'].mean()
a.plot.bar()
df['Stay_In_Current_City_Years'].unique()
a = df.groupby('Stay_In_Current_City_Years')['Purchase'].mean()
a.plot.bar()
#df['Product_Category_1'] = df['Product_Category_1'].astype("O")
df['Product_Category_2'] = df['Product_Category_2'].astype("O")
df['Product_Category_3'] = df['Product_Category_3'].astype("O")
#df['Product_Category_1'] = df['Product_Category_1'].fillna(df['Product_Category_1'].mode()[0])
df['Product_Category_2'] = df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])
df['Product_Category_3'] = df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])
df['Product_Category_2'] = df['Product_Category_2'].astype("int")
df['Product_Category_3'] = df['Product_Category_3'].astype("int")
df.info()
df['Gender'] = df['Gender'].map({'F':0,'M':1})
df['City_Category'] = df['City_Category'].map({'A':0,'B':1,'C':2})
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].map({'0':0,'1':1,'2':2,'3':3,'4+':4})
df['Age'] = df['Age'].map({'0-17':0,'18-25':0,'26-35':1,'36-45':1,'46-50':1,'51-55':2,'55+':2})
df.head()
df_train = df[:550068]
df_test = df[550068:]
df_train.drop(['User_ID','Product_ID'],axis=1,inplace=True)
from scipy import stats
z = np.abs(stats.zscore(df_train['Purchase']))

threshold = 2.33
np.where(z > 2.33)

df_train = df_train[(z<2.33)]
df_train.head()
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
X = df_train.drop('Purchase',axis=1)
y = df_train['Purchase']
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
model = SelectFromModel(Lasso(alpha=0.005,random_state=0))
model.fit(X,y)
model.get_support()
from sklearn.model_selection import RandomizedSearchCV
# number of trees
n_estimators = [int(x) for x in np.linspace(start=100,stop=200,num=5)]
# number of fetaures to consider at every split
max_features = ['sqrt']
# max level in tree
max_depth = [int(x) for x in np.linspace(5,10,num=5)]
# min sample required for split
min_samples_split = [10,15,100]
# min samples at each leaf node
min_samples_leaf = [5,10]
# create a random grid
random_grid = {'n_estimators': n_estimators}
#               'max_features': max_features}
#               'max_depth': max_depth}
#               'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf}
print(random_grid)
# use the random search to find best hyper parameters
# first create a base model to tune
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
# search of parameters
rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=1,cv=5,verbose=2,random_state=42,n_jobs=1)
rf_random.fit(X,y)
df_1 = df_test.copy()
df_test.drop(['User_ID','Product_ID'],axis=1,inplace=True)
df_test.drop(['Purchase'],axis=1,inplace=True)
y_pred = rf_random.predict(df_test)
submission = pd.DataFrame({
        "Purchase":y_pred,
        "User_ID": df_1["User_ID"],
        "Product_ID": df_1["Product_ID"]
        
    })

submission.to_csv('Black_Friday_Sales_submission.csv', index=False)
import xgboost as xgb

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.2,
                max_depth = 10, alpha = 15, n_estimators = 1000)

xg_reg.fit(X,y)

y_pred_XGB = xg_reg.predict(df_test)

submission_XGB = pd.DataFrame({
        "Purchase":y_pred_XGB,
        "User_ID": df_1["User_ID"],
        "Product_ID": df_1["Product_ID"]
        
    })

submission_XGB.to_csv('Black_Friday_Sales_submission_XGB.csv', index=False)
import lightgbm as lgb
train_data=lgb.Dataset(X,label=y)
#define parameters
params = {'learning_rate':0.2,'max_depth': 10,'num_leaves':200,'min_data_in_leaf':10,'max_bin':200}
model= lgb.train(params, train_data, 200) 
y_pred_LGB=model.predict(df_test)
submission_LGB = pd.DataFrame({
        "Purchase":y_pred_LGB,
        "User_ID": df_1["User_ID"],
        "Product_ID": df_1["Product_ID"]
        
    })

submission_LGB.to_csv('Black_Friday_Sales_submission_LGB.csv', index=False)
#from sklearn import metrics
#print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
#print('MSE:', metrics.mean_squared_error(y_test, y_pred))
#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print("R2 Score:", metrics.r2_score(y_test, y_pred))
# ensembled prediction over splitted test data
ensembled_prediction = (0.33*(y_pred)+0.33*(y_pred_LGB)+0.33*(y_pred_XGB))
submission_MIX = pd.DataFrame({
        "Purchase":ensembled_prediction,
        "User_ID": df_1["User_ID"],
        "Product_ID": df_1["Product_ID"]
        
    })

submission_MIX.to_csv('Black_Friday_Sales_submission_MIX.csv', index=False)