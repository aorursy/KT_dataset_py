import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
#creating dataframe for the required output

submission_MIX = pd.DataFrame()

submission_MIX['User_ID'] = df_test['User_ID']

submission_MIX['Product_ID'] = df_test['Product_ID']
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

df['Age'] = df['Age'].map({'0-17':0,'18-25':1,'26-35':2,'36-45':3,'46-50':4,'51-55':5,'55+':6})
df.head()
df_train = df[:550068]

df_test = df[550068:]
#df_train.drop(['User_ID','Product_ID'],axis=1,inplace=True)
from scipy import stats

z = np.abs(stats.zscore(df_train['Purchase']))



threshold = 2.33

np.where(z > 2.33)



df_train = df_train[(z<2.33)]
df_train.head()
df_train["Age_Count"] = df_train.groupby(['Age'])['Age'].transform('count')

age_count_dict = df_train.groupby(['Age']).size().to_dict()

df_test['Age_Count'] = df_test['Age'].apply(lambda x:age_count_dict.get(x,0))



df_train["Occupation_Count"] = df_train.groupby(['Occupation'])['Occupation'].transform('count')

occupation_count_dict = df_train.groupby(['Occupation']).size().to_dict()

df_test['Occupation_Count'] = df_test['Occupation'].apply(lambda x:occupation_count_dict.get(x,0))



df_train["User_ID_Count"] = df_train.groupby(['User_ID'])['User_ID'].transform('count')

userID_count_dict = df_train.groupby(['User_ID']).size().to_dict()

df_test['User_ID_Count'] = df_test['User_ID'].apply(lambda x:userID_count_dict.get(x,0))



df_train["Product_ID_Count"] = df_train.groupby(['Product_ID'])['Product_ID'].transform('count')

productID_count_dict = df_train.groupby(['Product_ID']).size().to_dict()

df_test['Product_ID_Count'] = df_test['Product_ID'].apply(lambda x:productID_count_dict.get(x,0))


df_train["Product_Category_1_Count"] = df_train.groupby(['Product_Category_1'])['Product_Category_1'].transform('count')

pc1_count_dict = df_train.groupby(['Product_Category_1']).size().to_dict()

df_test['Product_Category_1_Count'] = df_test['Product_Category_1'].apply(lambda x:pc1_count_dict.get(x,0))



df_train["Product_Category_2_Count"] = df_train.groupby(['Product_Category_2'])['Product_Category_2'].transform('count')

pc2_count_dict = df_train.groupby(['Product_Category_2']).size().to_dict()

df_test['Product_Category_2_Count'] = df_test['Product_Category_2'].apply(lambda x:pc2_count_dict.get(x,0))



df_train["Product_Category_3_Count"] = df_train.groupby(['Product_Category_3'])['Product_Category_3'].transform('count')

pc3_count_dict = df_train.groupby(['Product_Category_3']).size().to_dict()

df_test['Product_Category_3_Count'] = df_test['Product_Category_3'].apply(lambda x:pc3_count_dict.get(x,0))
df_train["User_ID_MinPrice"] = df_train.groupby(['User_ID'])['Purchase'].transform('min')

userID_min_dict = df_train.groupby(['User_ID'])['Purchase'].min().to_dict()

df_test['User_ID_MinPrice'] = df_test['User_ID'].apply(lambda x:userID_min_dict.get(x,0))



df_train["User_ID_MaxPrice"] = df_train.groupby(['User_ID'])['Purchase'].transform('max')

userID_max_dict = df_train.groupby(['User_ID'])['Purchase'].max().to_dict()

df_test['User_ID_MaxPrice'] = df_test['User_ID'].apply(lambda x:userID_max_dict.get(x,0))



df_train["User_ID_MeanPrice"] = df_train.groupby(['User_ID'])['Purchase'].transform('mean')

userID_mean_dict = df_train.groupby(['User_ID'])['Purchase'].mean().to_dict()

df_test['User_ID_MeanPrice'] = df_test['User_ID'].apply(lambda x:userID_mean_dict.get(x,0))


df_train["Product_ID_MinPrice"] = df_train.groupby(['Product_ID'])['Purchase'].transform('min')

productID_min_dict = df_train.groupby(['Product_ID'])['Purchase'].min().to_dict()

df_test['Product_ID_MinPrice'] = df_test['Product_ID'].apply(lambda x:productID_min_dict.get(x,0))



df_train["Product_ID_MaxPrice"] = df_train.groupby(['Product_ID'])['Purchase'].transform('max')

productID_max_dict = df_train.groupby(['Product_ID'])['Purchase'].max().to_dict()

df_test['Product_ID_MaxPrice'] = df_test['Product_ID'].apply(lambda x:productID_max_dict.get(x,0))



df_train["Product_ID_MeanPrice"] = df_train.groupby(['Product_ID'])['Purchase'].transform('mean')

productID_mean_dict = df_train.groupby(['Product_ID'])['Purchase'].mean().to_dict()

df_test['Product_ID_MeanPrice'] = df_test['Product_ID'].apply(lambda x:productID_mean_dict.get(x,0))
userID_25p_dict = df_train.groupby(['User_ID'])['Purchase'].apply(lambda x:np.percentile(x,25)).to_dict()

df_train['User_ID_25PercPrice'] = df_train['User_ID'].apply(lambda x:userID_25p_dict.get(x,0))

df_test['User_ID_25PercPrice'] = df_test['User_ID'].apply(lambda x:userID_25p_dict.get(x,0))



userID_75p_dict = df_train.groupby(['User_ID'])['Purchase'].apply(lambda x:np.percentile(x,75)).to_dict()

df_train['User_ID_75PercPrice'] = df_train['User_ID'].apply(lambda x:userID_75p_dict.get(x,0))

df_test['User_ID_75PercPrice'] = df_test['User_ID'].apply(lambda x:userID_75p_dict.get(x,0))



productID_25p_dict = df_train.groupby(['Product_ID'])['Purchase'].apply(lambda x:np.percentile(x,25)).to_dict()

df_train['Product_ID_25PercPrice'] = df_train['Product_ID'].apply(lambda x:productID_25p_dict.get(x,0))

df_test['Product_ID_25PercPrice'] = df_test['Product_ID'].apply(lambda x:productID_25p_dict.get(x,0))



productID_75p_dict = df_train.groupby(['Product_ID'])['Purchase'].apply(lambda x:np.percentile(x,75)).to_dict()

df_train['Product_ID_75PercPrice'] = df_train['Product_ID'].apply(lambda x:productID_75p_dict.get(x,0))

df_test['Product_ID_75PercPrice'] = df_test['Product_ID'].apply(lambda x:productID_75p_dict.get(x,0))



df_train["Product_Cat1_MinPrice"] = df_train.groupby(['Product_Category_1'])['Purchase'].transform('min')

pc1_min_dict = df_train.groupby(['Product_Category_1'])['Purchase'].min().to_dict()

df_test['Product_Cat1_MinPrice'] = df_test['Product_Category_1'].apply(lambda x:pc1_min_dict.get(x,0))



df_train["Product_Cat1_MaxPrice"] = df_train.groupby(['Product_Category_1'])['Purchase'].transform('max')

pc1_max_dict = df_train.groupby(['Product_Category_1'])['Purchase'].max().to_dict()

df_test['Product_Cat1_MaxPrice'] = df_test['Product_Category_1'].apply(lambda x:pc1_max_dict.get(x,0))



df_train["Product_Cat1_MeanPrice"] = df_train.groupby(['Product_Category_1'])['Purchase'].transform('mean')

pc1_mean_dict = df_train.groupby(['Product_Category_1'])['Purchase'].mean().to_dict()

df_test['Product_Cat1_MeanPrice'] = df_test['Product_Category_1'].apply(lambda x:pc1_mean_dict.get(x,0))





pc1_25p_dict = df_train.groupby(['Product_Category_1'])['Purchase'].apply(lambda x:np.percentile(x,25)).to_dict()

df_train['Product_Cat1_25PercPrice'] = df_train['Product_Category_1'].apply(lambda x:pc1_25p_dict.get(x,0))

df_test['Product_Cat1_25PercPrice'] = df_test['Product_Category_1'].apply(lambda x:pc1_25p_dict.get(x,0))



pc1_75p_dict = df_train.groupby(['Product_Category_1'])['Purchase'].apply(lambda x:np.percentile(x,75)).to_dict()

df_train['Product_Cat1_75PercPrice'] = df_train['Product_Category_1'].apply(lambda x:pc1_75p_dict.get(x,0))

df_test['Product_Cat1_75PercPrice'] = df_test['Product_Category_1'].apply(lambda x:pc1_75p_dict.get(x,0))





df_train["Product_Cat2_MinPrice"] = df_train.groupby(['Product_Category_2'])['Purchase'].transform('min')

pc2_min_dict = df_train.groupby(['Product_Category_2'])['Purchase'].min().to_dict()

df_test['Product_Cat2_MinPrice'] = df_test['Product_Category_2'].apply(lambda x:pc2_min_dict.get(x,0))





df_train["Product_Cat2_MaxPrice"] = df_train.groupby(['Product_Category_2'])['Purchase'].transform('max')

pc2_max_dict = df_train.groupby(['Product_Category_2'])['Purchase'].max().to_dict()

df_test['Product_Cat2_MaxPrice'] = df_test['Product_Category_2'].apply(lambda x:pc2_max_dict.get(x,0))



df_train["Product_Cat2_MeanPrice"] = df_train.groupby(['Product_Category_2'])['Purchase'].transform('mean')

pc2_mean_dict = df_train.groupby(['Product_Category_2'])['Purchase'].mean().to_dict()

df_test['Product_Cat2_MeanPrice'] = df_test['Product_Category_2'].apply(lambda x:pc2_mean_dict.get(x,0))



pc2_25p_dict = df_train.groupby(['Product_Category_2'])['Purchase'].apply(lambda x:np.percentile(x,25)).to_dict()

df_train['Product_Cat2_25PercPrice'] = df_train['Product_Category_2'].apply(lambda x:pc2_25p_dict.get(x,0))

df_test['Product_Cat2_25PercPrice'] = df_test['Product_Category_2'].apply(lambda x:pc2_25p_dict.get(x,0))



pc2_75p_dict = df_train.groupby(['Product_Category_2'])['Purchase'].apply(lambda x:np.percentile(x,75)).to_dict()

df_train['Product_Cat2_75PercPrice'] = df_train['Product_Category_2'].apply(lambda x:pc2_75p_dict.get(x,0))

df_test['Product_Cat2_75PercPrice'] = df_test['Product_Category_2'].apply(lambda x:pc2_75p_dict.get(x,0))





df_train["Product_Cat3_MinPrice"] = df_train.groupby(['Product_Category_3'])['Purchase'].transform('min')

pc3_min_dict = df_train.groupby(['Product_Category_3'])['Purchase'].min().to_dict()

df_test['Product_Cat3_MinPrice'] = df_test['Product_Category_3'].apply(lambda x:pc3_min_dict.get(x,0))



df_train["Product_Cat3_MaxPrice"] = df_train.groupby(['Product_Category_3'])['Purchase'].transform('max')

pc3_max_dict = df_train.groupby(['Product_Category_3'])['Purchase'].max().to_dict()

df_test['Product_Cat3_MaxPrice'] = df_test['Product_Category_3'].apply(lambda x:pc3_max_dict.get(x,0))



df_train["Product_Cat3_MeanPrice"] = df_train.groupby(['Product_Category_3'])['Purchase'].transform('mean')

pc3_mean_dict = df_train.groupby(['Product_Category_3'])['Purchase'].mean().to_dict()

df_test['Product_Cat3_MeanPrice'] = df_test['Product_Category_3'].apply(lambda x:pc3_mean_dict.get(x,0))



pc3_25p_dict = df_train.groupby(['Product_Category_3'])['Purchase'].apply(lambda x:np.percentile(x,25)).to_dict()

df_train['Product_Cat3_25PercPrice'] = df_train['Product_Category_3'].apply(lambda x:pc3_25p_dict.get(x,0))

df_test['Product_Cat3_25PercPrice'] = df_test['Product_Category_3'].apply(lambda x:pc3_25p_dict.get(x,0))



pc3_75p_dict = df_train.groupby(['Product_Category_3'])['Purchase'].apply(lambda x:np.percentile(x,75)).to_dict()

df_train['Product_Cat3_75PercPrice'] = df_train['Product_Category_3'].apply(lambda x:pc3_75p_dict.get(x,0))

df_test['Product_Cat3_75PercPrice'] = df_test['Product_Category_3'].apply(lambda x:pc3_75p_dict.get(x,0))
df_train.head()
#label encoding User ID and Product ID

from sklearn.preprocessing import LabelEncoder

cat_columns_list = ["User_ID", "Product_ID"]

for var in cat_columns_list:

    lb = LabelEncoder()

    full_var_data = pd.concat((df_train[var],df_test[var]),axis=0).astype('str')

    temp = lb.fit_transform(np.array(full_var_data))

    df_train[var] = lb.transform(np.array( df_train[var] ).astype('str'))

    df_test[var] = lb.transform(np.array( df_test[var] ).astype('str'))
#df_train.drop(['User_ID','Product_ID'],axis=1,inplace=True)
from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel
X = df_train.drop('Purchase',axis=1)

y = df_train['Purchase']
#from sklearn.model_selection import train_test_split

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
model = SelectFromModel(Lasso(alpha=0.005,random_state=0))
model.fit(X,y)
model.get_support()
df_1 = df_test.copy()

#df_test.drop(['User_ID','Product_ID'],axis=1,inplace=True)

df_test.drop(['Purchase'],axis=1,inplace=True)
from sklearn.model_selection import RandomizedSearchCV
# # number of trees

# n_estimators = [int(x) for x in np.linspace(start=100,stop=200,num=5)]

# # number of fetaures to consider at every split

# max_features = ['sqrt']

# # max level in tree

# max_depth = [int(x) for x in np.linspace(5,10,num=5)]

# # min sample required for split

# min_samples_split = [10,15,100]

# # min samples at each leaf node

# min_samples_leaf = [5,10]
# # create a random grid

# random_grid = {'n_estimators': n_estimators}

# #               'max_features': max_features}

# #               'max_depth': max_depth}

# #               'min_samples_split': min_samples_split,

# #                'min_samples_leaf': min_samples_leaf}

# print(random_grid)
# # use the random search to find best hyper parameters

# # first create a base model to tune

# from sklearn.ensemble import RandomForestRegressor

# rf = RandomForestRegressor()

# # search of parameters

# rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=1,cv=5,verbose=2,random_state=42,n_jobs=1)
#rf_random.fit(X,y)
# y_pred = rf_random.predict(df_test)

# submission = pd.DataFrame({

#         "Purchase":y_pred,

#         "User_ID": df_1["User_ID"],

#         "Product_ID": df_1["Product_ID"]

        

#     })



# submission.to_csv('Black_Friday_Sales_submission.csv', index=False)
# import xgboost as xgb



# xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.2,

#                 max_depth = 10, alpha = 15, n_estimators = 1000)

# xg_reg.fit(X,y)

# y_pred_XGB = xg_reg.predict(df_test)



# submission_XGB = pd.DataFrame({

#         "Purchase":y_pred_XGB,

#         "User_ID": df_1["User_ID"],

#         "Product_ID": df_1["Product_ID"]

        

#     })



# submission_XGB.to_csv('Black_Friday_Sales_submission_XGB.csv', index=False)
alist = ['User_ID',

'Product_ID',

'Gender',

'Age',

'Occupation',

'City_Category',

'Stay_In_Current_City_Years',

'Marital_Status',

'Product_Category_1',

'Product_Category_2',

'Product_Category_3',

'Age_Count',

'Occupation_Count',

'Product_Category_1_Count',

'Product_Category_2_Count',

'Product_Category_3_Count',

'User_ID_Count',

'Product_ID_Count']



         

blist = ['User_ID_MinPrice',

'User_ID_MaxPrice',

'User_ID_MeanPrice',

'Product_ID_MinPrice',

'Product_ID_MaxPrice',

'Product_ID_MeanPrice']





clist = ['User_ID_25PercPrice',

'User_ID_75PercPrice',

'Product_ID_25PercPrice',

'Product_ID_75PercPrice',

'Product_Cat1_MinPrice',

'Product_Cat1_MaxPrice',

'Product_Cat1_MeanPrice',

'Product_Cat1_25PercPrice',

'Product_Cat1_75PercPrice',

'Product_Cat2_MinPrice',

'Product_Cat2_MaxPrice',

'Product_Cat2_MeanPrice',

'Product_Cat2_25PercPrice',

'Product_Cat2_75PercPrice',

'Product_Cat3_MinPrice',

'Product_Cat3_MaxPrice',

'Product_Cat3_MeanPrice',

'Product_Cat3_25PercPrice',

'Product_Cat3_75PercPrice']
#LGB model 1 dataframe

train1 = X[alist+blist]

test1 = df_test[alist+blist]



#LGB model 2 dataframe 

train2 = X[alist+clist]

test2 = df_test[alist+clist]
import lightgbm as lgb

train_data=lgb.Dataset(train1,label=y)

#define parameters

params = {'n_estimators':205,'learning_rate':0.1,'max_depth': 10,'num_leaves':200,'min_data_in_leaf':10,'max_bin':200}

model= lgb.train(params, train_data, 200) 

y_pred_LGB1=model.predict(test1)



submission_LGB1 = pd.DataFrame({

        "Purchase":y_pred_LGB1,

        "User_ID": df_1["User_ID"],

        "Product_ID": df_1["Product_ID"]

        

    })



submission_LGB1.to_csv('Black_Friday_Sales_submission_LGB1.csv', index=False)
import lightgbm as lgb

train_data=lgb.Dataset(train2,label=y)

#define parameters

params = {'n_estimators':205,'learning_rate':0.1,'max_depth': 10,'num_leaves':200,'min_data_in_leaf':10,'max_bin':200}

model= lgb.train(params, train_data, 200) 

y_pred_LGB2=model.predict(test2)



submission_LGB2 = pd.DataFrame({

        "Purchase":y_pred_LGB2,

        "User_ID": df_1["User_ID"],

        "Product_ID": df_1["Product_ID"]

        

    })



submission_LGB2.to_csv('Black_Friday_Sales_submission_LGB2.csv', index=False)
# ensembled prediction over splitted test data

ensembled_prediction = (0.5*(y_pred_LGB1)+0.5*(y_pred_LGB2))



submission_MIX["Purchase"] = ensembled_prediction



submission_MIX.to_csv('Black_Friday_Sales_submission_MIX.csv', index=False)