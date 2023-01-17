# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv",low_memory=False,parse_dates=["saledate"])

df.info()
df.isna().sum()
df.saledate.dtype
df.sort_values(by=["saledate"], inplace=True, ascending=True)
test_df = pd.read_csv("/kaggle/input/bluebook-for-bulldozers/Test.csv",parse_dates=['saledate'],low_memory=False)
# sorting df according to the saledate
df.sort_values(by='saledate',inplace=True)
test_df.isna().sum()
Concat = pd.concat((df,test_df),axis = 0).reset_index(drop=True)

for label,content in Concat.items() :
    if pd.api.types.is_object_dtype(content):
        Concat[label] = content.astype('category')

Concat['year'] = Concat.saledate.dt.year
Concat['month']= Concat.saledate.dt.month
Concat['day']= Concat.saledate.dt.day
cat=[] # list for storing all columns with 'cstegory' dtype
cat_missing = [] # list for storing columns with 'category' dtype and having missing values
num_missing = [] # list for storing columns with 'numerical' dtype and having missing values
for label,content in Concat.items():
    
    if pd.api.types.is_numeric_dtype(content): # checking for numerical features
        if content.isna().sum() > 0: # checking if the feature has any missing values
            Concat[f'{label}_ismissing'] = content.isna()
            num_missing.append(label)
            
    if pd.api.types.is_categorical_dtype(content): # checking for categorical features
        cat.append(label) 
        if content.isna().sum() > 0: # checking if the feature has any missing values
            Concat[f'{label}_ismissing'] = content.isna()
            cat_missing.append(label)
            
cat_not_missing = list(set(cat) - set(cat_missing))
# For missing values in categorical datatype, by default `-1` is assigned for its code, so adding 1 before creating new column
Concat[cat_missing] = Concat[cat_missing].apply(lambda i : i.cat.codes+1)

# For features with no missing values, simply assigning code
Concat[cat_not_missing] = Concat[cat_not_missing].apply(lambda i : i.cat.codes)
train_df = Concat.loc[Concat.saledate.dt.year < 2012, :].drop('saledate', axis=1)

valid_df = Concat.loc[Concat.saledate <= pd.Timestamp(
    year=2012, month=4, day=30)].loc[Concat.saledate >= pd.Timestamp(year=2012, month=1, day=1)].drop('saledate', axis=1)

test_df = Concat.loc[Concat.saledate >=
                     pd.Timestamp(year=2012, month=4, day=30), :].drop(['SalePrice','saledate'], axis=1)
train_df.shape
test_df.shape
valid_df.shape
train_df[num_missing].isna().sum()
valid_df[num_missing].isna().sum()
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy='median')
transformer = ColumnTransformer(transformers=[('num_missing',num_imputer,train_df.columns)],remainder='passthrough',)

train_df_filled = transformer.fit_transform(train_df) # fitting on training data 
valid_df_filled = transformer.transform(valid_df) # transforming test based on training data to avoid data leakage

train_df_filled = pd.DataFrame(train_df_filled,columns=train_df.columns)
valid_df_filled = pd.DataFrame(valid_df_filled,columns=valid_df.columns)
train_df_filled[num_missing].isna().sum()
valid_df_filled[num_missing].isna().sum()
from sklearn.metrics import mean_squared_error,mean_squared_log_error,mean_absolute_error,make_scorer
def evaluate(model, X_valid, y_valid):
    y_pred = model.predict(X_valid)
    scores = {
        'R2': model.score(X_valid, y_valid),
        'MAE': mean_absolute_error(y_valid, y_pred),
        'RMLE': np.sqrt(mean_squared_log_error(y_valid, y_pred))}
    return(scores)
X_train_filled,y_train_filled = train_df_filled.drop(['SalePrice'],axis=1),train_df_filled.SalePrice 
X_valid_filled,y_valid_filled = valid_df_filled.drop(['SalePrice'],axis=1),valid_df_filled.SalePrice
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_jobs=-1,n_estimators=100,max_depth=4)
rfr.fit(X_train_filled,y_train_filled)
evaluate(rfr,X_valid_filled,y_valid_filled)
test_df.columns
test_df = test_df.dropna()
predict = rfr.predict(test_df)
submission = pd.DataFrame(predict)
submission.to_csv('submission.csv', index=False)

