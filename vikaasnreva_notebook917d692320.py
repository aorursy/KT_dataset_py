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
!pip3 install pymysql
import pymysql
import pandas as pd
import numpy as np
import seaborn as sns
myDB = pymysql.connect(host='209.126.3.200', port=int(3306), 
                       user='mi_user', passwd='ce251CwE584##878', 
                       db='mi_dbmark', autocommit=True);
sql1 = "SELECT * from truecar_"
sql2 = "SELECT * from scrapeData_"

truecar = pd.read_sql_query(sql1, myDB)
scrapeData = pd.read_sql_query(sql2, myDB)
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
scrapeData['miles']=scrapeData['miles'].astype(int)
scrapeData['year']=scrapeData['year'].astype(int)
scrapeData['price']=scrapeData['price'].astype(int)
scrapeData['zip_code']=scrapeData['zip_code'].astype(int)
scrapeData['listing_id']=scrapeData['listing_id'].astype(str)
scrapeData['upc_product_code']=scrapeData['upc_product_code'].astype(str)
scrapeData['added']=scrapeData['added'].apply(pd.to_datetime)
#Changing type of objects to category
for n,c in scrapeData.items():
    if is_string_dtype(c): scrapeData[n] = c.astype('category').cat.as_ordered()
sns.countplot(y="make" ,data=scrapeData)
sns.set(rc={'figure.figsize':(20,15)})
sns.countplot(y="year" ,data=scrapeData)
sns.set(rc={'figure.figsize':(20,15)})
def encoder(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = scrapeData.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res
#function which will give numerical vlues to the categorical data
def numericalize(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = pd.Categorical(col).codes+1
def fix_missing(df, col, name, na_dict):   
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict
new_data,y,nas=encoder(scrapeData,"price")
new_data=new_data.drop(['url','upc_product_code','listing_id','refTable','added','color'],axis=1)
new_data.head()
x = new_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_log_error , mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2,random_state = 1)
def mape(y_true,y_pred):
    mape=np.mean(np.abs((y_true-y_pred)/y_true))*100
    return mape
from sklearn.ensemble import RandomForestRegressor
#developing a random forest
m=RandomForestRegressor(n_estimators=100, n_jobs=-1, min_samples_leaf=7,max_depth=100)
m.fit(X_train, y_train)
y_pred= m.predict(X_test)
print("RMSE :",np.sqrt(mean_squared_error(y_test, y_pred)))
print ("R2_Score : ", r2_score(y_pred, y_test))
