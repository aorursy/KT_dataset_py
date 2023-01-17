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
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
pd.set_option('display.max_columns', 50)
def rmspe(a,b):
    '''
    a : 正解の配列
    b : 予想の配列
    '''
    if len(a)!=len(b):
        raise Exception('Error!')
    
    tmp = 0
    n = len(a)
    
    
    
    for i in range(len(a)):
        if a[i]==0:
            n = n-1
        else:
            tmp = tmp + ((b[i]-a[i])/a[i])**2
    print("評価値：",np.sqrt(tmp/n))
    return np.sqrt(tmp/n)
##01_前回

# Salesに上限を設定する
max_sale = 100000

#reading data files
store_df=pd.read_csv("../input/rossmann-store-sales/store.csv", low_memory=False)
store_df= pd.get_dummies(store_df,columns=["StoreType","Assortment"])

# train_dfとtest_dfを縦に結合
train_df=pd.read_csv("../input/rossmann-store-sales/train.csv", low_memory=False)
test_df=pd.read_csv("../input/rossmann-store-sales/test.csv", low_memory=False)
train_df['is_train']=True
test_df['is_train'] =False
df = pd.concat([train_df.drop(columns=['Sales']),test_df.drop(columns=['Id'])]).reset_index(drop=True).copy()

# 平均売上の算出(trainのみ使用)
dates1 = (df['Date'] >='2013-01-02') & (df['Date'] < '2014-07-01')
dates2 = (df['Date'] >='2015-01-01') & (df['Date'] < '2015-06-01')
dates12 = dates1 | dates2
except_sunday = (df['DayOfWeek'] != 7)
only_open = (df['Open']==1)
df_for_mean = train_df[dates12 & except_sunday & only_open].reset_index().copy()
df_for_mean['Sales'] = df_for_mean['Sales'].apply(lambda x: np.min([x,max_sale]))
store_mean_sales = df_for_mean.groupby('Store').mean()['Sales']

# dataframeに平均売上を付与
df['平均売上']   = df['Store'].apply(lambda x : store_mean_sales.loc[x])
df['平均売上log']= df['平均売上'].apply(lambda x: np.log(x) if x!=0 else -1)

df=df.merge(store_df,on=["Store"],how="left")

#競合店の距離は補完
df["CompetitionDistance"]=df["CompetitionDistance"].fillna(df["CompetitionDistance"].mode()[0])


#年月日や週番号を入れる
df["Date"] =pd.to_datetime(df["Date"])
df["Year"] =df["Date"].dt.year
df["Month"]=df["Date"].dt.month
df["Day"]  =df["Date"].dt.day
df["Week"] =df["Date"].dt.week%4


# 翌日お休みフラグ
df = df.sort_values(['Store', 'Date']).reset_index(drop=False)
date_column = df.columns.get_loc('Date')
open_column = df.columns.get_loc('Open')

before_holiday=[]
for i in range(len(df)-1):
    if df.iat[i+1,date_column]==df.iat[i,date_column]+ datetime.timedelta(days=1) and df.iat[i+1,open_column]==0:
        before_holiday.append(1)
    else:
        before_holiday.append(0)
        
before_holiday.append(0)
df['before_holiday'] = before_holiday

df=df.sort_values(['index']).reset_index(drop=True).drop(columns=['index'])

##HolidayPerWeekの追加
Holiday_Year_Month_Week_df=pd.DataFrame({"HolidayPerWeek":df["SchoolHoliday"],"Week":df["Week"],"Month":df["Month"],"Year":df["Year"],"Date":df["Date"]})
Holiday_Year_Month_Week_df=Holiday_Year_Month_Week_df.drop_duplicates(subset=['Date'])
Holiday_Year_Month_Week_df=Holiday_Year_Month_Week_df.groupby(["Year","Month","Week"]).sum()
Holiday_Year_Month_Week_df

df=df.merge(Holiday_Year_Month_Week_df, on=["Year","Month","Week"],how="left")

# promoCountPerWeekの追加
promo_time_df=pd.DataFrame({"PromoCountPerWeek":df["Promo"],"Year":df["Year"],"Month":df["Month"],"Week":df["Week"],"Date":df["Date"]})
promo_time_df=promo_time_df.drop_duplicates(subset=['Date'])
promo_time_df=promo_time_df.groupby(["Year","Month","Week"]).sum()

df=df.merge(promo_time_df,on=["Year","Month","Week"], how="left")

# 祝日
df["StateHoliday"]=np.where(df["StateHoliday"] == '0' ,0,1)

df  = pd.get_dummies(df ,columns=["Month"])
df  = pd.get_dummies(df,columns=["DayOfWeek"])

# dfのcolumnsの一覧を確認

features = ["Open","Promo",'平均売上log']
features.extend(["Assortment_a","Assortment_b"])        
#features.extend(["PromoCountPerWeek","HolidayPerWeek","SchoolHoliday"])
features.extend(["PromoCountPerWeek","HolidayPerWeek"])
features.extend(['StoreType_a', 'StoreType_b','StoreType_c'])
features.extend(['before_holiday',])
features.extend(['Month_1','Month_2','Month_3','Month_4' ,'Month_5' ,'Month_6'])
features.extend(['Month_7','Month_10','Month_11','Month_12'])
features.extend(['Month_8','Month_9',])
features.extend(['DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3','DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7'])
features.extend(['DayOfWeek_6', 'DayOfWeek_7'])
features.extend(["CompetitionDistance"])


train_x  = df[(df['is_train']==True) & (df['Open']==1)][features]
test_x   = df[df['is_train']==False][features].fillna(1)
train_y  = train_df['Sales']



train_df    = train_df[train_df['Open']==1]
train_sales = train_df['Sales' ]
train_df['logSales'] = train_df['Sales'].apply(lambda x: np.log(np.min([x,max_sale])) if x!=0 else -1)
train_y  = train_df['logSales']

scaler = StandardScaler()
scaler.fit(train_x)

train_x  = scaler.transform(train_x)
test_x   = scaler.transform(test_x)

model=LinearRegression()
model.fit(train_x,train_y)

predict_train = model.predict(train_x)

print(rmspe(train_sales.tolist(),np.exp(predict_train)))

predict_test  = model.predict(test_x)
predict_test = np.round(np.exp(predict_test))
##02_day mean

# Salesに上限を設定する
max_sale = 100000

#reading data files
store_df=pd.read_csv("../input/rossmann-store-sales/store.csv", low_memory=False)
store_df= pd.get_dummies(store_df,columns=["StoreType","Assortment"])

# train_dfとtest_dfを縦に結合
train_df=pd.read_csv("../input/rossmann-store-sales/train.csv", low_memory=False)
test_df=pd.read_csv("../input/rossmann-store-sales/test.csv", low_memory=False)
train_df['is_train']=True
test_df['is_train'] =False
df = pd.concat([train_df.drop(columns=['Sales']),test_df.drop(columns=['Id'])]).reset_index(drop=True).copy()

# 平均売上の算出(trainのみ使用)
dates1 = (df['Date'] >='2013-01-02') & (df['Date'] < '2014-07-01')
dates2 = (df['Date'] >='2015-01-01') & (df['Date'] < '2015-06-01')
dates12 = dates1 | dates2
except_sunday = (df['DayOfWeek'] != 7)
only_open = (df['Open']==1)
df_for_mean = train_df[dates12 & except_sunday & only_open].reset_index().copy()
df_for_mean['Sales'] = df_for_mean['Sales'].apply(lambda x: np.min([x,max_sale]))
store_mean_sales = df_for_mean.groupby('Store').mean()['Sales']

#Day平均を入れる
df_for_mean["Date"] =pd.to_datetime(df_for_mean["Date"])
df_for_mean["Year"] =df_for_mean["Date"].dt.year
df_for_mean["Month"]=df_for_mean["Date"].dt.month
df_for_mean["Day"]  =df_for_mean["Date"].dt.day
df_for_mean["Week"] =df_for_mean["Date"].dt.week%4
Day_mean_sales = df_for_mean.groupby('Day').mean()['Sales']

#年月日や週番号を入れる
df["Date"] =pd.to_datetime(df["Date"])
df["Year"] =df["Date"].dt.year
df["Month"]=df["Date"].dt.month
df["Day"]  =df["Date"].dt.day
df["Week"] =df["Date"].dt.week%4


# dataframeに平均売上を付与
df['平均売上']   = df['Store'].apply(lambda x : store_mean_sales.loc[x])
df['平均売上log']= df['平均売上'].apply(lambda x: np.log(x) if x!=0 else -1)

# dataframeに平均売上を付与
df['平均売上(Day)']   = df['Day'].apply(lambda x : Day_mean_sales.loc[x])
df['平均売上log(Day)']= df['平均売上(Day)'].apply(lambda x: np.log(x) if x!=0 else -1)


df=df.merge(store_df,on=["Store"],how="left")


#競合店の距離は補完
df["CompetitionDistance"]=df["CompetitionDistance"].fillna(df["CompetitionDistance"].mode()[0])


# 翌日お休みフラグ
df = df.sort_values(['Store', 'Date']).reset_index(drop=False)
date_column = df.columns.get_loc('Date')
open_column = df.columns.get_loc('Open')

before_holiday=[]
for i in range(len(df)-1):
    if df.iat[i+1,date_column]==df.iat[i,date_column]+ datetime.timedelta(days=1) and df.iat[i+1,open_column]==0:
        before_holiday.append(1)
    else:
        before_holiday.append(0)
        
before_holiday.append(0)
df['before_holiday'] = before_holiday

df=df.sort_values(['index']).reset_index(drop=True).drop(columns=['index'])

##HolidayPerWeekの追加
Holiday_Year_Month_Week_df=pd.DataFrame({"HolidayPerWeek":df["SchoolHoliday"],"Week":df["Week"],"Month":df["Month"],"Year":df["Year"],"Date":df["Date"]})
Holiday_Year_Month_Week_df=Holiday_Year_Month_Week_df.drop_duplicates(subset=['Date'])
Holiday_Year_Month_Week_df=Holiday_Year_Month_Week_df.groupby(["Year","Month","Week"]).sum()
Holiday_Year_Month_Week_df

df=df.merge(Holiday_Year_Month_Week_df, on=["Year","Month","Week"],how="left")

# promoCountPerWeekの追加
promo_time_df=pd.DataFrame({"PromoCountPerWeek":df["Promo"],"Year":df["Year"],"Month":df["Month"],"Week":df["Week"],"Date":df["Date"]})
promo_time_df=promo_time_df.drop_duplicates(subset=['Date'])
promo_time_df=promo_time_df.groupby(["Year","Month","Week"]).sum()

df=df.merge(promo_time_df,on=["Year","Month","Week"], how="left")

# 祝日
df["StateHoliday"]=np.where(df["StateHoliday"] == '0' ,0,1)

df  = pd.get_dummies(df ,columns=["Month"])
df  = pd.get_dummies(df,columns=["DayOfWeek"])

# dfのcolumnsの一覧を確認

features = ["Open","Promo",'平均売上log']
features.extend(["Assortment_a","Assortment_b"])        
#features.extend(["PromoCountPerWeek","HolidayPerWeek","SchoolHoliday"])
features.extend(["PromoCountPerWeek","HolidayPerWeek"])
features.extend(['StoreType_a', 'StoreType_b','StoreType_c'])
features.extend(['before_holiday',])
features.extend(['Month_1','Month_2','Month_3','Month_4' ,'Month_5' ,'Month_6'])
features.extend(['Month_7','Month_10','Month_11','Month_12'])
features.extend(['Month_8','Month_9',])
features.extend(['DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3','DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7'])
features.extend(['DayOfWeek_6', 'DayOfWeek_7'])
features.extend(["CompetitionDistance"])
features.extend(["平均売上log(Day)"])


train_x  = df[(df['is_train']==True) & (df['Open']==1)][features]
test_x   = df[df['is_train']==False][features].fillna(1)
train_y  = train_df['Sales']



train_df    = train_df[train_df['Open']==1]
train_sales = train_df['Sales' ]
train_df['logSales'] = train_df['Sales'].apply(lambda x: np.log(np.min([x,max_sale])) if x!=0 else -1)
train_y  = train_df['logSales']

scaler = StandardScaler()
scaler.fit(train_x)

train_x  = scaler.transform(train_x)
test_x   = scaler.transform(test_x)

model=LinearRegression()
model.fit(train_x,train_y)

predict_train = model.predict(train_x)

print(rmspe(train_sales.tolist(),np.exp(predict_train)))

predict_test  = model.predict(test_x)
predict_test = np.round(np.exp(predict_test))
##03_day,week,month mean

# Salesに上限を設定する
max_sale = 100000

#reading data files
store_df=pd.read_csv("../input/rossmann-store-sales/store.csv", low_memory=False)
store_df= pd.get_dummies(store_df,columns=["StoreType","Assortment"])

# train_dfとtest_dfを縦に結合
train_df=pd.read_csv("../input/rossmann-store-sales/train.csv", low_memory=False)
test_df=pd.read_csv("../input/rossmann-store-sales/test.csv", low_memory=False)
train_df['is_train']=True
test_df['is_train'] =False
df = pd.concat([train_df.drop(columns=['Sales']),test_df.drop(columns=['Id'])]).reset_index(drop=True).copy()

# 平均売上の算出(trainのみ使用)
dates1 = (df['Date'] >='2013-01-02') & (df['Date'] < '2014-07-01')
dates2 = (df['Date'] >='2015-01-01') & (df['Date'] < '2015-06-01')
dates12 = dates1 | dates2
except_sunday = (df['DayOfWeek'] != 7)
only_open = (df['Open']==1)
df_for_mean = train_df[dates12 & except_sunday & only_open].reset_index().copy()
df_for_mean['Sales'] = df_for_mean['Sales'].apply(lambda x: np.min([x,max_sale]))
store_mean_sales = df_for_mean.groupby('Store').mean()['Sales']

#Day平均を入れる
df_for_mean["Date"] =pd.to_datetime(df_for_mean["Date"])
df_for_mean["Year"] =df_for_mean["Date"].dt.year
df_for_mean["Month"]=df_for_mean["Date"].dt.month
df_for_mean["Day"]  =df_for_mean["Date"].dt.day
df_for_mean["Week"] =df_for_mean["Date"].dt.week%4
Day_mean_sales = df_for_mean.groupby('Day').mean()['Sales']
Week_mean_sales = df_for_mean.groupby('Week').mean()['Sales']
Month_mean_sales = df_for_mean.groupby('Month').mean()['Sales']

#年月日や週番号を入れる
df["Date"] =pd.to_datetime(df["Date"])
df["Year"] =df["Date"].dt.year
df["Month"]=df["Date"].dt.month
df["Day"]  =df["Date"].dt.day
df["Week"] =df["Date"].dt.week%4


# dataframeに平均売上を付与
df['平均売上']   = df['Store'].apply(lambda x : store_mean_sales.loc[x])
df['平均売上log']= df['平均売上'].apply(lambda x: np.log(x) if x!=0 else -1)

df['平均売上(Day)']   = df['Day'].apply(lambda x : Day_mean_sales.loc[x])
df['平均売上log(Day)']= df['平均売上(Day)'].apply(lambda x: np.log(x) if x!=0 else -1)
df['平均売上(Week)']   = df['Week'].apply(lambda x : Week_mean_sales.loc[x])
df['平均売上log(Week)']= df['平均売上(Week)'].apply(lambda x: np.log(x) if x!=0 else -1)
df['平均売上(Month)']   = df['Month'].apply(lambda x : Month_mean_sales.loc[x])
df['平均売上log(Month)']= df['平均売上(Month)'].apply(lambda x: np.log(x) if x!=0 else -1)


df=df.merge(store_df,on=["Store"],how="left")


#競合店の距離は補完
df["CompetitionDistance"]=df["CompetitionDistance"].fillna(df["CompetitionDistance"].mode()[0])


# 翌日お休みフラグ
df = df.sort_values(['Store', 'Date']).reset_index(drop=False)
date_column = df.columns.get_loc('Date')
open_column = df.columns.get_loc('Open')

before_holiday=[]
for i in range(len(df)-1):
    if df.iat[i+1,date_column]==df.iat[i,date_column]+ datetime.timedelta(days=1) and df.iat[i+1,open_column]==0:
        before_holiday.append(1)
    else:
        before_holiday.append(0)
        
before_holiday.append(0)
df['before_holiday'] = before_holiday

df=df.sort_values(['index']).reset_index(drop=True).drop(columns=['index'])

##HolidayPerWeekの追加
Holiday_Year_Month_Week_df=pd.DataFrame({"HolidayPerWeek":df["SchoolHoliday"],"Week":df["Week"],"Month":df["Month"],"Year":df["Year"],"Date":df["Date"]})
Holiday_Year_Month_Week_df=Holiday_Year_Month_Week_df.drop_duplicates(subset=['Date'])
Holiday_Year_Month_Week_df=Holiday_Year_Month_Week_df.groupby(["Year","Month","Week"]).sum()
Holiday_Year_Month_Week_df

df=df.merge(Holiday_Year_Month_Week_df, on=["Year","Month","Week"],how="left")

# promoCountPerWeekの追加
promo_time_df=pd.DataFrame({"PromoCountPerWeek":df["Promo"],"Year":df["Year"],"Month":df["Month"],"Week":df["Week"],"Date":df["Date"]})
promo_time_df=promo_time_df.drop_duplicates(subset=['Date'])
promo_time_df=promo_time_df.groupby(["Year","Month","Week"]).sum()

df=df.merge(promo_time_df,on=["Year","Month","Week"], how="left")

# 祝日
df["StateHoliday"]=np.where(df["StateHoliday"] == '0' ,0,1)

df  = pd.get_dummies(df ,columns=["Month"])
df  = pd.get_dummies(df,columns=["DayOfWeek"])

# dfのcolumnsの一覧を確認

features = ["Open","Promo",'平均売上log']
features.extend(["Assortment_a","Assortment_b"])        
#features.extend(["PromoCountPerWeek","HolidayPerWeek","SchoolHoliday"])
features.extend(["PromoCountPerWeek","HolidayPerWeek"])
features.extend(['StoreType_a', 'StoreType_b','StoreType_c'])
features.extend(['before_holiday',])
features.extend(['Month_1','Month_2','Month_3','Month_4' ,'Month_5' ,'Month_6'])
features.extend(['Month_7','Month_10','Month_11','Month_12'])
features.extend(['Month_8','Month_9',])
features.extend(['DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3','DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7'])
features.extend(['DayOfWeek_6', 'DayOfWeek_7'])
features.extend(['CompetitionDistance'])
features.extend(['平均売上log(Day)','平均売上log(Week)','平均売上log(Month)'])


train_x  = df[(df['is_train']==True) & (df['Open']==1)][features]
test_x   = df[df['is_train']==False][features].fillna(1)
train_y  = train_df['Sales']



train_df    = train_df[train_df['Open']==1]
train_sales = train_df['Sales' ]
train_df['logSales'] = train_df['Sales'].apply(lambda x: np.log(np.min([x,max_sale])) if x!=0 else -1)
train_y  = train_df['logSales']

scaler = StandardScaler()
scaler.fit(train_x)

train_x  = scaler.transform(train_x)
test_x   = scaler.transform(test_x)

model=LinearRegression()
model.fit(train_x,train_y)

predict_train = model.predict(train_x)

print(rmspe(train_sales.tolist(),np.exp(predict_train)))

predict_test  = model.predict(test_x)
predict_test = np.round(np.exp(predict_test))