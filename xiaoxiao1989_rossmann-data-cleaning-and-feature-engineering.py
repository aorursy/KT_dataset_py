from __future__ import print_function  # Compatability with Python 3

# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)
from pandas import DataFrame
from pandas import TimeGrouper

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
%matplotlib inline 

# Seaborn for easier visualization
import seaborn as sns

# datetime
import datetime
# Load data from CSV
df = pd.read_csv("../input/train.csv")
df2 = pd.read_csv("../input/store.csv")
test = pd.read_csv("../input/test.csv")
# Drop duplicates
df = df.drop_duplicates()
df2 = df2.drop_duplicates()
# drop sales == 0 observations
df = df[df.Sales != 0]
def find_low_high(feature):
    # find store specific Q1 - 3*IQ and Q3 + 3*IQ
    IQ = df.groupby('Store')[feature].quantile(0.75)-df.groupby('Store')[feature].quantile(0.25)
    Q1 = df.groupby('Store')[feature].quantile(0.25)
    Q3 = df.groupby('Store')[feature].quantile(0.75)
    low = Q1 - 3*IQ
    high = Q3 + 3*IQ
    low = low.to_frame()
    low = low.reset_index()
    low = low.rename(columns={feature: "low"})
    high = high.to_frame()
    high = high.reset_index()
    high = high.rename(columns={feature: "high"})
    return {'low':low, 'high':high}
def find_outlier_index(feature):
    main_data = df[['Store',feature]]
    low = find_low_high(feature)["low"]
    high = find_low_high(feature)["high"]
    
    new_low = pd.merge(main_data, low, on='Store', how='left')
    new_low['outlier_low'] = (new_low[feature] < new_low['low'])
    index_low = new_low[new_low['outlier_low'] == True].index
    index_low = list(index_low)
    
    new_high = pd.merge(main_data, high, on='Store', how='left')
    new_high['outlier_high'] = new_high[feature] > new_high['high']
    index_high = new_high[new_high['outlier_high'] == True].index
    index_high = list(index_high)
    
    index_low.extend(index_high)
    index = list(set(index_low))
    return index
# decide to delete the 1113 observations above to delete the sales outlier
df=df.reset_index()
df.drop(find_outlier_index("Sales"), inplace=True, axis=0)
df2.fillna(0, inplace=True)
test.fillna(0, inplace=True)
df2.CompetitionDistance=df2.CompetitionDistance.astype(float)
df.StateHoliday.replace(0, '0',inplace=True)
df2.PromoInterval.replace('0', 0,inplace=True)
df2.PromoInterval.replace('Jan,Apr,Jul,Oct', 1,inplace=True)
df2.PromoInterval.replace('Feb,May,Aug,Nov', 2,inplace=True)
df2.PromoInterval.replace('Mar,Jun,Sept,Dec', 3,inplace=True)
def get_store_sales_statistics(df, df2):
    mean = df.groupby('Store')['Sales'].mean()
    std = df.groupby('Store')['Sales'].std()
    mean_dataframe = pd.DataFrame(mean).reset_index()
    std_dataframe = pd.DataFrame(std).reset_index()
    df2 = pd.merge(df2,mean_dataframe, on='Store', how='left').rename(columns={"Sales": "SalesMean"})
    df2 = pd.merge(df2,std_dataframe, on='Store', how='left').rename(columns={"Sales": "SalesStd"})
    return df2
def get_sales_level_groups(df2):
    Q1 = df2.SalesMean.quantile(0.25)
    Q2 = df2.SalesMean.quantile(0.50)
    Q3 = df2.SalesMean.quantile(0.75)
    df2['StoreGroup1'] = (df2.SalesMean < Q1).astype(int)
    df2['StoreGroup2'] = ((df2.SalesMean>=Q1) & (df2.SalesMean<Q2)).astype(int)
    df2['StoreGroup3'] = ((df2.SalesMean>=Q2) & (df2.SalesMean<Q3)).astype(int)
    df2['StoreGroup4'] = (df2.SalesMean>=Q3).astype(int)
    df2['StoreGroup']= df2['StoreGroup1'] + 2*df2['StoreGroup2'] + 3*df2['StoreGroup3'] + 4*df2['StoreGroup4']
    df2.drop(['StoreGroup1','StoreGroup2','StoreGroup3','StoreGroup4'],axis=1, inplace=True)
    return df2
df2 = get_store_sales_statistics(df, df2)
df2 = get_sales_level_groups(df2)
def get_date_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Week'] = df['Date'].dt.week  
    return df
df = get_date_features(df)
test = get_date_features(test)
df['Sales'] = np.log(df.Sales)
df2['SalesMean'] = np.log(df2.SalesMean)
df2['SalesStd'] = np.log(df2.SalesStd)
df_combined = pd.merge(df[['Store','DayOfWeek','Date','Sales','Promo','SchoolHoliday','Year','Month','Day','Week']], df2, on='Store', how='left')
test_combined = pd.merge(test, df2, on='Store', how='left')
def get_Promo2_ongoing(df_combined):
    df_combined['Promo2_ongoing'] = ((df_combined.Promo2 == 1) & (df_combined.Year > df_combined.Promo2SinceYear)).astype(int) + ((df_combined.Promo2 == 1) & (df_combined.Year == df_combined.Promo2SinceYear) & (df_combined.Week >= df_combined.Promo2SinceWeek)).astype(int)
    df_combined['Promo2_ongoing_now'] = ((df_combined['Promo2_ongoing'] == 1) & (df_combined.Month % 3 == df_combined.PromoInterval % 3)).astype(int)
    df_combined.drop(['Promo2','Promo2SinceWeek','Promo2SinceYear','PromoInterval','Promo2_ongoing'], axis = 1, inplace=True)
    return df_combined
df_combined = get_Promo2_ongoing(df_combined)
test_combined = get_Promo2_ongoing(test_combined)
def helper_get_date(a,b):
    c=[]
    for i in range(len(a)):
        if a[i] == 0:
            c.append(datetime.date(1900,1,1))
        else:
            c.append(datetime.date(a[i],b[i],1))
    return c
def get_Competition_ongoing(df_combined):
    df_combined['CompetitionOpen'] = ((df_combined.Year > df_combined.CompetitionOpenSinceYear)).astype(int) + ((df_combined.Year == df_combined.CompetitionOpenSinceYear) & (df_combined.Month >= df_combined.CompetitionOpenSinceMonth)).astype(int)
    df_combined.CompetitionOpenSinceYear = df_combined.CompetitionOpenSinceYear.astype(int)
    df_combined.CompetitionOpenSinceMonth = df_combined.CompetitionOpenSinceMonth.astype(int)
    df_combined['CompetitionOpenDate'] = helper_get_date(df_combined.CompetitionOpenSinceYear, df_combined.CompetitionOpenSinceMonth)
    df_combined['NumOfCompetitionOpenMonths'] = ((df_combined['CompetitionOpen'] * ((df_combined.Date - pd.to_datetime(df_combined['CompetitionOpenDate']))/30))/ np.timedelta64(1, 'D')).astype(int)
    df_combined.drop(['CompetitionOpenSinceYear','CompetitionOpenSinceMonth','CompetitionOpenDate'], axis = 1, inplace=True)
    return df_combined
df_combined = get_Competition_ongoing(df_combined)
test_combined = get_Competition_ongoing(test_combined)
df_ABT1 = df_combined.drop(['Date','Store'], axis=1)
df_ABT1 = pd.get_dummies(df_ABT1,columns=['DayOfWeek', 'StoreType', 'Assortment'])
test_ABT1 = test_combined.drop(['Id','Date','Open','Store','StateHoliday'], axis=1)
test_ABT1 = pd.get_dummies(test_ABT1,columns=['DayOfWeek', 'StoreType', 'Assortment'])