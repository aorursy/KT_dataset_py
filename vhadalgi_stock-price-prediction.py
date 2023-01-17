import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

Data=pd.read_csv('../input/consolidated.csv',header=None,names=["Company_share_code","Date","Opening_price","Highest_price","Lowest_price","Close","Volume_traded","Openint"])



Data['Company_share_code']=Data['Company_share_code'].str.replace(".us.txt",'')
Data=Data[Data['Company_share_code'].isin (['aapl','intc','msft','csco','bac'])]
Data=Data[Data['Company_share_code'].isin (['aapl','intc','msft','csco','bac'])]

Enc=pd.get_dummies(Data['Company_share_code'])

Data=Data.drop(['Company_share_code'],axis=1)

Data=Data.join(Enc)
Data.head(5)
Data['Stock_Price']=(Data['Opening_price']+Data['Highest_price']+Data['Lowest_price']+Data['Close'])/4
#Datepart function from fast ai to get features

def add_datepart(df, fldname, drop=True, time=False):

    "Helper function that adds columns relevant to a date."

    fld = df[fldname]

    fld_dtype = fld.dtype

    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):

        fld_dtype = np.datetime64



    if not np.issubdtype(fld_dtype, np.datetime64):

        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)

    targ_pre = re.sub('[Dd]ate$', '', fldname)

    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',

            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

    if time: attr = attr + ['Hour', 'Minute', 'Second']

    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())

    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9

    if drop: df.drop(fldname, axis=1, inplace=True)
import re

add_datepart(Data,"Date",drop=False)


#Dropping columns

# Data=Data.drop(['Highest_price','Lowest_price','Close','Opening_price','Volume_traded'],axis=1)

# Data=Data.drop(['Openint'],axis=1)

#Converting True/False to 1...0 and dropping Elapsed and Date

Data=Data.drop(['Date','Elapsed','Highest_price','Lowest_price','Close','Opening_price','Volume_traded','Openint'],axis=1)

Data=Data*1
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,LogisticRegression,perceptron



LR=LinearRegression()

LogR=LogisticRegression()

iv=Data.drop(['Stock_Price'],axis=1)

dv=Data[['Stock_Price']]

iv_train,iv_test,dv_train,dv_test=train_test_split(iv,dv,test_size=0.25,random_state=0)
#Predicting dataset

LR.fit(iv_train,dv_train)

y_pred_LR=LR.predict(iv_test)

print('The accuracy of LR is',LR.score(iv_test,dv_test)*100)
#Predicting with Decision Tree

from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor

DT=DecisionTreeRegressor(max_depth=3)

DT.fit(iv_train,dv_train)

y_pred_DT=DT.predict(iv_test)
print('The accuracy of DT is',r2_score(y_pred_DT,dv_test)*100)
#Trend Analysis



# %matplotlib inline

# import seaborn as sns

# # Data['Date']=pd.to_datetime(Data['Date'],format='%Y-%m-%d')

# # Data.index=Data['Date']

# plt.figure(figsize=(12,10))

# sns.lineplot(Data['Date'],Data['Close'],hue='Company_share_code',data=Data)

Data.shape
%matplotlib inline

import matplotlib.pyplot as plt

Data.hist(bins=50, figsize=(20,15))

plt.savefig("attribute_histogram_plots")

plt.show()