from  datetime import datetime, timedelta

import gc

import numpy as np, pandas as pd
CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 

         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",

        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }

PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
pd.options.display.max_columns = 50
h = 28 

max_lags = 57

tr_last = 1913

fday = datetime(2016,4, 25) 

fday
def create_dt(is_train = True, nrows = None, first_day = 1200):

    prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv", dtype = PRICE_DTYPES)

    for col, col_dtype in PRICE_DTYPES.items():

        if col_dtype == "category":

            prices[col] = prices[col].cat.codes.astype("int16")

            prices[col] -= prices[col].min()

            

    cal = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)

    cal["date"] = pd.to_datetime(cal["date"])

    for col, col_dtype in CAL_DTYPES.items():

        if col_dtype == "category":

            cal[col] = cal[col].cat.codes.astype("int16")

            cal[col] -= cal[col].min()

    

    start_day = max(1 if is_train  else tr_last-max_lags, first_day)

    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]

    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

    dtype = {numcol:"float32" for numcol in numcols} 

    dtype.update({col: "category" for col in catcols if col != "id"})

    dt = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv", 

                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)

    

    for col in catcols:

        if col != "id":

            dt[col] = dt[col].cat.codes.astype("int16")

            dt[col] -= dt[col].min()

    

    if not is_train:

        for day in range(tr_last+1, tr_last+ 28 +1):

            dt[f"d_{day}"] = np.nan

    

    dt = pd.melt(dt,

                  id_vars = catcols,

                  value_vars = [col for col in dt.columns if col.startswith("d_")],

                  var_name = "d",

                  value_name = "sales")

    

    dt = dt.merge(cal, on= "d", copy = False)

    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)

    

    return dt
def create_fea(dt):

    lags = [7, 28]

    lag_cols = [f"lag_{lag}" for lag in lags ]

    for lag, lag_col in zip(lags, lag_cols):

        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)



    wins = [7, 28]

    for win in wins :

        for lag,lag_col in zip(lags, lag_cols):

            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())



    

    

    date_features = {

        

        "wday": "weekday",

        "week": "weekofyear",

        "month": "month",

        "quarter": "quarter",

        "year": "year",

        "mday": "day",

#         "ime": "is_month_end",

#         "ims": "is_month_start",

    }

    

#     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)

    

    for date_feat_name, date_feat_func in date_features.items():

        if date_feat_name in dt.columns:

            dt[date_feat_name] = dt[date_feat_name].astype("int16")

        else:

            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")
FIRST_DAY = 1050
%%time



df = create_dt(is_train=True, first_day= FIRST_DAY)

df.shape
df.head()
%%time



create_fea(df)

df.shape
df.dropna(inplace = True)

df.shape
useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]

train_cols = df.columns[~df.columns.isin(useless_cols)]



train_cols = list(train_cols)

train_cols.append("sales")

train_cols
X_train = df[train_cols]
X_train.to_csv("X_train.csv", index=False)
del df, X_train 

gc.collect()

test = create_dt(False)
test.shape
create_fea(test)
train_cols[:-1]
X_test = test[train_cols[:-1]]
X_test.to_csv("X_test.csv", index=False)
del test, X_test

gc.collect()