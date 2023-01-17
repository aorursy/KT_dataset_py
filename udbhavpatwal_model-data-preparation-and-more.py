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
import gc
calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")

sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

sales_train = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

sample_sub = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")

sell_prices.isnull().sum()
def memory_reduction(dataset):

    column_types = dataset.dtypes

    temp = None

    for x in range(len(column_types)):

        column_types[x] = str(column_types[x])

    for x in range(len(column_types)):

        temp = dataset.columns[x]

        if dataset.columns[x] == "date":

            dataset[temp] = dataset[temp].astype("datetime64")

        if column_types[x] == "int64" and dataset.columns[x] != "date":

            dataset[temp] = dataset[temp].astype("int16")

        if column_types[x] == "object" and dataset.columns[x] != "date":

            dataset[temp] = dataset[temp].astype("category")

        if column_types[x] == "float64" and dataset.columns[x] != "date":

            dataset[temp] = dataset[temp].astype("float16")

    return dataset
calendar = memory_reduction(calendar)

sell_prices = memory_reduction(sell_prices)

sales_train = memory_reduction(sales_train)

sample_sub = memory_reduction(sample_sub)
#we arent removing data for the moment like we did previously 

# Next we need to transform our dates to usable or model efficient formats

# For that we will be creating a day column and week num columns and reducing there memory

calendar["day"] = pd.DatetimeIndex(calendar["date"]).day

calendar["day"] = calendar["day"].astype("int8")

calendar["week_num"] = (calendar["day"] - 1) // 7 + 1

calendar["week_num"] = calendar["week_num"].astype("int8")
# Now lets see it 

calendar.tail()
#we have to add a category named missing in order to omit the NaN values as we have described object data type to category data type

calendar["event_name_1"] = calendar["event_name_1"].cat.add_categories('missing')
from sklearn.preprocessing import OrdinalEncoder

# we are droping date

# we are stripping d_ values to int

# changing Nan values to mising in events 

# and also integer encoding the values of catagorical variable

def prep_calendar(df):

    df = df.drop(["date", "weekday"], axis=1)

    df = df.assign(d = df.d.str[2:].astype(int))

    df["event_type_1"] = df["event_type_1"].cat.add_categories('missing')

    df["event_name_2"] = df["event_name_2"].cat.add_categories('missing')

    df["event_type_2"] = df["event_type_2"].cat.add_categories('missing')

    

    df = df.fillna("missing")

    cols = list(set(df.columns) - {"wm_yr_wk", "d"})

    df[cols] = OrdinalEncoder(dtype="int").fit_transform(df[cols])

    df = memory_reduction(df)

    return df



calendar = prep_calendar(calendar)
calendar.head()
sales_train.head()
sales_train.shape
#this is basically changing sales data from wide to long format 





def reshape_sales(df, drop_d = None):

    if drop_d is not None:

        df = df.drop(["d_" + str(i + 1) for i in range(drop_d)], axis=1)

    df = df.assign(id=df.id.str.replace("_validation", ""))

    df = df.reindex(columns=df.columns.tolist() + ["d_" + str(1913 + i + 1) for i in range(2 * 28)])

    df = df.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],

                 var_name='d', value_name='demand')

    df = df.assign(d=df.d.str[2:].astype("int16"))

    return df



sales_train = reshape_sales(sales_train, 488)
sales_train.head()
from sklearn.preprocessing import OrdinalEncoder
#now we are calculating rolling mean and standard deviation

def prep_sales(df):

    #this is shifting the data by 28

    df['lag_t28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28))

    #rolling mean window 7

    df['rolling_mean_t7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())

    #rolling mean window 15

    df['rolling_mean_t15'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(15).mean())

    #rolling mean window 30

    df['rolling_mean_t30'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())

    #rolling std window 7

    df['rolling_std_t7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())

    #rolling mean window 30

    df['rolling_std_t30'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())



    # Remove rows with NAs except for submission rows. rolling_mean_t180 was selected as it produces most missings

    df = df[(df.d >= 1914) | (pd.notna(df.rolling_mean_t30))]

    df = memory_reduction(df)



    return df



sales_train = prep_sales(sales_train)
sales_train.head()
sell_prices.dtypes
#we have only calculated rolling mean, std , cummulative relation btween them 



def prep_selling_prices(df):

    

    gr = df.groupby(["store_id", "item_id"])["sell_price"]

    df["sell_price_rel_diff"] = gr.pct_change()

    df["sell_price_rel_diff"] = df["sell_price_rel_diff"].astype("float16")

    df["sell_price_rel_diff"] = df["sell_price_rel_diff"].fillna(0) 

    df["sell_price_roll_sd7"] = gr.transform(lambda x: x.rolling(7).std())

    df["sell_price_roll_sd7"] = df["sell_price_roll_sd7"].astype("float16")

    df["sell_price_roll_sd7"] = df["sell_price_roll_sd7"].fillna(0)

    df["sell_price_cumrel"] = (gr.shift(0) - gr.cummin()) / (1 + gr.cummax() - gr.cummin())

    df["sell_price_cumrel"] = df["sell_price_cumrel"].astype("float16")

    df["sell_price_cumrel"] = df["sell_price_cumrel"].fillna(0)

    df = memory_reduction(df)

    return df



sell_prices = prep_selling_prices(sell_prices)
sell_prices.dtypes
sell_prices.head()
#merging with calendar

sales_train = sales_train.merge(calendar, how="left", on="d")

gc.collect()

sales_train.head()
sales_train = sales_train.merge(sell_prices, how="left", on=["wm_yr_wk", "store_id", "item_id"])

sales_train.drop(["wm_yr_wk"], axis=1, inplace=True)

gc.collect()

sales_train.head()
del sell_prices

gc.collect()

sales_train.info()
gc.collect()
sales_train.tail()
"""

cat_id_cols = ["item_id", "dept_id", "store_id", "cat_id", "state_id"]

cat_cols = cat_id_cols + ["wday", "month", "year", "event_name_1", 

                          "event_type_1", "event_name_2", "event_type_2"]

# if you want to check the progress of it use 

# from tqdm import tqdm

# for i, v in tqdm(enumerate(cat_id_cols)):

# In loop to minimize memory use

for i, v in enumerate(cat_id_cols):

    sales_train[v] = OrdinalEncoder(dtype="int").fit_transform(sales_train[[v]])



sales_train = memory_reduction(sales_train)

gc.collect()

sales_train.head()

"""
"""

num_cols = ["sell_price", "sell_price_rel_diff", "sell_price_roll_sd7", "sell_price_cumrel",

            "lag_t28", "rolling_mean_t7", "rolling_mean_t15", "rolling_mean_t30", 

            "rolling_std_t7", "rolling_std_t30"]

bool_cols = ["snap_CA", "snap_TX", "snap_WI"]

dense_cols = num_cols + bool_cols



# Need to do column by column due to memory constraints

for i, v in enumerate(num_cols):

    sales_train[v] = sales_train[v].fillna(sales_train[v].median())

    

sales_train.head()

"""
"""

test = sales_train[sales_train.d >= 1914]

test["id"] = test["id"].astype("str")

test = test.assign(id=test.id + "_" + np.where(test.d <= 1941, "validation", "evaluation"),

                   F="F" + (test.d - 1913 - 28 * (test.d > 1941)).astype("str"))

test.head()

gc.collect()

"""
"""

# Input dict for training with a dense array and separate inputs for each embedding input

def make_X(df):

    X = {"dense1": df[dense_cols].to_numpy()}

    for i, v in enumerate(cat_cols):

        X[v] = df[[v]].to_numpy()

    return X



# Submission data

X_test = make_X(test)



# One month of validation data

flag = (sales_train.d < 1914) & (sales_train.d >= 1914 - 28)

valid = (make_X(sales_train[flag]),

         sales_train["demand"][flag])



# Rest is used for training

flag = sales_train.d < 1914 - 28

X_train = make_X(sales_train[flag])

y_train = sales_train["demand"][flag]

                             

del sales_train, flag

gc.collect()

"""
"""

gc.collect()

"""
"""

y_train.head()

"""
#lets first redue the data to d 1913 for train and rest for the purpose of test 



train = sales_train[sales_train['d'] <= 1913]

test = sales_train[sales_train['d'] >= 1913]

test = sales_train[sales_train['d'] >= 1941]

del sales_train

gc.collect()
train.info()
test.info()
gc.collect()
#new memory reduction

def memory_reduction(dataset):

    column_types = dataset.dtypes

    temp = None

    for x in range(len(column_types)):

        column_types[x] = str(column_types[x])

    for x in range(len(column_types)):

        temp = dataset.columns[x]

        if dataset.columns[x] == "date":

            dataset[temp] = dataset[temp].astype("datetime64")

        if column_types[x] == "int16" and dataset.columns[x] != "date":

            dataset[temp] = dataset[temp].astype("int8")

        if column_types[x] == "object" and dataset.columns[x] != "date":

            dataset[temp] = dataset[temp].astype("category")

        if column_types[x] == "float16" and dataset.columns[x] != "date":

            dataset[temp] = dataset[temp].astype("float16")

    return dataset
train = memory_reduction(train)

train.info()
# saving ids for future 

train_ids = train['id']

test_ids = test['id']
#droping id for training purpose

train = train.drop('id', axis = 1)

test = test.drop('id', axis = 1)
gc.collect()
columns = ['item_id' ,'dept_id', 'cat_id', 'store_id', 'state_id']

ode = OrdinalEncoder()

train = ode.fit_transform(train)

gc.collect()
from sklearn.linear_model import LinearRegression
lreg = LinearRegression()
lreg.fit(X_train, y_train)