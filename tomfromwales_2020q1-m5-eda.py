# libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy

import os

import plotly.express as px

from scipy import stats



# read in data provided by kaggle for the competition

source_filesinfolder = os.listdir('../input/m5-forecasting-accuracy')

source_filenames = [file.replace('.csv', '') for file in source_filesinfolder]

source_data_path = '/kaggle/input/m5-forecasting-accuracy'

for file in source_filenames:

    globals()[file] = pd.read_csv(f'{source_data_path}/{file}.csv')   
print(source_filenames)

for file in source_filenames:

    print(globals()[file])
# Get a dataset with info on each item

item_info = sales_train_validation[["dept_id","cat_id","item_id"]].drop_duplicates()



# Join category onto sell_prices

sell_prices_for_eda = (

    sell_prices

    .merge(

        item_info

        , how = "left"

        , on = "item_id"

    )

)



# Summarise by item & store

item_store_summaries = (

    sell_prices_for_eda

    .groupby(['item_id', 'store_id' ,'dept_id', 'cat_id'])['sell_price']

    .agg(

        price_mode = lambda x: stats.mode(x)[0][0]

        , price_mean = 'mean'

        , price_min = 'min'

        , price_max = 'max'

        , price_sd = np.std

        , price_sum = 'sum'

        , price_size = 'size'

    )

    .reset_index()

    .assign(max_discount_from_peak = lambda x: 1 - x['price_min']/x['price_max'])

)



# Summarise by item

item_summaries = (

    item_store_summaries

    .groupby(['item_id','dept_id'])['price_mode']

    .agg(

        price_mode__nationwide_mean = 'mean'

        , min = 'min'

        , max = 'max'

    )

    .reset_index()

    .assign(price_mode__nationwide_range = lambda x: x['max'] - x['min'])

)



# Summarise by time

time_dept_summaries = (

    sell_prices_for_eda

    .groupby(['wm_yr_wk','dept_id'])['sell_price']

    .agg(

        perct_25th = lambda x: np.percentile(x, q = 25)

        , perct_75th = lambda x: np.percentile(x, q = 75)

        , perct_95th = lambda x: np.percentile(x, q = 95)

        , median = 'median'

        , mean = 'mean'

    )

    .reset_index()

    .merge(

        calendar.groupby('wm_yr_wk').first().reset_index()[['wm_yr_wk','date']].rename(columns={'date':'start_of_week'})

        , how = "left"

        , on = "wm_yr_wk"

    )

)
px.histogram(

    item_info

    , x = "dept_id"

    , title = 'Count of unique items per department'

).show()
px.box(

    item_store_summaries

    , x = "dept_id"

    , y = "price_mode"

    , color = "store_id"

    , title = 'Distributions of standard (mode) item prices across dept and store'

).show()
px.box(

    item_summaries

    , x = 'dept_id'

    , y = 'price_mode__nationwide_mean'

    #, color = "store_id"

    , title = 'Distributions of item prices across dept'

).show()
px.line(

    time_dept_summaries

    , x = "start_of_week"

    , y = "mean"

    , color = "dept_id"

    , title = 'Mean price per department over time'

).show()