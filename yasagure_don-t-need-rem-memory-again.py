import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def reduce_mem_usage(df, verbose=True):

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]

    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == "int":

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if (

                    c_min > np.finfo(np.float16).min

                    and c_max < np.finfo(np.float16).max

                ):

                    df[col] = df[col].astype(np.float16)

                elif (

                    c_min > np.finfo(np.float32).min

                    and c_max < np.finfo(np.float32).max

                ):

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose:

        print(

            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(

                end_mem, 100 * (start_mem - end_mem) / start_mem

            )

        )

    return df
!free
sales_train_validations = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")
sales_train_validations.info()
!free
sales_train_validations = reduce_mem_usage(sales_train_validations)
!free
sales_train_validations.info()
!free
a = pd.DataFrame(np.ones(10000000))
a.info(all)
a.iloc[:,0] = a.iloc[:,0].astype(np.int8)
a.info()
!free
pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv").info()
sell_prices = reduce_mem_usage(pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv"))

samplesubmission = reduce_mem_usage(pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv"))

calendar = reduce_mem_usage(pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv"))

sales_train_validation = reduce_mem_usage(pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv"))
samplesubmission.to_feather("sample_submission.feather")
##float16 may not be changed to feather, we should possibly use pickle

sell_prices["sell_price"] = sell_prices["sell_price"].astype(np.float32)
sell_prices.to_feather("sell_prices.feather")

samplesubmission.to_feather("sample_submission.feather")

calendar.to_feather("calendar.feather")

sales_train_validation.to_feather("sales_train_validation.feather")
pd.read_feather('sell_prices.feather').info()
%%time

pd.read_feather("sell_prices.feather")

pd.read_feather("sample_submission.feather")

pd.read_feather("calendar.feather")

pd.read_feather("sales_train_validation.feather")