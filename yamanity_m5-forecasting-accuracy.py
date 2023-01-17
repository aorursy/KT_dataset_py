import numpy as np

import pandas as pd
df_calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

df_sales_train_evaluation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')

df_sales_train_validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

df_sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
df_calendar.info()
df_calendar.groupby(['event_type_1', 'event_name_1'])['date'].count()
df_calendar.groupby(['event_type_2', 'event_name_2'])['date'].count()
df_sales_train_validation.groupby(['cat_id', 'dept_id'])['item_id'].count()
df_sales_train_validation.groupby(['state_id', 'store_id'])['item_id'].count()