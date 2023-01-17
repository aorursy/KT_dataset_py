import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
df.head()
len(df)
df.info()
df.isnull().sum()
df = df.drop(['company', 'agent'], axis=1)
df.head(5)
df_gorup_by_year = df.is_canceled.groupby(df.arrival_date_year).mean()
df_gorup_by_year
my_colors = 'rgbkymc'
df_gorup_by_year.plot.bar(color=list(my_colors))
df_gorup_by_month = df.is_canceled.groupby(df.arrival_date_month).mean()
df_gorup_by_year
df_gorup_by_month.plot.bar(color=list(my_colors))
df.hotel.unique()
df_gorup_by_type = df.is_canceled.groupby(df.hotel).mean()
df_gorup_by_type
df_gorup_by_type.plot( kind='bar', color=list(my_colors))

df_gorup_by_type_year = df.groupby(["hotel","arrival_date_year"])["is_canceled"].count()
df_gorup_by_type_year
df_gorup_by_type_year.plot( kind='bar', color=list(my_colors))

df_gorup_by_type_year_month = df.groupby(["hotel","arrival_date_year","arrival_date_month"])["is_canceled"].count()
df_gorup_by_type_year_month
df_gorup_by_type_year_month.plot( kind='bar', color=list(my_colors))

more_than_3500 = df_gorup_by_type_year_month[df_gorup_by_type_year_month>3500]
more_than_3500
more_than_3500.plot( kind='bar', color=list(my_colors))

df_gorup_by_type_year_repated_guest = df.groupby(["hotel","arrival_date_year"])["is_repeated_guest"].count()
df_gorup_by_type_year_repated_guest
df_gorup_by_type_year_repated_guest.plot( kind='bar', color=list(my_colors))

df_gorup_by_type_year_type_customer_repated_guest = df.groupby(["hotel","arrival_date_year","customer_type"])["stays_in_weekend_nights"].count()
df_gorup_by_type_year_type_customer_repated_guest
df_gorup =df_gorup_by_type_year_type_customer_repated_guest[df_gorup_by_type_year_type_customer_repated_guest>3500]
df_gorup 
df_gorup.plot( kind='bar', color=list(my_colors))
