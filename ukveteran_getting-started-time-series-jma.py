import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')

%matplotlib inline
input_df = pd.read_csv("../input/website-traffic/website-traffic.csv")
input_df['date_of_visit'] = pd.to_datetime(input_df.MonthDay.str.cat(input_df.Year.astype(str), sep=' '))

input_df[['date_of_visit','Visits']].head(10)
input_df.plot(x='date_of_visit',
              y='Visits', 
              title= "Website Visits per Day")
ts_visits = pd.Series(input_df.Visits.values
                      ,index=pd.date_range(
                                            input_df.date_of_visit.min()
                                            , input_df.date_of_visit.max()
                                            , freq='D')
                     )

deompose = seasonal_decompose(ts_visits.interpolate(),
                                freq=24)
deompose.plot()
input_df['moving_average'] = input_df['Visits'].rolling(window=3,
                                                        center=False).mean()

input_df[['Visits','moving_average']].head(10)
plt.plot(input_df.Visits,'-',color='black',alpha=0.3)
plt.plot(input_df.moving_average,color='b')
plt.title('Website Visit and Moving Average Smoothening')
plt.legend()
input_df['ewma'] = input_df['Visits'].ewm(halflife=3,
                                            ignore_na=False,
                                            min_periods=0,
                                            adjust=True).mean()

plt.plot(input_df.Visits,'-',color='black',alpha=0.3)
plt.plot(input_df.ewma,color='g')
plt.title('Website Visit and Exponential Smoothening')
plt.legend()