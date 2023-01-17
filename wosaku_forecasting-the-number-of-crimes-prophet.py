import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib auto

from fbprophet import Prophet
# Importing the data

df = pd.read_csv('../input/crime.csv')

df.head()
# Using pandas function to_datetime to convert it to a datetime data type

df['DATE'] = pd.to_datetime({'year':df['YEAR'], 'month':df['MONTH'], 'day':df['DAY']})
# Group the data by date

df = df.groupby('DATE').count()['TYPE'].to_frame()
# The input must be a data frame with two columns 'ds' and 'y' 

# (ds is the date and y is the number of crimes). Let's adjust it.

df.reset_index(inplace=True)

df.columns = ['ds','y']

df.head()
# Starting by making a copy of the data frame

df_m1 = df.copy()
# The only transformation we'll do is a 'log transformation' of y

df_m1['y'] = np.log(df_m1['y'])
# Prophet code are basically these lines

m1_plain = Prophet()

m1_plain.fit(df_m1)



# Let's try a forecast for 365 days

future = m1_plain.make_future_dataframe(periods=365)

forecast_m1 = m1_plain.predict(future)
m1_plain.plot(forecast_m1)
m1_plain.plot_components(forecast_m1)
# First, let's get some useful variables: "y" for the actual value and "n" for the number of observations.

y = df['y'].to_frame()

y.index = df['ds']

n = np.int(y.count())
# The forecast is 'log transformed', so we need to 'inverse' it back by using the exp

forecast_m1_exp = np.exp(forecast_m1[['yhat','yhat_lower','yhat_upper']])

forecast_m1_exp.index = forecast_m1['ds']
# Now let's calculate the MAPE for m1

error = forecast_m1_exp['yhat'] - y['y']

MAPE_m1 = (error/y['y']).abs().sum()/n *100

round(MAPE_m1,2)
# Make another copy of the data frame as m2

df_m2 = df.copy()
# Define the Upper Control Limit and Lower Control Limit as 3 standard deviations from the mean

ucl = df_m2.mean() + df_m2.std()*3

lcl = df_m2.mean() - df_m2.std()*3
# Print the number of outliers found

print('Above 3 standard deviations: ', df_m2[df_m2['y'] > ucl['y']]['y'].count(), 'entries')

print('Below 3 standard deviations: ', df_m2[df_m2['y'] < lcl['y']]['y'].count(), 'entries')
# Remove them by setting their value to None. Prophet says it can handle null values.

df_m2.loc[df_m2['y'] > ucl['y'], 'y'] = None

df_m2.loc[df_m2['y'] < lcl['y'], 'y'] = None
# Log transformation

df_m2['y'] = np.log(df_m2['y'])
# Run Prophet using model 2

m2_no_outlier = Prophet()

m2_no_outlier.fit(df_m2)

future = m2_no_outlier.make_future_dataframe(periods=365)

forecast_m2 = m2_no_outlier.predict(future)
# Inverse the log

forecast_m2_exp = np.exp(forecast_m2[['yhat','yhat_lower','yhat_upper']])

forecast_m2_exp.index = forecast_m2['ds']
# Calculate the error

error = forecast_m2_exp['yhat'] - y['y']

MAPE_m2 = (error/y['y']).abs().sum()/n *100

round(MAPE_m2,2)
holidays_0 = pd.DataFrame({

        'holiday': '0 window',

        'ds' :pd.to_datetime(

            ['2003-05-11','2004-05-09','2005-05-08','2006-05-14','2007-05-13','2008-05-11','2009-05-10','2010-05-09','2011-05-08','2012-05-13','2013-05-12','2014-05-11','2015-05-10','2016-05-08','2017-05-14','2018-05-13','2019-05-12','2020-05-10','2003-05-19','2004-05-24','2005-05-23','2006-05-22','2007-05-21','2008-05-19','2009-05-18','2010-05-24','2011-05-23','2012-05-21','2013-05-20','2014-05-19','2015-05-18','2016-05-23','2017-05-22','2018-05-21','2019-05-20','2020-05-18','2003-07-01','2004-07-01','2005-07-01','2006-07-01','2007-07-01','2008-07-01','2009-07-01','2010-07-01','2011-07-01','2012-07-01','2013-07-01','2014-07-01','2015-07-01','2016-07-01','2017-07-01','2018-07-01','2019-07-01','2020-07-01','2003-09-01','2004-09-06','2005-09-05','2006-09-04','2007-09-03','2008-09-01','2009-09-07','2010-09-06','2011-09-05','2012-09-03','2013-09-02','2014-09-01','2015-09-07','2016-09-05','2017-09-04','2018-09-03','2019-09-02','2020-09-07','2003-11-11','2004-11-11','2005-11-11','2006-11-11','2007-11-11','2008-11-11','2009-11-11','2010-11-11','2011-11-11','2012-11-11','2013-11-11','2014-11-11','2015-11-11','2016-11-11','2017-11-11','2018-11-11','2019-11-11','2020-11-11','2003-12-25','2004-12-25','2005-12-25','2006-12-25','2007-12-25','2008-12-25','2009-12-25','2010-12-25','2011-12-25','2012-12-25','2013-12-25','2014-12-25','2015-12-25','2016-12-25','2017-12-25','2018-12-25','2019-12-25','2020-12-25']),

        'lower_window' : 0,

        'upper_window' : 0,       

    })



holidays_1 = pd.DataFrame({

        'holiday': '1 window',

        'ds' :pd.to_datetime(

            ['2003-10-31','2004-10-31','2005-10-31','2006-10-31','2007-10-31','2008-10-31','2009-10-31','2010-10-31','2011-10-31','2012-10-31','2013-10-31','2014-10-31','2015-10-31','2016-10-31','2017-10-31','2018-10-31','2019-10-31','2020-10-31','2003-01-01','2004-01-01','2005-01-01','2006-01-01','2007-01-01','2008-01-01','2009-01-01','2010-01-01','2011-01-01','2012-01-01','2013-01-01','2014-01-01','2015-01-01','2016-01-01','2017-01-01','2018-01-01','2019-01-01','2020-01-01']),

        'lower_window' : -1,

        'upper_window' : 1,       

    })



holidays_2 = pd.DataFrame({

        'holiday': '2 window',

        'ds' :pd.to_datetime(

            ['2003-08-04','2004-08-02','2005-08-01','2006-08-07','2007-08-06','2008-08-04','2009-08-03','2010-08-02','2011-08-01','2012-08-06','2013-08-05','2014-08-04','2015-08-03','2016-08-01','2017-08-07','2018-08-06','2019-08-05','2020-08-03','2003-10-13','2004-10-11','2005-10-10','2006-10-09','2007-10-08','2008-10-13','2009-10-12','2010-10-11','2011-10-10','2012-10-08','2013-10-14','2014-10-13','2015-10-12','2016-10-10','2017-10-09','2018-10-08','2019-10-14','2020-10-12']),

        'lower_window' : -2,

        'upper_window' : 1,       

    })



# Concatenate all 3 df into 1

holidays_list = pd.concat((holidays_0, holidays_1, holidays_2))
# Now we pass the holidays variable when we instantiate Prophet

m3_holidays = Prophet(holidays=holidays_list)

m3_holidays.fit(df_m1)

future = m3_holidays.make_future_dataframe(periods=365)

forecast_m3 = m3_holidays.predict(future)
# Inverse the log

forecast_m3_exp = np.exp(forecast_m3[['yhat','yhat_lower','yhat_upper']])

forecast_m3_exp.index = forecast_m3['ds']
# Calculate error

error = forecast_m3_exp['yhat'] - y['y']

MAPE_m3 = (error/y['y']).abs().sum()/n *100

round(MAPE_m3,2)
print('M1:', round(MAPE_m1,2), '--> Plain','\n')

print('M2:', round(MAPE_m2,2), '--> Without outliers','\n')

print('M3:', round(MAPE_m3,2),'--> Plain with holidays','\n')
m3_holidays.plot(forecast_m3)
start = '2017-09-01'

end = '2017-09-05'

forecast_m3_exp[(forecast_m3_exp.index >= start) & (forecast_m3_exp.index <= end)].astype(int)