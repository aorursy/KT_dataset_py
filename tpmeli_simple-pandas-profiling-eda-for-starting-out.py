# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from pandas_profiling import ProfileReport

from IPython.display import HTML



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# train_sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

# sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

# calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

# submission_file = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
#train_report = ProfileReport(train_sales, title='Sales Train Validation Profiling Report')

#train_report.to_file(output_file="train_report.html")
HTML(filename='/kaggle/input/m5-pandas-profiles-of-sell-prices-calendar/sell_prices_report.html')
#sell_prices_report = ProfileReport(sell_prices, title='Sell Prices Profiling Report')

#sell_prices_report.to_file(output_file="sell_prices_report.html")
HTML(filename='/kaggle/input/m5-pandas-profiles-of-sell-prices-calendar/calendar_report.html')



#calendar_report = ProfileReport(calendar, title='Calendar Profiling Report')

#calendar_report.to_file(output_file="calendar_report.html")