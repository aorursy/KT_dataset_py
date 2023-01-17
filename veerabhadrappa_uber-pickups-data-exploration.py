# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
NYPickJul14 = pd.read_csv('../input/uber-raw-data-jul14.csv')
NYPickJul14.columns
NYPickJul14.info()
NYPickJul14.describe()
NYPickJul14.head(5)
NYPickJul14['Date/Time'] = pd.to_datetime(NYPickJul14['Date/Time'], format="%m/%d/%Y %H:%M:%S")

NYPickJul14['DayOfWeekNum'] = NYPickJul14['Date/Time'].dt.dayofweek

NYPickJul14['DayOfWeek'] = NYPickJul14['Date/Time'].dt.weekday_name

NYPickJul14['MonthDayNum'] = NYPickJul14['Date/Time'].dt.day

NYPickJul14['HourOfDay'] = NYPickJul14['Date/Time'].dt.hour
weekdays = NYPickJul14.pivot_table(index=['DayOfWeekNum','DayOfWeek'], values='Base', aggfunc='count')

weekdays.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journey on Week Day');