# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
temp_time = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')
temp_time_it = temp_time[temp_time.Country=='Italy']
print(len(temp_time_it))
print(type(temp_time_it.index))

temp_time_it.index = pd.DatetimeIndex(temp_time_it['dt'])
#temp_time_it['AverageTemperature'].plot()
#try grouper to group by year (A)
temp_time_it_5Y = temp_time_it.resample('5A').mean()
temp_time_it_5Y['AverageTemperature'].plot()             
          




