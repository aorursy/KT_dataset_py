# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
lemon_df = pd.read_csv("../input/Lemonade.csv")
lemon_df.head()
#-------------------Average Lemonades sales -------------------------------------
print(round(lemon_df.Sales.mean()))
lemon_df.describe()
#-------------------Records whose sales are lower than average -------------------------------------
lemon_df.loc[lemon_df.Sales < 25]
#-------------------scatter plot of sales and temperature. -------------------------------------
lemon_df.plot.scatter(x = 'Sales', y= 'Temperature', c= 'DarkBlue')
#-------------------the bar chart of average sales on each day -------------------------------------
Monday = lemon_df.loc[lemon_df.Day == 'Monday']['Sales'].mean()
Tuesday = lemon_df.loc[lemon_df.Day == 'Tuesday']['Sales'].mean()
Wednesday = lemon_df.loc[lemon_df.Day == 'Tuesday']['Sales'].mean()
Thursday = lemon_df.loc[lemon_df.Day == 'Thursday']['Sales'].mean()
Friday = lemon_df.loc[lemon_df.Day == 'Friday']['Sales'].mean()
Saturday = lemon_df.loc[lemon_df.Day == 'Saturday']['Sales'].mean()
Sunday = lemon_df.loc[lemon_df.Day == 'Sunday']['Sales'].mean()
avg_sales_by_days = pd.DataFrame([Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday],columns = ['Average Sales'])
avg_sales_by_days.index = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
avg_sales_by_days.plot(kind='bar', figsize=(5,4), title="Average sales on each day")
avg_sales_by_days
