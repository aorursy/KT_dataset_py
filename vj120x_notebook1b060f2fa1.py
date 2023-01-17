# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Loading the Data
file_path = "../input/hotel-booking-demand/hotel_bookings.csv"
df = pd.read_csv(file_path)

#Print random 10 rows of the dataset
df.sample(10)
#shape of the dataset
df.shape
#getting all the columns with their data type inside the dataset
df.dtypes
#Check the data for any null values, count the number of null values in each column.
df.isnull().sum()
#Replace the null values in attribute “children” by 0.
df["children"].fillna( 0, inplace=True)
#Replace the null value in attribute “country” by the last not-null value available.
df['country'].ffill(inplace=True)
#Replace the null values in attribute “agent” by mean of that column.
df['agent'].fillna(df['agent'].mean(), inplace=True)
#Check how many different “hotel” and “customer” types are there.
df['hotel'].unique()
df['customer_type'].unique()
#Plotting 'is_cancelled' attribute using seaborn
import seaborn as sns
sns.countplot(df.is_canceled)
#Using seaborn pairplot
sns.pairplot(df)