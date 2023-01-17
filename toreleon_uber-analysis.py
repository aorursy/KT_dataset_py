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

import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
uber_2014_apr=pd.read_csv('/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-apr14.csv',header=0)

uber_2014_may=pd.read_csv('/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-may14.csv',header=0)

uber_2014_jun=pd.read_csv('/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-jun14.csv',header=0)

uber_2014_jul=pd.read_csv('/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-jul14.csv',header=0)

uber_2014_aug=pd.read_csv('/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-aug14.csv',header=0)

uber_2014_sep=pd.read_csv('/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-sep14.csv',header=0)



data = uber_2014_apr.append([uber_2014_may,uber_2014_jun,uber_2014_jul,uber_2014_aug,uber_2014_sep], ignore_index=True)

data.head()
data['Date/Time'] = pd.to_datetime(data['Date/Time'])