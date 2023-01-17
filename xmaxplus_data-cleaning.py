# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# 1.Import necessary modules and dataset

import pandas as pd

hotel_bookings = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
# 2.Quick check on how many row/columns, whether have missing values, format of datetime

print(hotel_bookings.shape)

print('---------')

print(hotel_bookings.isnull().sum())

hotel_bookings['reservation_status_date_parsed'] = pd.to_datetime(hotel_bookings['reservation_status_date'], format = '%Y-%m-%d')

hotel_bookings.sample(10)
