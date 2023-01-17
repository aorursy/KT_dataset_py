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
import math

df = pd.read_csv('../input/ether-price-and-volume-dataset/Ether Price and Volume Dataset/4h_data_eth.csv')
# Check attributes   

df.columns
df.head()
# get close price

df['close']
# get volume

df['volume']
# get timestamp

df['date']
# Get log-returns

price = df['close']   #获取收盘价

Return = []           #用于存储每一天的对数收益率

for i in range(len(price)-1):

    Return.append(math.log(price[i+1]) - math.log(price[i]))
print(Return[:5])