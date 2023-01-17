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
import pandas as pd

import numpy as np

sales_data=pd.read_csv("../input/sales_data_sample.csv",encoding='unicode_escape')
sales_data.shape
sales_data.head()
#Find total sales yearwise



sales_data['YEAR_ID'].value_counts()
sales_data.groupby(['YEAR_ID','QTR_ID']).sum()['SALES']
sales_data.groupby('YEAR_ID').sum()['SALES']
#Find sales Product line wise

(sales_data.groupby('PRODUCTLINE').sum()['SALES']).sort_values()
#Products that they deal in



sales_data['PRODUCTLINE'].value_counts()
#Country wise sales:



(sales_data.groupby('COUNTRY').sum()['SALES']).sort_values()