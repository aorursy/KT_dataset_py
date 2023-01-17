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
%load_ext autoreload

%autoreload 2



%matplotlib inline
from fastai.imports import *

# from fastai.structured import *



from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics



from IPython.display import display

! which fastai

PATH_GROCERY = "../input/favorita-grocery-sales-forecasting/"
!ls {PATH_GROCERY}
types = {'id':'int64', 'date':'int32', 'store_nbr':'int8', 'unit_sales':'float64', 'onpromotion':'object'}
%%time

df_grocery_all = pd.read_csv(f'{PATH_GROCERY}train.csv', dtype=types, parse_dates=['date'], infer_datetime_format=True)

df_grocery_all.head()
df_grocery_all.shape
df_grocery_all.onpromotion.fillna(value=False, inplace=True)

df_grocery_all.onpromotion = df_grocery_all.onpromotion.map({'False':False, 'True':True})

df_grocery_all.onpromotion = df_grocery_all.onpromotion.astype(bool)



%time df_grocery_all.to_feather('data/favorita-grocery-sales-forecasting/raw_groceries')
%time df_grocery_all.describe