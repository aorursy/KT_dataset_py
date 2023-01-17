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
df_orig = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')
df_orig
#get subsample of data
df = df_orig.sample(frac=0.2,axis=0)
df.columns
df_no_outliers = df_orig[df_orig['price'] <= 100000]
# correlation between id and price?
import seaborn as sb
import matplotlib.pyplot as plt
ax = sb.scatterplot(x="model", y="price", data=(df_orig))
df.drop(columns = ['id',
                   'url', 
                   'region', 
                   'region_url', 
                   'title_status', 
                   'size', 
                   'description', 
                   'vin', 
                   'lat', 
                   'long', 
                   'image_url',
                   'county',
                   'state',
                    'model'])
from sklearn.model_selection import train_test_split
y = df['price']
X = df.drop(columns=['price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
pd.get_dummies(df, columns=['boro'])