import numpy as np

import pandas as pd

import random

import missingno as msno 

from sklearn.model_selection import KFold

import lightgbm as lgb

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/ncctg-lung-cancer-data/cancer.csv')

df.head()
msno.bar(df) 
msno.heatmap(df) 
msno.matrix(df)