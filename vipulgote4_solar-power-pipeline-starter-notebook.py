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
!pip install pycaret==1.0.0
import seaborn as sns 

import matplotlib.pyplot as plt

import missingno

import sklearn

import xgboost

import lightgbm

import catboost

import pycaret





from pycaret.regression import *
data=pd.read_csv('/kaggle/input/solar-power-generation/BigML_Dataset_5f50a4cc0d052e40e6000034.csv')

data
data.isna().sum()
data.info()
data.columns
reg = setup(data = data, 

             target = 'Power Generated',

             numeric_imputation = 'mean',

            ignore_features = ['Day of Year', 'Year', 'Month', 'Day'],

            categorical_features = ['Is Daylight'],

             normalize = True,

             silent = True)
compare_models()
et = create_model('et')
tuned_et = tune_model('et')
plot_model(estimator = tuned_et, plot = 'feature')
interpret_model(tuned_et)