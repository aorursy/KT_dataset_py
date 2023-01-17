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
import pandas as pd





sub_lgbm_sk = pd.read_csv('../input/python-kaggle-start-book-ch02-07/submission_lightgbm_skfold.csv')

sub_lgbm_ho = pd.read_csv('../input/python-kaggle-start-book-ch02-07/submission_lightgbm_holdout.csv')

sub_rf = pd.read_csv('../input/python-kaggle-start-book-ch02-05/submission_randomforest.csv')
sub_lgbm_sk.head()
# pandas.DataFrame内の列同士の相関を計算する corr()を利用



# DataFrameにまとめる

df = pd.DataFrame({

    'sub_lgbm_sk': sub_lgbm_sk['Survived'].values,

    'sub_lgbm_ho': sub_lgbm_ho['Survived'].values,

    'sub_rf': sub_rf['Survived'].values

                  })



df.head()
# 相関をだす

df.corr()
sub = pd.read_csv('../input/titanic/gender_submission.csv')

sub['Survived'] = sub_lgbm_sk['Survived'] + sub_lgbm_ho['Survived'] + sub_rf['Survived']

sub.head()
sub['Survived'] = (sub['Survived'] >= 2).astype(int)

sub.to_csv('submission_lightgbm_ensemble.csv', index=False)

sub.head()