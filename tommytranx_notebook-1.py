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
pastLoans = pd.read_csv('../input/lending-game/PastLoans_2.csv')
loansApplications = pd.read_csv('../input/lending-game/LoanApplications_2.csv')
pastLoans.default.value_counts()
pastLoans.head()
print(pastLoans.columns,'\n')
print(pastLoans.employment.value_counts())
print(pastLoans.marital.value_counts())
print(loansApplications.columns,'\n')
print(loansApplications.employment.value_counts())
print(loansApplications.marital.value_counts())
pastLoans.head()
sns.heatmap(pastLoans[[ 'marital', 'income', 'facebook', 'default']].corr(), annot = True)
#same old stuff
# Both datasets are balanced and have the same distrib accross vairables
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#one hot + normalize


#classic stuff
#To do later

# k = rf + rp