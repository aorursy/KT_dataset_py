# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from imblearn.over_sampling import RandomOverSampler,BorderlineSMOTE,SMOTE, ADASYN

from imblearn.under_sampling import RandomUnderSampler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")



y = train.filter(regex='target')

X = train.filter(regex='var.+')

y = np.array(y)

y = y.reshape(-1)

rus = RandomUnderSampler(random_state=0)

X_resampled, y_resampled = rus.fit_resample(X, y)

X_df = pd.DataFrame(X_resampled,columns=X.columns)

X_df['target'] = y_resampled

X_df.to_csv('train_rus.csv',index=False)
ros = RandomOverSampler(random_state=0)

X_resampled, y_resampled = ros.fit_resample(X, y)

X_df = pd.DataFrame(X_resampled,columns=X.columns)

X_df['target'] = y_resampled

X_df.to_csv('train_ros.csv',index=False)