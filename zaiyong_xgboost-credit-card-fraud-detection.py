# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split
df=pd.read_csv('../input/creditcard.csv')

train_df,test_df=train_test_split(df,test_size=0.2,stratify=df['Class'])

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_df.iloc[:,0:30], train_df['Class'])
from sklearn.metrics import f1_score,accuracy_score

predictions = gbm.predict(test_df.iloc[:,0:30])

f1=f1_score(test_df['Class'],predictions)

acc=accuracy_score(test_df['Class'],predictions)
acc