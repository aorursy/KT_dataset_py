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
data = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

data.head()
data.columns
features = ['age','creatinine_phosphokinase','ejection_fraction','serum_creatinine','time']

X = pd.get_dummies(data[features])

y = data["DEATH_EVENT"]
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
from xgboost import XGBRFClassifier

from sklearn.ensemble import GradientBoostingClassifier



model_1 = XGBRFClassifier(max_depth=3,random_state=256)

model_1.fit(X_train, y_train,verbose=True)



model_3 = GradientBoostingClassifier(max_depth=2, random_state=4)

model_3.fit(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier



model_2= RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1,verbose=True)

model_2.fit(X_train, y_train)



from sklearn.metrics import mean_absolute_error, accuracy_score



preds_1 = model_1.predict(X_valid)

acc_1 = accuracy_score(y_valid,preds_1)



preds_2 = model_2.predict(X_valid)

acc_2 = accuracy_score(y_valid,preds_2)



preds_3 = model_3.predict(X_valid)

acc_3 = accuracy_score(y_valid,preds_3)

                            

print("MEA for XGBoost is {:.1f}".format(acc_1*100))

print("MEA for RandomForest is {:.1f}".format(acc_2*100))

print("MEA for GradientBoost is {:.1f}".format(acc_3*100))
output = pd.DataFrame({'Age': X_valid.age, 'time':X_valid.time, 'XGBR': preds_1, 'RandomForest': preds_2, 'GradientBoost': preds_3, 'Ground Truth': y_valid})

output