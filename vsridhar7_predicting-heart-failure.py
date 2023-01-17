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

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.isnull().sum()
df.DEATH_EVENT.value_counts()
%matplotlib inline

df.age.plot(kind='hist')
df.corr()
df[['diabetes','high_blood_pressure','smoking','DEATH_EVENT']].corr()
mms = MinMaxScaler()

df[['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']] = mms.fit_transform(df[['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']])

df['time'].plot(kind='hist')
from sklearn.preprocessing import OneHotEncoder



object_cols = ['anaemia','diabetes','high_blood_pressure','sex','smoking']



OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(df[object_cols]))





# Adding column names to the encoded data set.

OH_cols_train.columns = OH_encoder.get_feature_names(object_cols)





# One-hot encoding removed index; put it back

OH_cols_train.index = df.index





# Remove categorical columns (will replace with one-hot encoding)

num_X_train = df.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

df_data = pd.concat([num_X_train, OH_cols_train], axis=1)

df_data.DEATH_EVENT.value_counts()
pred_var = list(set(df_data.columns)-set(['DEATH_EVENT']))

target = 'DEATH_EVENT'
df_predict = df_data[pred_var]

y = df_data[target]
X_train, X_test, y_train, y_test = train_test_split(df_predict, y, test_size=0.2,random_state=0)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)

m = GradientBoostingClassifier(n_estimators=700, max_depth = 5,max_features=3, random_state = 0).fit(X_train,y_train)

pred =m.predict(X_test)
print("Accuracy of this model is {:f}".format(accuracy_score(pred,y_test)))