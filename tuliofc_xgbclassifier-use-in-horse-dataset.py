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
horse_data = pd.read_csv("/kaggle/input/horse-colic/horse.csv")

horse_data
horse_data.dtypes
display(pd.DataFrame(horse_data.isna().sum()))
from sklearn.preprocessing import LabelEncoder

from category_encoders import TargetEncoder



encoder = LabelEncoder()



y = pd.DataFrame({'outcome' : encoder.fit_transform(horse_data['outcome'])})



encoder1 = TargetEncoder(handle_missing='return_nan',return_df=True)



horse_enc = encoder1.fit_transform(horse_data,y)



horse_enc
from sklearn.impute import KNNImputer



imputer = KNNImputer(n_neighbors=5,weights='uniform')



horse_enc = pd.DataFrame(imputer.fit_transform(horse_enc),columns=horse_data.columns)

    

horse_enc['outcome'] = horse_enc['outcome'].astype(np.uint8)

horse_enc.dtypes
display(horse_enc.corr().abs()['outcome'].sort_values())
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



data = horse_enc.drop(['nasogastric_reflux_ph','outcome','rectal_temp','nasogastric_tube','lesion_3','pain'],axis=1)

target = horse_enc['outcome']



X_train, X_test, y_train, y_test = train_test_split(data,target, random_state=2020)



xgb = XGBClassifier(n_estimators=1000,learning_rate=0.025,random_state=2020)



xgb.fit(X_train,y_train,early_stopping_rounds=10,eval_set=[(X_test,y_test)])
y_pred = xgb.predict(X_test)



print('Accuracy: ',accuracy_score(y_test, y_pred))