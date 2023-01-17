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
#importing packages
import pandas as pd
import numpy as np
#reading a file
Train = pd.read_csv("../input/bda-2019-ml-test/Train_Mask.csv")
Train.isnull().sum()
Train.describe()
Train.head()
Train.tail()
Train.info()
#importing package
import seaborn as sns
# finding out the outliers
sns.boxplot(x=Train['motorTempFront'])
#finding out the outliers
sns.boxplot(x=Train['motorTempBack'])
Train['motorTempFront'].describe()
Train['motorTempBack'].describe()
Train.shape
Train_New = Train[(Train.motorTempFront<=49) & (Train.motorTempBack<=40)]
Train_New.shape
sns.boxplot(Train_New['motorTempBack'])
sns.boxplot(Train_New['motorTempFront'])
Train_New.corr()
import seaborn as sb
import scipy
from scipy.stats import spearmanr
back = Train_New["motorTempBack"]
front = Train_New["motorTempFront"]
spearmanr_coff,p_value = spearmanr(back,front)
spearmanr_coff
sb.countplot(x= "flag", data = Train_New, palette = "hls")
Test = pd.read_csv("../input/bda-2019-ml-test/Test_Mask_Dataset.csv")
Test.describe()
Test.isnull().sum()
Test.head()
Test.tail()
Test.info()
Test.corr()
x_train = Train[['currentBack','currentFront','motorTempBack','motorTempFront']]
y_train = Train[['flag']]
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(x_train, y_train)
print(regressor.score(x_train,y_train))
y_pred = regressor.predict(x_test)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
x_test = Test[['currentBack','currentFront','motorTempBack','motorTempFront']]
y_test = Test[['timeindex']]
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(x_test, y_test)
y_predict = regressor.predict(x_test)
y_predict
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)
from sklearn.ensemble import RandomForestClassifier

regressor = RandomForestClassifier(n_estimators=60,criterion="entropy",random_state=42)
regressor.fit(x_train, y_train)
sample = pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv")
sample.to_csv("submit_2.csv",index=False)